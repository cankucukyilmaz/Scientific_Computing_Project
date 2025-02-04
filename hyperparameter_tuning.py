from src.utils import *
from src.transform import *
from model.resnet_18 import *

import optuna
from torchvision.transforms import v2
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

import warnings
import numpy as np

warnings.filterwarnings("ignore")

config = load_config("config.yaml")

base_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor()
])

def objective(trial):
    # Hyperparameters to tune
    batch_size = trial.suggest_int("batch_size", 4, 32, 4)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Load dataset
    dataset = CustomImageDataset("input", transform=base_transform)

    # Initialize model
    model = ResNet18()

    # Choose optimizer
    optimizers = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    
    match optimizers:
        case "SGD":
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            momentum = trial.suggest_float("momentum", 1e-5, 1, log=True)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        case "Adam":
            beta1 = trial.suggest_float("beta1", 0.8, 0.99)
            beta2 = trial.suggest_float("beta2", 0.9, 0.999)
            epsilon = trial.suggest_float("epsilon", 1e-9, 1e-6, log=True)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
        case "RMSprop":
            alpha = trial.suggest_float("alpha", 0.8, 0.999)
            epsilon = trial.suggest_float("epsilon", 1e-9, 1e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=epsilon, weight_decay=weight_decay)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # K-Fold Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    # Store losses for each fold
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.img_labels)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Create subsets for training and validation
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)

        # Compute mean and std from the training set
        mean, std = mean_std(train_subsampler)

        # Create transformations with normalization
        train_transform, val_transform = create_train_test_transformers(
            mean,
            std,
            config["height"],
            config["width"],
            config["random_rotation_degrees"],
            config["random_affine_degrees"],
            config["random_translation"],
            config["brightness"],
            config["contrast"],
            config["saturation"],
            config["hue"]
        )

        # Apply transformations to the datasets
        train_subsampler.dataset.transform = train_transform
        val_subsampler.dataset.transform = val_transform

        # Create DataLoaders
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        # Training loop
        model.train()
        for epoch in range(1):  # You can increase the number of epochs if needed
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Compute average validation loss for this fold
        avg_val_loss = val_loss / len(val_loader)
        fold_losses.append(avg_val_loss)
        print(f"Validation Loss (Fold {fold + 1}): {avg_val_loss}")

    # Compute the average validation loss across all folds
    avg_loss_across_folds = np.mean(fold_losses)
    print(f"Average Validation Loss: {avg_loss_across_folds}")

    # Return the average validation loss for optimization
    return avg_loss_across_folds

if __name__ == "__main__":
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")