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
warnings.filterwarnings("ignore")

config = load_config("config.yaml")

base_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor()
])

def objective(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128, 16)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3,log=True)

    model = ResNet18()

    optimizers = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    
    match optimizers:
        case "SGD":
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            momentum = trial.suggest_float("momentum", 1e-5, 1, log=True)
            optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay)
        case "Adam":
            beta1 = trial.suggest_float("beta1", 0.8, 0.99)
            beta2 = trial.suggest_float("beta2", 0.9, 0.999)
            epsilon = trial.suggest_float("epsilon", 1e-9, 1e-6, log=True)
            optimizer = optim.Adam(model.parameters(), lr, (beta1, beta2))
        case "RMSprop":
            alpha = trial.suggest_float("alpha", 0.8, 0.999)
            epsilon = trial.suggest_float("epsilon", 1e-9, 1e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
            optimizer = optim.RMSprop(model.parameters(), lr, alpha, epsilon, weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    images = ImageFolder("input",transform=base_transform)

    image_paths = [sample[0] for sample in images.imgs]
    labels = [sample[1] for sample in images.imgs]

    n_splits = 5
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    avg_val_loss = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_subset = Subset(images, train_idx)
        val_subset = Subset(images, val_idx)
        
        mean, std = subset_mean_std(train_subset)

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

        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        train_loader = DataLoader(train_subset, batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size, shuffle=False)

        for epoch in range(1):
            model.train()

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss += val_loss / len(val_loader)
    
    return avg_val_loss / 5

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
