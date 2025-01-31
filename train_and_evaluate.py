from src.utils import *
from model.resnet_18 import *
from torch import nn, optim
import torch

config = load_config("config.yaml")
model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

def train():
    train_loader = load_data_loader(config["train_loader_dir"])
    test_loader = load_data_loader(config["test_loader_dir"])
    classes = config["classes"]

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in tqdm(range(config["epochs"]), desc="Training"):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100. * correct / total)

        model.eval()
        test_loss, correct, total = 0, 0, 0

        # Prepare per-class accuracy tracking
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predictions = torch.max(outputs, 1)

                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()

                # Count correct predictions per class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100. * correct / total)

        print(f"Epoch {epoch+1}/{config['epochs']}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    torch.save(model.state_dict(), "model/model.pth")

    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname:5s}: {accuracy:.1f} %')

    # Generate plots
    plot_metrics(train_accuracies, test_accuracies, "Accuracy")
    plot_metrics(train_losses, test_losses, "Loss")

if __name__ == "__main__":
    train()
