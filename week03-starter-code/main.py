import os
# Avoid OpenMP conflicts between NumPy and PyTorch on Windows.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path
from itertools import islice

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from data.ImageNet10 import classes_imagenet10, get_dataloaders
from model.AlexNet import AlexNet
from model.VGG import VGG
from model.ResNet34 import ResNet34
from model.ResNet50 import ResNet50
from model.ResNet152 import ResNet152
from utils.visualization import save_predictions


DATA_ROOT = Path(r"E:\data\BMI701\ImageNet10")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "classroom"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "classroom"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01
NUM_WORKERS = 0  # Load data in the main process.
MAX_TRAIN_BATCHES = None  # Use all training batches when set to None.


def evaluate(model, dataloader, loss_fn, device):
    """Return mean loss and accuracy without updating model parameters."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (pred.argmax(1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The validation/test dataset is empty.")
    return total_loss / total, correct / total


def save_training_curves(history):
    """Save training and validation loss and accuracy curves."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", label="Train")
    axes[0].plot(epochs, [r["valid_loss"] for r in history], "o-", label="Validation")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, [100 * r["train_accuracy"] for r in history], "o-", label="Train")
    axes[1].plot(epochs, [100 * r["valid_accuracy"] for r in history], "o-", label="Validation")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_progress.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    # Load the training, validation, and test sets.
    train_loader, valid_loader, test_loader = get_dataloaders(
        root_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=42,
    )

    # Initialize the model.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ResNet34(num_classes=len(classes_imagenet10))
    # model = AlexNet(num_classes=len(classes_imagenet10))
    # model = VGG(num_classes=len(classes_imagenet10))  # VGG-16
    # model = ResNet50(num_classes=len(classes_imagenet10))
    # model = ResNet152(num_classes=len(classes_imagenet10))
    model = model.to(device)
    print(f"Model: {type(model).__name__}; device: {device}")

    # Define the loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4
    )
    history = []
    best_valid_accuracy = -1.0
    best_valid_loss = float("inf")

    # Train and validate at each epoch.
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        num_batches = len(train_loader)
        if MAX_TRAIN_BATCHES is not None:
            num_batches = min(num_batches, MAX_TRAIN_BATCHES)

        for images, labels in tqdm(
            islice(train_loader, num_batches),
            total=num_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}",
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Clear gradients.
            pred = model(images)  # Forward pass.
            loss = loss_fn(pred, labels)
            loss.backward()  # Backpropagate.
            optimizer.step()  # Update parameters.

            total_loss += loss.item() * labels.size(0)
            correct += (pred.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if total == 0:
            raise ValueError("Training needs at least one batch.")
        # Accumulate metrics over augmented training batches.
        train_loss, train_accuracy = total_loss / total, correct / total
        valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_fn, device)

        # Select by validation accuracy, breaking ties with validation loss.
        if valid_accuracy > best_valid_accuracy or (
            valid_accuracy == best_valid_accuracy and valid_loss < best_valid_loss
        ):
            best_valid_accuracy, best_valid_loss = valid_accuracy, valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_accuracy": train_accuracy,
            "valid_loss": valid_loss, "valid_accuracy": valid_accuracy,
        })
        save_training_curves(history)
        print(
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.1%} | "
            f"Validation loss: {valid_loss:.4f}, accuracy: {valid_accuracy:.1%}"
        )

    # Evaluate the best checkpoint on the test set.
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.1%}")

    # Save example predictions.
    save_predictions(
        model, test_loader.dataset, classes_imagenet10,
        device, OUTPUT_DIR / "test_predictions.png",
    )
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Figures: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
