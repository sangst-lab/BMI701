from pathlib import Path
from itertools import islice
import torch
from torch import nn
from tqdm import tqdm


from data.FashionMNIST import classes_fashion_mnist, get_dataloaders
from model.QuickStartModel import QuickStartNeuralNetwork
from utils.visualization import save_training_progress, show_random_predictions


DATA_ROOT = Path(r"E:\data\BMI701\FashionMNIST")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MODEL_PATH = OUTPUT_DIR / "fashion_mnist_model.pth"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
PROGRESS_PANEL_PATH = OUTPUT_DIR / "training_progress.png"
SHOW_PROGRESS_PANEL = False

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 5000

# For a fast classroom demo, use a small number such as 5 or 20.
# For full training, set this to None.
MAX_TRAIN_BATCHES = None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_loader, valid_loader, test_loader = get_dataloaders(
        root_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        validation_size=VALIDATION_SIZE,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = QuickStartNeuralNetwork(num_classes=len(classes_fashion_mnist)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(f"Using {device} device")

    epochs = []
    train_losses = []
    validation_losses = []
    validation_accuracies = []
    best_validation_accuracy = 0
    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_batches = 0
        num_train_batches = len(train_loader)

        if MAX_TRAIN_BATCHES is not None:
            num_train_batches = min(num_train_batches, MAX_TRAIN_BATCHES)

        train_batches_iter = islice(train_loader, num_train_batches)
        for images, labels in tqdm(
            train_batches_iter,
            total=num_train_batches,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
        ):
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        average_train_loss = train_loss / train_batches

        model.eval()
        valid_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)

                valid_loss += loss_fn(pred, labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / len(valid_loader.dataset)

        is_best_model = (
            valid_accuracy > best_validation_accuracy
            or (
                valid_accuracy == best_validation_accuracy
                and valid_loss < best_validation_loss
            )
        )
        if is_best_model:
            best_validation_accuracy = valid_accuracy
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to: {BEST_MODEL_PATH}")

        epochs.append(epoch + 1)
        train_losses.append(average_train_loss)
        validation_losses.append(valid_loss)
        validation_accuracies.append(valid_accuracy)

        save_training_progress(
            epochs=epochs,
            train_losses=train_losses,
            validation_losses=validation_losses,
            validation_accuracies=validation_accuracies,
            output_path=PROGRESS_PANEL_PATH,
            show_panel=SHOW_PROGRESS_PANEL,
        )

        prediction_grid = OUTPUT_DIR / f"epoch_{epoch + 1:02d}_predictions.png"
        show_random_predictions(
            model=model,
            dataset=test_loader.dataset,
            classes=classes_fashion_mnist,
            device=device,
            output_path=prediction_grid,
            n_images=20,
        )

        print(
            f"Validation accuracy: {100 * valid_accuracy:.1f}%, "
            f"validation loss: {valid_loss:.4f}"
        )

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)

            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)

    print("\nFinal test result")
    print(f"Test accuracy: {100 * test_accuracy:.1f}%, test loss: {test_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
