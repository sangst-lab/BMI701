from torch import nn


class QuickStartNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    import torch

    classes_fashion_mnist = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model = QuickStartNeuralNetwork(num_classes=10)

    # Create a fake batch: 64 grayscale images, each image is 28 x 28
    x = torch.randn(64, 1, 28, 28)

    # Forward pass
    logits = model(x)
    predicted_class = logits[0].argmax().item()

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Predicted class index:", predicted_class)
    print("Predicted class name:", classes_fashion_mnist[predicted_class])