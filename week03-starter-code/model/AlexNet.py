"""AlexNet: five convolutional layers + three fully connected layers."""
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Input: [batch, 3, 224, 224]
        # Layer 1: 3 -> 96 channels, image size 224 -> 55 -> 27.
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        # PyTorch divides alpha by size: 5e-4 / 5 = the paper's 1e-4.
        self.lrn1 = nn.LocalResponseNorm(5, alpha=5e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2: 96 -> 256 channels, image size 27 -> 13.
        # groups=2 keeps the original AlexNet connection pattern.
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(5, alpha=5e-4, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3: 256 -> 384 channels, image size stays 13.
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Layer 4: 384 -> 384 channels, image size stays 13.
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU()

        # Layer 5: 384 -> 256 channels, image size 13 -> 6.
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        # Layer 6: 256 * 6 * 6 -> 4096.
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)

        # Layer 7: 4096 -> 4096.
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)

        # Layer 8: 4096 -> number of classes.
        self.fc8 = nn.Linear(4096, num_classes)

        # Weight initialization only; all layers are already listed above.
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4,
                      self.conv5, self.fc6, self.fc7, self.fc8]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.zeros_(layer.bias)
        for layer in [self.conv2, self.conv4, self.conv5, self.fc6, self.fc7]:
            nn.init.ones_(layer.bias)

    def forward(self, x):
        # Layer 1.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        # Layer 2.
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        # Layer 3.
        x = self.conv3(x)
        x = self.relu3(x)

        # Layer 4.
        x = self.conv4(x)
        x = self.relu4(x)

        # Layer 5.
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.flatten(x)

        # Layer 6.
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        # Layer 7.
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)

        # Layer 8. Return logits directly to CrossEntropyLoss.
        x = self.fc8(x)
        return x


if __name__ == '__main__':
    model = AlexNet(num_classes=10)
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
    print(model)
    print('Input:', images.shape)
    print('Output:', logits.shape)
