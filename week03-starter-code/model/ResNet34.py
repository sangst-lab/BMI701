"""ResNet-34: explicit residual blocks, original ResNet v1 architecture."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    """Two 3x3 convolutions, followed by shortcut addition and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main branch: learn the residual F(x).
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut branch: keep x, or change its size to match F(x).
        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Main branch: F(x).
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut branch: x or a projection of x.
        if self.shortcut_conv is not None:
            identity = self.shortcut_conv(identity)
            identity = self.shortcut_bn(identity)

        # Residual connection: F(x) + x, then ReLU.
        out = out + identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    """1 stem convolution + 16 blocks x 2 convolutions + 1 FC = 34 layers."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: [batch, 3, 224, 224] -> [batch, 64, 56, 56].
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: 3 basic blocks -> [batch, 64, 56, 56].
        self.stage1_block1 = BasicBlock(64, 64, stride=1)
        self.stage1_block2 = BasicBlock(64, 64, stride=1)
        self.stage1_block3 = BasicBlock(64, 64, stride=1)

        # Stage 2: 4 basic blocks -> [batch, 128, 28, 28].
        self.stage2_block1 = BasicBlock(64, 128, stride=2)
        self.stage2_block2 = BasicBlock(128, 128, stride=1)
        self.stage2_block3 = BasicBlock(128, 128, stride=1)
        self.stage2_block4 = BasicBlock(128, 128, stride=1)

        # Stage 3: 6 basic blocks -> [batch, 256, 14, 14].
        self.stage3_block1 = BasicBlock(128, 256, stride=2)
        self.stage3_block2 = BasicBlock(256, 256, stride=1)
        self.stage3_block3 = BasicBlock(256, 256, stride=1)
        self.stage3_block4 = BasicBlock(256, 256, stride=1)
        self.stage3_block5 = BasicBlock(256, 256, stride=1)
        self.stage3_block6 = BasicBlock(256, 256, stride=1)

        # Stage 4: 3 basic blocks -> [batch, 512, 7, 7].
        self.stage4_block1 = BasicBlock(256, 512, stride=2)
        self.stage4_block2 = BasicBlock(512, 512, stride=1)
        self.stage4_block3 = BasicBlock(512, 512, stride=1)

        # Global average pooling and the final classification layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization only; this loop does not create any layers.
        # Keep the same initialization as the earlier course implementation.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Stem.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1: each line applies one residual block.
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)

        # Stage 2: each line applies one residual block.
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)
        x = self.stage2_block4(x)

        # Stage 3: each line applies one residual block.
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        x = self.stage3_block4(x)
        x = self.stage3_block5(x)
        x = self.stage3_block6(x)

        # Stage 4: each line applies one residual block.
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)
        x = self.stage4_block3(x)

        # Classification head.
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x  # Logits: CrossEntropyLoss already includes log-softmax.


if __name__ == '__main__':
    model = ResNet34(num_classes=10)
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
    print(model)
    print('Input:', images.shape)
    print('Output:', logits.shape)
