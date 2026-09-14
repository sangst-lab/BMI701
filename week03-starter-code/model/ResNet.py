import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if use_skip and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)  # The essential residual connection.
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        # Original ResNet: downsample in the first 1x1 convolution.
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if use_skip and (stride != 1 or in_channels != out_channels * 4):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=10, depth=18, use_skip=True):
        super().__init__()
        if depth == 18:
            block, counts = BasicBlock, [2, 2, 2, 2]
        elif depth == 34:
            block, counts = BasicBlock, [3, 4, 6, 3]
        elif depth == 50:
            block, counts = Bottleneck, [3, 4, 6, 3]
        else:
            raise ValueError('Choose ResNet depth 18, 34, or 50.')
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self.make_layer(block, 64, counts[0], 1, use_skip)
        self.layer2 = self.make_layer(block, 128, counts[1], 2, use_skip)
        self.layer3 = self.make_layer(block, 256, counts[2], 2, use_skip)
        self.layer4 = self.make_layer(block, 512, counts[3], 2, use_skip)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def make_layer(self, block, out_channels, count, stride, use_skip):
        layers = [block(self.in_channels, out_channels, stride, use_skip)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, count):
            layers.append(block(self.in_channels, out_channels, 1, use_skip))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


if __name__ == '__main__':
    model = ResNet(depth=18)
    x = torch.randn(2, 3, 224, 224)
    print(model)
    print('Input:', x.shape, 'Output:', model(x).shape)
