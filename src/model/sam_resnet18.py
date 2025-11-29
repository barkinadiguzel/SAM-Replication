import torch
import torch.nn as nn
from src.layers.conv_layer import conv3x3
from src.layers.fc_layer import FCLayer
from src.layers.pool_layers.avgpool_layer import AdaptiveAvgPool
from src.blocks.resnet_block import BasicBlock
from src.optimizers.sam_optimizer import SAMOptimizer

class SAMResNet18(nn.Module):
    def __init__(self, num_classes=1000, rho=0.05):
        super(SAMResNet18, self).__init__()
        self.in_channels = 64
        # Stem: Conv + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-18 blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Global average pool + FC
        self.avgpool = AdaptiveAvgPool((1, 1))
        self.fc = FCLayer(512, num_classes)

        # SAM optimizer wrapper
        self.rho = rho
        self.sam_optimizer = None  

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
