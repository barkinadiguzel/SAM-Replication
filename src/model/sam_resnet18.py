import torch
import torch.nn as nn
from src.layers.conv_layer import ConvLayer
from src.layers.flatten_layer import FlattenLayer
from src.layers.fc_layer import FCLayer
from src.layers.pool_layers.maxpool_layer import MaxPoolLayer
from src.layers.pool_layers.avgpool_layer import AvgPoolLayer
from src.blocks.resnet_block import BasicBlock

class SAMResNet18(nn.Module):
    def __init__(self, num_classes=1000, use_relu=True):
        super(SAMResNet18, self).__init__()
        self.use_relu = use_relu

        # Stem
        self.conv1 = ConvLayer(3, 64, kernel_size=7, stride=2, padding=3, use_relu=use_relu)
        self.maxpool = MaxPoolLayer(kernel_size=3, stride=2, padding=1)

        # ResNet18 Block Layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Pool + Flatten + FC
        self.avgpool = AvgPoolLayer(output_size=(1, 1))
        self.flatten = FlattenLayer()
        self.fc = FCLayer(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, use_relu=self.use_relu))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, use_relu=self.use_relu))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
