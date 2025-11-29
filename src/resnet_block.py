import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_relu=True):
        super(BasicBlock, self).__init__()
        self.use_relu = use_relu

        # first conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.use_relu:
            out = self.relu(out)
        return out
