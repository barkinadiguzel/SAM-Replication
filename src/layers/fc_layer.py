import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
