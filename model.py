"""
File for define model structure.
"""

import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    AutoEncoder model with bottleneck size 8.
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * 64 * 64, 1024),  # Adjust the input size
            nn.ReLU(True),
            nn.Linear(1024, 128),  # Adjust the input size
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64,8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh(),
        )  # Adjust the output size

    def forward(self, x):
        """
        Encode the input tensor
        """
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 3, 64, 64)  # Reshape back to original image shape
        return x
