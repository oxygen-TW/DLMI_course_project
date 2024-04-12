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
            nn.Linear(3 * 128 * 128, 6400),  # Adjust the input size
            nn.ReLU(True),
            nn.Linear(6400, 1280),  # Adjust the input size
            nn.ReLU(True),
            nn.Linear(1280, 640),
            nn.ReLU(True),
            nn.Linear(640,128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 640),
            nn.ReLU(True),
            nn.Linear(640, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 6400),
            nn.ReLU(True),
            nn.Linear(6400, 3 * 128 * 128),
            nn.Tanh(),
        )  # Adjust the output size

    def forward(self, x):
        """
        Encode the input tensor
        """
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 3, 128, 128)  # Reshape back to original image shape
        return x
