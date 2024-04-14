"""
File for define model structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvAEN(nn.Module):
    def __init__(self):
        super(ConvAEN, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(8)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(16)
        self.enc_conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Encoder
        self.enc_dense = nn.Linear(32 * 16 * 16, 128)

        # Decoder
        self.dec_dense1 = nn.Linear(128, 32 * 16 * 16)
        self.dec_reshape = lambda x: x.view(-1, 32, 16, 16)
        self.dec_convtrans1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsamples to 32x32
        self.dec_bn4 = nn.BatchNorm2d(16)
        self.dec_convtrans2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsamples to 64x64
        self.dec_convtrans3 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsamples to 128x128
        self.dec_bn5 = nn.BatchNorm2d(8)
        self.dec_conv4 = nn.Conv2d(8, 3, kernel_size=3, padding=1)  # Output size is 3x128x128


    def forward(self, x):
        # Encoder
        x = self.pool(F.leaky_relu(self.enc_bn1(self.enc_conv1(x))))
        x = self.pool(F.leaky_relu(self.enc_bn2(self.enc_conv2(x))))
        x = self.pool(F.leaky_relu(self.enc_bn3(self.enc_conv3(x))))
        x = self.flatten(x)
        x = F.leaky_relu(self.enc_dense(x))

        # Decoder
        x = F.leaky_relu(self.dec_dense1(x))
        x = self.dec_reshape(x)
        x = F.leaky_relu(self.dec_bn4(self.dec_convtrans1(x)))
        x = F.leaky_relu(self.dec_bn5(self.dec_convtrans2(x)))
        x = self.dec_convtrans3(x)
        x = torch.tanh(self.dec_conv4(x))

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x))