import torch
import torch.nn as nn
from model.Blocks import ConvolutionBlock, LinearBlock


class GateRegressionModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        """
        Create a CNN model which will predict the location of the gate (coordinates)

        :param in_channels: number of color channels in images
        :param out_features: number of coordinates to predict
        """

        super(GateRegressionModel, self).__init__()
        # Model needs to end with nn.Linear to avoid accidental dropout
        self.sequential = nn.Sequential(
            ConvolutionBlock(in_channels, 16),
            ConvolutionBlock(16, 32),
            nn.MaxPool2d(3),
            ConvolutionBlock(32, 32),
            ConvolutionBlock(32, 32),
            nn.MaxPool2d(3),
            nn.Flatten(),
            LinearBlock(7680, 1024),
            LinearBlock(1024, 256),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        return self.sequential(x)
