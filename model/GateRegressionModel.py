import torch
import torch.nn as nn
from model.Blocks import ConvolutionBlock, LinearBlock


class GateRegressionModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        """
        Create a model based on convolutions which will predict the location of the gate (coordinates)

        :param in_channels: number of color channels in images
        :param out_features: number of coordinates to predict
        """

        super(GateRegressionModel, self).__init__()
        self.sequential = nn.Sequential(
            ConvolutionBlock(in_channels, 16),
            ConvolutionBlock(16, 32),
            nn.Flatten(),
            LinearBlock(8736, 4096),
            LinearBlock(4096, 1024),
            LinearBlock(1024, 256),
            LinearBlock(256, 64),
            LinearBlock(64, 16),
            LinearBlock(16, out_features)
        )

    def forward(self, x):
        return self.sequential(x)
