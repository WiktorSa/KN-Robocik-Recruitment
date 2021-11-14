import torch
import torch.nn as nn
from model.Blocks import ConvolutionBlock, LinearBlock


class GateRegressionModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        """
        Create a model based on convolutions which will predict the coordinates of the gate

        :param in_channels: number of color channels in images
        :param out_features: number of coordinates to predict
        """

        super(GateRegressionModel, self).__init__()
        self.sequential = nn.Sequential(
            ConvolutionBlock(in_channels, 4),
            ConvolutionBlock(4, 8),
            nn.Flatten(),
            LinearBlock(38016, 4096),
            LinearBlock(4096, 256),
            LinearBlock(256, 64),
            LinearBlock(64, out_features)
        )

    def forward(self, x):
        return self.sequential(x)
