import torch
import torch.nn as nn
from model.Blocks import ConvolutionBlock, LinearBlock


class GateClassificationModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        """
        Create a model based on convolutions which will predict the location of the gate (code)

        :param in_channels: number of color channels in images
        :param out_features: number of coordinates to predict
        """

        super(GateClassificationModel, self).__init__()
        self.sequential = nn.Sequential(
            ConvolutionBlock(in_channels, 8),
            ConvolutionBlock(8, 16),
            nn.Flatten(),
            LinearBlock(4368, 1024),
            LinearBlock(1024, 256),
            LinearBlock(256, 64),
            LinearBlock(64, 16),
            LinearBlock(16, out_features)
        )

    def forward(self, x):
        return self.sequential(x)
