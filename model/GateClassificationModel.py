import torch
import torch.nn as nn
from model.Blocks import ConvolutionBlock, LinearBlock


class GateClassificationModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        """
        Create a CNN model which will predict the location of the gate (code)

        :param in_channels: number of color channels in images
        :param out_features: number of coordinates to predict
        """

        super(GateClassificationModel, self).__init__()
        # Model needs to end with nn.Linear to avoid accidental dropout
        self.sequential = nn.Sequential(
            ConvolutionBlock(in_channels, 8),
            ConvolutionBlock(8, 16),
            nn.MaxPool2d(3),
            ConvolutionBlock(16, 16),
            ConvolutionBlock(16, 16),
            nn.MaxPool2d(3),
            nn.Flatten(),
            LinearBlock(3840, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 256),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        return self.sequential(x)
