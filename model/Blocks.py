import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Create a convolution block consisting of:
        1. Conv2d layer
        2. BatchNorm2d layer
        3. ReLU activation function

        :param in_channels: number of channels in the input for convolution
        :param out_channels: number of channels produced by the convolution
        :param kernel_size: kernel size
        :return: one convolution block
        """

        super(ConvolutionBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5):
        """
        Create a linear block consisting of:
        1. Linear layer
        2. ReLU activation function
        3. Dropout layer

        :param in_features: number of inputs
        :param out_features: number of outputs
        :param dropout: value of dropout in dropout layer.
        """

        super(LinearBlock, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.linear_block(x)
