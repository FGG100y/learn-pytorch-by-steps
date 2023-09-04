"""
ResidualBlock: The skip connection in neuralnet
"""
import torch.nn as nn


# residual block:
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # optional Conv2d layer to handle that the number of output channels of
        # the last convolution layer may be different than the number of
        # channels in the input:
        self.downsample = None
        if out_channels != in_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        # first "weight layer" + activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second "weight layer"
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # NOTE that the original input has been modified here;
            # even though it goes somewhat against the idea of learning the
            # identity function, the usefulness of a shortcut still stands.
            identity = self.downsample(identity)
        # adding inputs before activation
        out = out + identity
        out = self.relu(out)

        return out
