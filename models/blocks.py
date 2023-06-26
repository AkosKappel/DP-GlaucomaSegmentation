import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unsupported activation function: {activation}')


class SingleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True, act: str = 'relu'):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None
        self.act = get_activation(act)

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        out = self.act(out)
        return out


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBatchAct(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False, bn: bool = True, act: str = 'relu'):
        super(ConvBatchAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None
        self.act = get_activation(act)

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        out = self.act(out)
        return out


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2,
                 mode: str = 'bilinear', align_corners: bool = False):
        super(UpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True,
                 relu_before: bool = True, shortcut_conv: bool = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

        self.bn = bn
        self.relu_before = relu_before
        self.shortcut_conv = shortcut_conv

        if self.bn:
            self.batch_norm1 = nn.BatchNorm2d(out_channels)
            self.batch_norm2 = nn.BatchNorm2d(out_channels)

        if self.shortcut_conv:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        residual = x

        # 1x1 convolution layer for shortcut
        if self.shortcut_conv:
            residual = self.conv1x1(residual)

        # first convolution block
        out = self.conv1(x)
        if self.bn:
            out = self.batch_norm1(out)
        out = F.relu(out)

        # second convolution block
        out = self.conv2(out)
        if self.bn:
            out = self.batch_norm2(out)
        # activation before residual shortcut
        if self.relu_before:
            out = F.relu(out)

        # residual shortcut (identity mapping)
        out += residual

        # activation after residual shortcut
        if not self.relu_before:
            out = F.relu(out)

        return out


class RecurrentBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t: int = 2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1 = ConvBatchAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = ConvBatchAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        # 1x1 convolution to set correct number of channels for recurrent blocks
        out = self.conv1x1(x)

        # first recurrent block
        for i in range(self.t):

            # one pass is done before the recursion loop
            if i == 0:
                out = self.conv1(out)

            out = self.conv1(out + x)

        # second recurrent block
        for i in range(self.t):
            if i == 0:
                out = self.conv2(out)
            out = self.conv2(out + x)

        return out


class RecurrentResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t: int = 2):
        super(RecurrentResidualBlock, self).__init__()
        self.t = t
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1 = ConvBatchAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = ConvBatchAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        # 1x1 convolution to set correct number of channels for recurrent blocks
        out = self.conv1x1(x)

        # identity mapping
        residual = out

        # first recurrent block
        for i in range(self.t):

            # one pass is done before the recursion loop
            if i == 0:
                out = self.conv1(out)

            out = self.conv1(out + x)

        # second recurrent block
        for i in range(self.t):
            if i == 0:
                out = self.conv2(out)
            out = self.conv2(out + x)

        return out + residual
