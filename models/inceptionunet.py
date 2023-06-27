import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['InceptionUNet']


class ConvBatchRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False, bn: bool = True):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = F.relu(out)
        return out


class InceptionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, naive: bool = False):
        super(InceptionBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels // 2

        # 1x1 convolutions for dimensionality reduction of branches output
        self.naive = naive
        if not naive:
            self.conv3_1x1 = ConvBatchRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1)
            self.conv5_1x1 = ConvBatchRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1)
            self.pool_1x1 = ConvBatchRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        # parallel branches
        self.conv1 = ConvBatchRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3 = ConvBatchRelu(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = ConvBatchRelu(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, dilation=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # TODO: Work in progress
        self.final_conv = ConvBatchRelu(
            mid_channels * 3 + in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1,
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out_pool = self.pool(x)
        out = torch.cat([out1, out3, out5, out_pool], dim=1)
        out = self.final_conv(out)
        return out


class InceptionUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(InceptionUNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Inception U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.en1 = InceptionBlock(in_channels, features[0])
        self.en2 = InceptionBlock(features[0], features[1])
        self.en3 = InceptionBlock(features[1], features[2])
        self.en4 = InceptionBlock(features[2], features[3])
        self.en5 = InceptionBlock(features[3], features[4])

        self.de1 = InceptionBlock(features[4], features[3])
        self.de2 = InceptionBlock(features[3], features[2])
        self.de3 = InceptionBlock(features[2], features[1])
        self.de4 = InceptionBlock(features[1], features[0])

        self.conv1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Contracting path
        e1 = self.en1(x)
        e2 = self.en2(self.pool(e1))
        e3 = self.en3(self.pool(e2))
        e4 = self.en4(self.pool(e3))
        e5 = self.en5(self.pool(e4))

        # Expanding path
        d1 = self.de1(torch.cat([self.up(e5), e4], dim=1))
        d2 = self.de2(torch.cat([self.up(d1), e3], dim=1))
        d3 = self.de3(torch.cat([self.up(d2), e2], dim=1))
        d4 = self.de4(torch.cat([self.up(d3), e1], dim=1))

        # Output
        return self.conv1x1(d4)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        InceptionUNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
