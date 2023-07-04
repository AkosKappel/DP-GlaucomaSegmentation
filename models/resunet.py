import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['ResUnet']


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, mode: str = 'bilinear'):
        super(UpConv, self).__init__()
        if mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.up(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, relu_before: bool = False, downsample: bool = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu_before = relu_before
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
        ) if downsample else None

    def forward(self, x):
        residual = x

        # first convolution block
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)

        # second convolution block
        out = self.conv2(out)
        out = self.batch_norm2(out)
        # activation before residual shortcut
        if self.relu_before:
            out = F.relu(out)

        # 1x1 convolution to match channels of residual shortcut to output
        if self.downsample is not None:
            residual = self.downsample(residual)

        # residual shortcut (identity mapping)
        out = out + residual

        # activation after residual shortcut
        if not self.relu_before:
            out = F.relu(out)

        return out


class ResUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(ResUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Residual U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en1 = ResidualBlock(in_channels, features[0])
        self.en2 = ResidualBlock(features[0], features[1])
        self.en3 = ResidualBlock(features[1], features[2])
        self.en4 = ResidualBlock(features[2], features[3])
        self.en5 = ResidualBlock(features[3], features[4])

        self.up1 = UpConv(features[4], features[3], scale_factor=2, mode='transpose')
        self.up2 = UpConv(features[3], features[2], scale_factor=2, mode='transpose')
        self.up3 = UpConv(features[2], features[1], scale_factor=2, mode='transpose')
        self.up4 = UpConv(features[1], features[0], scale_factor=2, mode='transpose')

        self.de1 = ResidualBlock(features[4], features[3])
        self.de2 = ResidualBlock(features[3], features[2])
        self.de3 = ResidualBlock(features[2], features[1])
        self.de4 = ResidualBlock(features[1], features[0])

        self.conv1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
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
        d1 = self.de1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.de2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.de3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.de4(torch.cat([self.up4(d3), e1], dim=1))

        # Output
        return self.conv1x1(d4)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        ResUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
