import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['SqueezeUnet', 'DualSqueezeUnet']


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, bn: bool = True):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn),
            *([nn.BatchNorm2d(mid_channels), ] if bn else []),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn),
            *([nn.BatchNorm2d(out_channels), ] if bn else []),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FireModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, squeeze_channels: int = None):
        super(FireModule, self).__init__()

        if not squeeze_channels:
            squeeze_channels = out_channels // 4

        self.conv = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, padding=0)
        self.conv1x1 = nn.Conv2d(squeeze_channels, out_channels // 2, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(squeeze_channels, out_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        x1 = self.conv1x1(x)
        x1 = F.relu(x1)

        x3 = self.conv3x3(x)
        x3 = F.relu(x3)

        return torch.cat([x1, x3], dim=1)


class TransposedFireModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, squeeze_channels: int = None):
        super(TransposedFireModule, self).__init__()

        if not squeeze_channels:
            squeeze_channels = in_channels // 4

        self.conv = nn.ConvTranspose2d(in_channels, squeeze_channels, kernel_size=1, padding=0)
        self.conv1x1 = nn.ConvTranspose2d(squeeze_channels, out_channels // 2, kernel_size=1, padding=0)
        self.conv2x2 = nn.ConvTranspose2d(squeeze_channels, out_channels // 2, kernel_size=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        x1 = self.conv1x1(x)
        x1 = F.relu(x1)

        x3 = self.conv2x2(x)
        x3 = F.relu(x3)

        # x1.shape = (batch_size, channels, height, width)
        # x3.shape = (batch_size, channels, height + 1, width + 1)
        # fix x3.shape to x1.shape
        x3 = x3[:, :, :x1.shape[2], :x1.shape[3]]

        return torch.cat([x1, x3], dim=1)


class DownSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DownSample, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.f1 = FireModule(in_channels, mid_channels)
        self.f2 = FireModule(mid_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.f1(x)
        x = self.f2(x)
        return x


class UpSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(UpSample, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.tf1 = TransposedFireModule(in_channels, out_channels)
        # multiply by 2 because of concatenation
        self.f1 = FireModule(out_channels * 2, mid_channels)
        self.f2 = FireModule(mid_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = self.tf1(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.f1(x)
        x = self.f2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()

        self.en1 = DoubleConv(in_channels, features[0])
        self.ds1 = DownSample(features[0], features[1])
        self.ds2 = DownSample(features[1], features[2])
        self.ds3 = DownSample(features[2], features[3])
        self.ds4 = DownSample(features[3], features[4])

    def forward(self, x):
        x1 = self.en1(x)
        x2 = self.ds1(x1)
        x3 = self.ds2(x2)
        x4 = self.ds3(x3)
        x5 = self.ds4(x4)
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int):
        super(Decoder, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.us1 = UpSample(features[4], features[3])
        self.us2 = UpSample(features[3], features[2])
        self.us3 = UpSample(features[2], features[1])
        self.de1 = TransposedFireModule(features[1], features[0])
        # multiply by 2 because of concatenation
        self.de2 = DoubleConv(features[0] * 2, features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.us1(x5, x4)
        x = self.us2(x, x3)
        x = self.us3(x, x2)
        x = self.de1(x)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.de2(x)
        x = self.final(x)
        return x


class SqueezeUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(SqueezeUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Squeeze U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x


class DualSqueezeUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(DualSqueezeUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual Squeeze U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features)
        self.decoder1 = Decoder(features, out_channels)
        self.decoder2 = Decoder(features, out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out1 = self.decoder1(x1, x2, x3, x4, x5)
        out2 = self.decoder2(x1, x2, x3, x4, x5)
        return out1, out2


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        SqueezeUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        DualSqueezeUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for _model in _models:
        predictions = _model(random_data)
        if isinstance(predictions, tuple):
            for prediction in predictions:
                assert prediction.shape == (_batch_size, _out_channels, _height, _width)
        else:
            assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(_model)
        summary(_model.cuda(), (_in_channels, _height, _width))
        print()