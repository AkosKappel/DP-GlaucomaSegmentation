import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['USEnet', 'DualUSEnet']


class SEBlock(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        return x + residual


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


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()

        self.conv1 = DoubleConv(in_channels, features[0])
        self.conv2 = DoubleConv(features[0], features[1])
        self.conv3 = DoubleConv(features[1], features[2])
        self.conv4 = DoubleConv(features[2], features[3])
        self.conv5 = DoubleConv(features[3], features[4])

        self.se1 = SEBlock(features[0])
        self.se2 = SEBlock(features[1])
        self.se3 = SEBlock(features[2])
        self.se4 = SEBlock(features[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))

        e1 = self.se1(e1)
        e2 = self.se2(e2)
        e3 = self.se3(e3)
        e4 = self.se4(e4)

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int):
        super(Decoder, self).__init__()

        self.conv1 = DoubleConv(features[4] + features[3], features[3])
        self.conv2 = DoubleConv(features[3] + features[2], features[2])
        self.conv3 = DoubleConv(features[2] + features[1], features[1])
        self.conv4 = DoubleConv(features[1] + features[0], features[0])

        self.se1 = SEBlock(features[4])
        self.se2 = SEBlock(features[3])
        self.se3 = SEBlock(features[2])
        self.se4 = SEBlock(features[1])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.se = SEBlock(features[0])

    def forward(self, x1, x2, x3, x4, x5):
        x5 = self.se1(x5)
        x4 = self.se2(x4)
        x3 = self.se3(x3)
        x2 = self.se4(x2)

        d1 = self.conv1(torch.cat((self.up(x5), x4), dim=1))
        d2 = self.conv2(torch.cat((self.up(d1), x3), dim=1))
        d3 = self.conv3(torch.cat((self.up(d2), x2), dim=1))
        d4 = self.conv4(torch.cat((self.up(d3), x1), dim=1))

        out = self.se(d4)
        out = self.final(out)
        return out


class USEnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(USEnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'USEnet requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        return self.decoder(x1, x2, x3, x4, x5)


class DualUSEnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(DualUSEnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual USEnet requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder1 = Decoder(features, out_channels)
        self.decoder2 = Decoder(features, out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder1(*x)
        x2 = self.decoder2(*x)
        return x1, x2


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        USEnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        DualUSEnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
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
