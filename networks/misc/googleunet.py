import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['GoogleUnet', 'DualGoogleUnet']


# Inception module with dimension reductions
class InceptionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(InceptionBlock, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], dim=1)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()

        self.conv1 = InceptionBlock(in_channels, features[0])
        self.conv2 = InceptionBlock(features[0] * 4, features[1])
        self.conv3 = InceptionBlock(features[1] * 4, features[2])
        self.conv4 = InceptionBlock(features[2] * 4, features[3])
        self.conv5 = InceptionBlock(features[3] * 4, features[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.pool(x1)

        x2 = self.conv2(x)
        x = self.pool(x2)

        x3 = self.conv3(x)
        x = self.pool(x3)

        x4 = self.conv4(x)
        x = self.pool(x4)

        x5 = self.conv5(x)

        return x1, x2, x3, x4, x5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int):
        super(Decoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(features[4] * 4, features[3], kernel_size=2, stride=2)
        self.conv1 = InceptionBlock(features[3] * 5, features[3])

        self.up2 = nn.ConvTranspose2d(features[3] * 4, features[2], kernel_size=2, stride=2)
        self.conv2 = InceptionBlock(features[2] * 5, features[2])

        self.up3 = nn.ConvTranspose2d(features[2] * 4, features[1], kernel_size=2, stride=2)
        self.conv3 = InceptionBlock(features[1] * 5, features[1])

        self.up4 = nn.ConvTranspose2d(features[1] * 4, features[0], kernel_size=2, stride=2)
        self.conv4 = InceptionBlock(features[0] * 5, features[0])

        self.conv5 = nn.Conv2d(features[0] * 4, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        x = self.conv5(x)

        return x


class GoogleUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(GoogleUnet, self).__init__()

        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert len(features) == 5, 'Google U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x


class DualGoogleUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(DualGoogleUnet, self).__init__()

        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert len(features) == 5, 'Dual Google U-Net requires a list of 5 features'

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
        GoogleUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        DualGoogleUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
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
