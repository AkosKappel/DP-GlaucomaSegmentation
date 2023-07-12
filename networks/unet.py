import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['Unet', 'DualUnet', 'GenericUnet']


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


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mode: str = 'transpose', scale_factor: int = 2,
                 align_corners: bool = True):
        super(UpConv, self).__init__()
        if mode == 'transpose':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor,
                padding=0, dilation=1, bias=True,
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(in_channels, features[0])
        self.conv2 = DoubleConv(features[0], features[1])
        self.conv3 = DoubleConv(features[1], features[2])
        self.conv4 = DoubleConv(features[2], features[3])
        self.conv5 = DoubleConv(features[3], features[4])  # Bridge

    def forward(self, x):
        skip1 = self.conv1(x)
        x = self.pool(skip1)

        skip2 = self.conv2(x)
        x = self.pool(skip2)

        skip3 = self.conv3(x)
        x = self.pool(skip3)

        skip4 = self.conv4(x)
        x = self.pool(skip4)

        x = self.conv5(x)

        return x, skip1, skip2, skip3, skip4


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int):
        super(Decoder, self).__init__()

        self.up1 = UpConv(features[4], features[3], 'transpose', scale_factor=2)
        self.up2 = UpConv(features[3], features[2], 'transpose', scale_factor=2)
        self.up3 = UpConv(features[2], features[1], 'transpose', scale_factor=2)
        self.up4 = UpConv(features[1], features[0], 'transpose', scale_factor=2)

        self.conv1 = DoubleConv(features[4], features[3])
        self.conv2 = DoubleConv(features[3], features[2])
        self.conv3 = DoubleConv(features[2], features[1])
        self.conv4 = DoubleConv(features[1], features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, skips):
        x = self.up1(x)
        x = torch.cat((x, skips[3]), dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat((x, skips[2]), dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat((x, skips[1]), dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat((x, skips[0]), dim=1)
        x = self.conv4(x)

        return self.final(x)


class Unet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(Unet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'U-Net requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        x, *skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x


class DualUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(DualUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual U-Net requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder1 = Decoder(features, out_channels)
        self.decoder2 = Decoder(features, out_channels)

    def forward(self, x):
        x, *skips = self.encoder(x)
        x1 = self.decoder1(x, skips)
        x2 = self.decoder2(x, skips)
        return x1, x2


class GenericUnet(nn.Module):
    # generic models can have any number levels and features (e.g. 3 levels with 32, 64, 96 features)

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(GenericUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features[:-1]:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bridge between encoder and decoder
        self.bottleneck = DoubleConv(features[-2], features[-1])

        # Decoder
        for feature in reversed(features[:-1]):
            # Multiply by 2 because of skip connection concatenation
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.last = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []

        # Contracting path
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom part of the U-Net
        x = self.bottleneck(x)
        # Reverse the order of skip connections
        skip_connections = skip_connections[::-1]

        # Expanding path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]

            # Concatenate the skip connection with the upsampled final
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concat_skip)

        return self.last(x)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        Unet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        DualUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        GenericUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
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
