import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['InceptionUnet']


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

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(InceptionBlock, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBatchRelu(in_channels, mid_channels, kernel_size=3, padding=1),
            ConvBatchRelu(mid_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBatchRelu(in_channels, mid_channels, kernel_size=5, padding=2),
            ConvBatchRelu(mid_channels, out_channels, kernel_size=5, padding=2),
        )

        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBatchRelu(in_channels, mid_channels, kernel_size=1, padding=0),
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBatchRelu(in_channels, mid_channels, kernel_size=3, padding=1),
            ConvBatchRelu(mid_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        outputs = [self.block1(x), self.block2(x), self.block3(x), self.block4(x)]
        return torch.cat(outputs, dim=1)


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

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(UpConv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = self.up(x3)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x3, x2, x1], dim=1)

        return self.conv(x)


class DownConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(DownConv, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int], factor: int):
        super(Encoder, self).__init__()

        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert len(features) == 5, 'Encoder requires a list of 5 features'

        self.inception1 = InceptionBlock(features[0], features[0] // 2)
        self.inception2 = InceptionBlock(features[1], features[0])
        self.inception3 = InceptionBlock(features[2], features[1])
        self.inception4 = InceptionBlock(features[3], features[1])

        self.en1 = DoubleConv(in_channels, features[0])
        self.en2 = DownConv(features[0], features[1])
        self.en3 = DownConv(features[1], features[2])
        self.en4 = DownConv(features[2], features[3])
        self.bridge = DownConv(features[3], features[4] // factor)

    def forward(self, x):
        # Contracting path
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)

        # Bottleneck
        e5 = self.bridge(e4)

        # Inception blocks
        i1 = self.inception1(e1)
        i2 = self.inception2(i1)
        i3 = self.inception3(i2)
        i4 = self.inception4(i3)

        return e1, e2, e3, e4, e5, i1, i2, i3, i4


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int, factor: int, bilinear: bool):
        super(Decoder, self).__init__()

        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert len(features) == 5, 'Decoder requires a list of 5 features'

        self.de1 = UpConv(features[4] + features[3], features[2] // factor, bilinear)
        self.de2 = UpConv(features[3] + features[2] + features[1], features[1] // factor, bilinear)
        self.de3 = UpConv(features[2] + features[1] + features[0], features[0] // 2 // factor, bilinear)
        self.de4 = UpConv(features[1] + features[0] + features[0] // 4, features[0] // 4, bilinear)

        self.output = nn.Conv2d(features[0] // 4, out_channels, kernel_size=1)

    def forward(self, e1, e2, e3, e4, e5, i1, i2, i3, i4):
        # Expanding path
        d1 = self.de1(e5, e4, i4)
        d2 = self.de2(d1, e3, i3)
        d3 = self.de3(d2, e2, i2)
        d4 = self.de4(d3, e1, i1)

        # Output
        out = self.output(d4)
        return out


class InceptionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(InceptionUnet, self).__init__()

        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert len(features) == 5, 'Inception U-Net requires a list of 5 features'

        bilinear = True
        factor = 2 if bilinear else 1

        self.encoder = Encoder(in_channels, features, factor)
        self.decoder = Decoder(features, out_channels, factor, bilinear)

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
        return self.decoder(*self.encoder(x))


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        InceptionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        print(predictions.shape, (_batch_size, _out_channels, _height, _width))
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
