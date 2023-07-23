import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['R2Unet', 'DualR2Unet']


class ConvBatchRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False, bn: bool = True):
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


class RecurrentResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t: int = 2):
        super(RecurrentResidualBlock, self).__init__()
        self.t = t
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = ConvBatchRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = ConvBatchRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        # 1x1 convolution to set correct number of channels for recurrent blocks
        x = self.conv1x1(x)

        # identity mapping
        residual = x

        # one pass is done before the recursion loop
        out = self.conv1(x)
        # first recurrent block
        for i in range(self.t):
            out = self.conv1(out + x)

        # second recurrent block
        x = out
        out = self.conv2(x)
        for i in range(self.t):
            out = self.conv2(out + x)

        return out + residual


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int], n_repeats: int = 2):
        super(Encoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en1 = RecurrentResidualBlock(in_channels, features[0], n_repeats)
        self.en2 = RecurrentResidualBlock(features[0], features[1], n_repeats)
        self.en3 = RecurrentResidualBlock(features[1], features[2], n_repeats)
        self.en4 = RecurrentResidualBlock(features[2], features[3], n_repeats)
        self.en5 = RecurrentResidualBlock(features[3], features[4], n_repeats)

    def forward(self, x):  # Contracting path
        e1 = self.en1(x)
        e2 = self.en2(self.pool(e1))
        e3 = self.en3(self.pool(e2))
        e4 = self.en4(self.pool(e3))
        e5 = self.en5(self.pool(e4))
        return e1, e2, e3, e4, e5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int, n_repeats: int = 2):
        super(Decoder, self).__init__()

        self.up1 = UpConv(features[4], features[3])
        self.up2 = UpConv(features[3], features[2])
        self.up3 = UpConv(features[2], features[1])
        self.up4 = UpConv(features[1], features[0])

        self.de1 = RecurrentResidualBlock(features[4], features[3], n_repeats)
        self.de2 = RecurrentResidualBlock(features[3], features[2], n_repeats)
        self.de3 = RecurrentResidualBlock(features[2], features[1], n_repeats)
        self.de4 = RecurrentResidualBlock(features[1], features[0], n_repeats)

        self.conv1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, e1, e2, e3, e4, e5):  # Expanding path
        d1 = self.de1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.de2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.de3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.de4(torch.cat([self.up4(d3), e1], dim=1))
        return self.conv1x1(d4)


class R2Unet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None, n_repeats: int = 2):
        super(R2Unet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Recurrent Residual U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features, n_repeats)
        self.decoder = Decoder(features, out_channels, n_repeats)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        return self.decoder(e1, e2, e3, e4, e5)


class DualR2Unet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None, n_repeats: int = 2):
        super(DualR2Unet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual Recurrent Residual U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features, n_repeats)
        self.decoder1 = Decoder(features, out_channels, n_repeats)
        self.decoder2 = Decoder(features, out_channels, n_repeats)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out1 = self.decoder1(e1, e2, e3, e4, e5)
        out2 = self.decoder2(e1, e2, e3, e4, e5)
        return out1, out2


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _k = 2
    _layers = [16, 32, 64, 128, 256]
    _models = [
        R2Unet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, n_repeats=_k),
        DualR2Unet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, n_repeats=_k),
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
