import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['AttentionUnet', 'DualAttentionUnet']


class SingleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super(AttentionGate, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(f_int),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(f_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating_signal, skip_connection):
        g1 = self.w_g(gating_signal)
        x1 = self.w_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int], multi_scale_input: bool = False):
        super(Encoder, self).__init__()

        self.multi_scale_input = multi_scale_input
        multiplier = 2 if multi_scale_input else 1

        self.en1 = DoubleConv(in_channels, features[0])
        self.en2 = DoubleConv(features[0] * multiplier, features[1])
        self.en3 = DoubleConv(features[1] * multiplier, features[2])
        self.en4 = DoubleConv(features[2] * multiplier, features[3])
        self.en5 = DoubleConv(features[3], features[4])

        if multi_scale_input:
            self.side1 = SingleConv(in_channels, features[0])
            self.side2 = SingleConv(in_channels, features[1])
            self.side3 = SingleConv(in_channels, features[2])

            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avgpool4 = nn.AvgPool2d(kernel_size=4, stride=4)
            self.avgpool8 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.en1(x)

        e2 = self.maxpool(e1)
        if self.multi_scale_input:
            x_half = self.avgpool2(x)
            e2 = torch.cat([e2, self.side1(x_half)], dim=1)
        e2 = self.en2(e2)

        e3 = self.maxpool(e2)
        if self.multi_scale_input:
            x_quarter = self.avgpool4(x)
            e3 = torch.cat([e3, self.side2(x_quarter)], dim=1)
        e3 = self.en3(e3)

        e4 = self.maxpool(e3)
        if self.multi_scale_input:
            x_eighth = self.avgpool8(x)
            e4 = torch.cat([e4, self.side3(x_eighth)], dim=1)
        e4 = self.en4(e4)

        e5 = self.maxpool(e4)
        e5 = self.en5(e5)

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int):
        super(Decoder, self).__init__()

        self.up1 = UpConv(features[4], features[3])
        self.ag1 = AttentionGate(features[3], features[3], features[3] // 2)
        self.de1 = DoubleConv(features[4], features[3])

        self.up2 = UpConv(features[3], features[2])
        self.ag2 = AttentionGate(features[2], features[2], features[2] // 2)
        self.de2 = DoubleConv(features[3], features[2])

        self.up3 = UpConv(features[2], features[1])
        self.ag3 = AttentionGate(features[1], features[1], features[1] // 2)
        self.de3 = DoubleConv(features[2], features[1])

        self.up4 = UpConv(features[1], features[0])
        self.ag4 = AttentionGate(features[0], features[0], features[0] // 2)
        self.de4 = DoubleConv(features[1], features[0])

        self.last_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, e1, e2, e3, e4, e5):
        d1 = self.up1(e5)
        a1 = self.ag1(d1, e4)
        d1 = torch.cat((a1, d1), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d1 = self.de1(d1)

        d2 = self.up2(d1)
        a2 = self.ag2(d2, e3)
        d2 = self.de2(torch.cat((a2, d2), dim=1))

        d3 = self.up3(d2)
        a3 = self.ag3(d3, e2)
        d3 = self.de3(torch.cat((a3, d3), dim=1))

        d4 = self.up4(d3)
        a4 = self.ag4(d4, e1)
        d4 = self.de4(torch.cat((a4, d4), dim=1))

        out = self.last_conv(d4)
        return out


class AttentionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 multi_scale_input: bool = False):
        super(AttentionUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Attention U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features, multi_scale_input)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out = self.decoder(e1, e2, e3, e4, e5)
        return out


class DualAttentionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 multi_scale_input: bool = False):
        super(DualAttentionUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual Attention U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features, multi_scale_input)
        self.decoder1 = Decoder(features, out_channels)
        self.decoder2 = Decoder(features, out_channels)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out1 = self.decoder1(e1, e2, e3, e4, e5)
        out2 = self.decoder2(e1, e2, e3, e4, e5)
        return out1, out2


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        AttentionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, multi_scale_input=False),
        AttentionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, multi_scale_input=True),
        DualAttentionUnet(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, multi_scale_input=False),
        DualAttentionUnet(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, multi_scale_input=True),
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
