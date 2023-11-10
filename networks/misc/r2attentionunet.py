import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['R2AttentionUnet']


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


class AttentionBlock(nn.Module):

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super(AttentionBlock, self).__init__()

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


class RecurrentBlock(nn.Module):
    def __init__(self, out_channels: int, t: int = 2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = None
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RecurrentResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t=2):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.recursion = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t),
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.recursion(x)
        return x + x1


class R2AttentionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None, t: int = 2):
        super(R2AttentionUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Attention U-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en_rr1 = RecurrentResidualBlock(in_channels, features[0], t)
        self.en_rr2 = RecurrentResidualBlock(features[0], features[1], t)
        self.en_rr3 = RecurrentResidualBlock(features[1], features[2], t)
        self.en_rr4 = RecurrentResidualBlock(features[2], features[3], t)
        self.en_rr5 = RecurrentResidualBlock(features[3], features[4], t)

        self.up1 = UpConv(features[4], features[3])
        self.att1 = AttentionBlock(features[3], features[3], features[3] // 2)
        self.de_rr1 = RecurrentResidualBlock(features[4], features[3], t)

        self.up2 = UpConv(features[3], features[2])
        self.att2 = AttentionBlock(features[2], features[2], features[2] // 2)
        self.de_rr2 = RecurrentResidualBlock(features[3], features[2], t)

        self.up3 = UpConv(features[2], features[1])
        self.att3 = AttentionBlock(features[1], features[1], features[1] // 2)
        self.de_rr3 = RecurrentResidualBlock(features[2], features[1], t)

        self.up4 = UpConv(features[1], features[0])
        self.att4 = AttentionBlock(features[0], features[0], features[0] // 2)
        self.de_rr4 = RecurrentResidualBlock(features[1], features[0], t)

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.en_rr1(x)
        e2 = self.en_rr2(self.pool(e1))
        e3 = self.en_rr3(self.pool(e2))
        e4 = self.en_rr4(self.pool(e3))
        e5 = self.en_rr5(self.pool(e4))

        d1 = self.up1(e5)
        d1 = self.de_rr1(torch.cat((self.att1(d1, e4), d1), dim=1))

        d2 = self.up2(d1)
        d2 = self.de_rr2(torch.cat((self.att2(d2, e3), d2), dim=1))

        d3 = self.up3(d2)
        d3 = self.de_rr3(torch.cat((self.att3(d3, e2), d3), dim=1))

        d4 = self.up4(d3)
        d4 = self.de_rr4(torch.cat((self.att4(d4, e1), d4), dim=1))

        out = self.final(d4)
        return out


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        R2AttentionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
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
