import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['R2AttentionUNet']


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
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

    def __init__(self, f_g, f_l, f_int):
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


class R2AttentionUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(R2AttentionUNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Attention U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en1 = ConvBlock(in_channels, features[0])
        self.en2 = ConvBlock(features[0], features[1])
        self.en3 = ConvBlock(features[1], features[2])
        self.en4 = ConvBlock(features[2], features[3])
        self.en5 = ConvBlock(features[3], features[4])

        self.up1 = UpConv(features[4], features[3])
        self.att1 = AttentionBlock(features[3], features[3], features[3] // 2)
        self.de1 = ConvBlock(features[4], features[3])

        self.up2 = UpConv(features[3], features[2])
        self.att2 = AttentionBlock(features[2], features[2], features[2] // 2)
        self.de2 = ConvBlock(features[3], features[2])

        self.up3 = UpConv(features[2], features[1])
        self.att3 = AttentionBlock(features[1], features[1], features[1] // 2)
        self.de3 = ConvBlock(features[2], features[1])

        self.up4 = UpConv(features[1], features[0])
        self.att4 = AttentionBlock(features[0], features[0], features[0] // 2)
        self.de4 = ConvBlock(features[1], features[0])

        self.last_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(self.pool(e1))
        e3 = self.en3(self.pool(e2))
        e4 = self.en4(self.pool(e3))
        e5 = self.en5(self.pool(e4))

        d1 = self.up1(e5)
        a1 = self.att1(d1, e4)
        d1 = torch.cat((a1, d1), dim=1)  # concatenate attention-weighted skip connection with previous layer output
        d1 = self.de1(d1)

        d2 = self.up2(d1)
        a2 = self.att2(d2, e3)
        d2 = self.de2(torch.cat((a2, d2), dim=1))

        d3 = self.up3(d2)
        a3 = self.att3(d3, e2)
        d3 = self.de3(torch.cat((a3, d3), dim=1))

        d4 = self.up4(d3)
        a4 = self.att4(d4, e1)
        d4 = self.de4(torch.cat((a4, d4), dim=1))

        out = self.last_conv(d4)
        return out


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        R2AttentionUNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
