import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['DoubleAttentionUnet']


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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        avg_x = self.mlp(avg_x)

        max_x = self.max_pool(x).squeeze(-1).squeeze(-1)
        max_x = self.mlp(max_x)

        out = avg_x + max_x
        out = self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_x, max_x], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return out


class CBAM(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SingleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, side_input_channels: int = 0):
        super(EncoderUnit, self).__init__()
        if side_input_channels > 0:
            self.side_conv = SingleConv(side_input_channels, in_channels)
            in_channels *= 2
        else:
            self.side_conv = None
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, side_input=None):
        if self.side_conv is not None and side_input is not None:
            side_input = self.side_conv(side_input)
            x = torch.cat([x, side_input], dim=1)
        x = self.conv(x)
        return x


class DecoderUnit(nn.Module):

    def __init__(self, up_channels: int, skip_channels: int, out_channels: int):
        super(DecoderUnit, self).__init__()

        self.cbam = CBAM(skip_channels)
        self.ag = AttentionGate(up_channels, skip_channels, skip_channels // 2)
        self.conv = DoubleConv(skip_channels + up_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        skip = self.cbam(skip)
        x = self.up(x)
        skip = self.ag(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DoubleAttentionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 side_input: bool = False):
        super(DoubleAttentionUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Double Attention U-Net requires a list of 5 features'

        self.side_input = side_input
        side_input_channels = in_channels if side_input else 0

        self.en1 = EncoderUnit(in_channels, features[0])
        self.en2 = EncoderUnit(features[0], features[1], side_input_channels)
        self.en3 = EncoderUnit(features[1], features[2], side_input_channels)
        self.en4 = EncoderUnit(features[2], features[3], side_input_channels)
        self.en5 = EncoderUnit(features[3], features[4])

        self.de1 = DecoderUnit(features[4], features[3], features[3])
        self.de2 = DecoderUnit(features[3], features[2], features[2])
        self.de3 = DecoderUnit(features[2], features[1], features[1])
        self.de4 = DecoderUnit(features[1], features[0], features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.side_input:
            x_2 = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True)
            x_4 = F.interpolate(x, scale_factor=1 / 4, mode='bilinear', align_corners=True)
            x_8 = F.interpolate(x, scale_factor=1 / 8, mode='bilinear', align_corners=True)
        else:
            x_2 = x_4 = x_8 = None

        x1 = self.en1(x)
        x2 = self.en2(self.pool(x1), x_2)
        x3 = self.en3(self.pool(x2), x_4)
        x4 = self.en4(self.pool(x3), x_8)
        x5 = self.en5(self.pool(x4))

        x = self.de1(x5, x4)
        x = self.de2(x, x3)
        x = self.de3(x, x2)
        x = self.de4(x, x1)

        x = self.final(x)
        return x


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        DoubleAttentionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, side_input=False),
        DoubleAttentionUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, side_input=True),
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
