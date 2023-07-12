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


class TripleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_conv: int = 2, side_input_channels: int = 0):
        super(EncoderUnit, self).__init__()

        if num_conv == 1:
            self.conv = SingleConv(in_channels + side_input_channels, out_channels)
        elif num_conv == 2:
            self.conv = DoubleConv(in_channels + side_input_channels, out_channels)
        elif num_conv == 3:
            self.conv = TripleConv(in_channels + side_input_channels, out_channels)
        else:
            raise ValueError(f'Number of convolutions must be 1, 2 or 3, got {num_conv} instead')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, side_input=None):
        if side_input is not None:
            x = torch.cat([x, side_input], dim=1)
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderUnit(nn.Module):

    def __init__(self, up_channels: int, skip_channels: int, out_channels: int, num_conv: int = 2):
        super(DecoderUnit, self).__init__()

        if num_conv == 1:
            self.conv = SingleConv(skip_channels + up_channels, out_channels)
        elif num_conv == 2:
            self.conv = DoubleConv(skip_channels + up_channels, out_channels)
        elif num_conv == 3:
            self.conv = TripleConv(skip_channels + up_channels, out_channels)
        else:
            raise ValueError(f'Number of convolutions must be 1, 2 or 3, got {num_conv} instead')

        self.cbam = CBAM(skip_channels)
        self.ag = AttentionGate(up_channels, skip_channels, skip_channels // 2)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(up_channels, skip_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            # nn.BatchNorm2d(skip_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cbam(skip)
        skip = self.ag(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DoubleAttentionUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 side_input: bool = False):
        super(DoubleAttentionUnet, self).__init__()
        # side_input = multi-scale side input to the encoder

        if features is None:
            features = [64, 128, 256, 512]
        assert len(features) == 4, 'Attention U-Net requires a list of 4 features'

        self.side_input = side_input
        sic = in_channels if side_input else 0

        self.en1 = EncoderUnit(in_channels, features[0], num_conv=2)
        self.en2 = EncoderUnit(features[0], features[1], num_conv=2, side_input_channels=sic)
        self.en3 = EncoderUnit(features[1], features[2], num_conv=3, side_input_channels=sic)
        self.en4 = EncoderUnit(features[2], features[3], num_conv=3, side_input_channels=sic)

        self.de1 = DecoderUnit(features[3], features[2], features[2], num_conv=2)
        self.de2 = DecoderUnit(features[2], features[1], features[1], num_conv=2)
        self.de3 = DecoderUnit(features[1], features[0], features[0], num_conv=2)

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        pool1, skip1 = self.en1(x)
        x_2 = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True) if self.side_input else None
        pool2, skip2 = self.en2(pool1, x_2)
        x_4 = F.interpolate(x, scale_factor=1 / 4, mode='bilinear', align_corners=True) if self.side_input else None
        pool3, skip3 = self.en3(pool2, x_4)
        x_8 = F.interpolate(x, scale_factor=1 / 8, mode='bilinear', align_corners=True) if self.side_input else None
        _, skip4 = self.en4(pool3, x_8)

        up1 = self.de1(skip4, skip3)
        up2 = self.de2(up1, skip2)
        up3 = self.de3(up2, skip1)

        return self.final(up3)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [32, 64, 128, 256]
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
