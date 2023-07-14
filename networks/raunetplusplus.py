import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Residual Attention U-Net++
__all__ = ['RAUnetPlusPlus', 'DualRAUnetPlusPlus']


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

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderUnit(nn.Module):

    def __init__(self, in_channels: int, up_channels: int, n_skip_connections: int, out_channels: int):
        super(DecoderUnit, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = AttentionGate(in_channels, up_channels, in_channels // 2)
        self.conv = DoubleConv(in_channels * n_skip_connections + up_channels, out_channels)

    def forward(self, x, skips):
        large_skips, small_skip = skips[:-1], skips[-1]  # shortest skip connection goes through attention gate
        x = self.up(x)
        a = self.attention(x, small_skip)
        x = torch.cat((x, a, *large_skips), dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ResidualConv(in_channels, features[0])
        self.conv2 = ResidualConv(features[0], features[1])
        self.conv3 = ResidualConv(features[1], features[2])
        self.conv4 = ResidualConv(features[2], features[3])
        self.conv5 = ResidualConv(features[3], features[4])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int, deep_supervision: bool = False):
        super(Decoder, self).__init__()
        self.deep_supervision = deep_supervision

        self.branch1 = nn.ModuleList([
            DecoderUnit(features[0], features[1], 1, features[0]),
        ])

        self.branch2 = nn.ModuleList([
            DecoderUnit(features[1], features[2], 1, features[1]),
            DecoderUnit(features[0], features[1], 2, features[0]),
        ])

        self.branch3 = nn.ModuleList([
            DecoderUnit(features[2], features[3], 1, features[2]),
            DecoderUnit(features[1], features[2], 2, features[1]),
            DecoderUnit(features[0], features[1], 3, features[0]),
        ])

        self.branch4 = nn.ModuleList([
            DecoderUnit(features[3], features[4], 1, features[3]),
            DecoderUnit(features[2], features[3], 2, features[2]),
            DecoderUnit(features[1], features[2], 3, features[1]),
            DecoderUnit(features[0], features[1], 4, features[0]),
        ])

        # Output
        self.output = nn.ModuleList([
            OutConv(features[0], out_channels),
            OutConv(features[0], out_channels),
            OutConv(features[0], out_channels),
            OutConv(features[0], out_channels),
        ]) if self.deep_supervision else \
            OutConv(features[0], out_channels)

    def forward(self, *x):
        # Skip connections from backbone (top to bottom)
        x0_0, x1_0, x2_0, x3_0, x4_0 = x

        # Branch 1
        x0_1 = self.branch1[0](x1_0, [x0_0])

        # Branch 2
        x1_1 = self.branch2[0](x2_0, [x1_0])
        x0_2 = self.branch2[1](x1_1, [x0_0, x0_1])

        # Branch 3
        x2_1 = self.branch3[0](x3_0, [x2_0])
        x1_2 = self.branch3[1](x2_1, [x1_0, x1_1])
        x0_3 = self.branch3[2](x1_2, [x0_0, x0_1, x0_2])

        # Branch 4
        x3_1 = self.branch4[0](x4_0, [x3_0])
        x2_2 = self.branch4[1](x3_1, [x2_0, x2_1])
        x1_3 = self.branch4[2](x2_2, [x1_0, x1_1, x1_2])
        x0_4 = self.branch4[3](x1_3, [x0_0, x0_1, x0_2, x0_3])

        # Output
        if self.deep_supervision:
            out1 = self.output[0](x0_1)
            out2 = self.output[1](x0_2)
            out3 = self.output[2](x0_3)
            out4 = self.output[3](x0_4)
            return (out1 + out2 + out3 + out4) / 4
        else:
            return self.output(x0_4)


class RAUnetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, init_weights: bool = True):
        super(RAUnetPlusPlus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Residual Attention U-Net++ requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels, deep_supervision)

        # Initialize weights
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(*skips)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Use Kaiming initialization for ReLU activation function
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Use zero bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize weight to 1 and bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Dual decoder branch network
class DualRAUnetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, init_weights: bool = True):
        super(DualRAUnetPlusPlus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual Residual Attention U-Net++ requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder = Encoder(in_channels, features)
        self.decoder1 = Decoder(features, out_channels, deep_supervision)
        self.decoder2 = Decoder(features, out_channels, deep_supervision)

        # Initialize weights
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        skips = self.encoder(x)
        x1 = self.decoder1(*skips)
        x2 = self.decoder2(*skips)
        return x1, x2

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Use Kaiming initialization for ReLU activation function
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Use zero bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize weight to 1 and bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        RAUnetPlusPlus(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, deep_supervision=False),
        RAUnetPlusPlus(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, deep_supervision=True),
        DualRAUnetPlusPlus(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, deep_supervision=False),
        DualRAUnetPlusPlus(
            in_channels=_in_channels, out_channels=_out_channels, features=_layers, deep_supervision=True),
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
