import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['RefUnet3PlusCBAM']


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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):  # Conv3x3 -> BN -> ReLU (1x)
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):  # Conv3x3 -> BN -> ReLU (2x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(out)), inplace=True)


class ConvCBAM(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super(ConvCBAM, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x):  # Conv3x3 -> BN -> CBAM -> ReLU
        return F.relu(self.cbam(self.bn(self.conv(x))), inplace=True)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int]):
        super(Encoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Backbone encoder
        self.en1 = DoubleConv(in_channels, features[0])
        self.en2 = DoubleConv(features[0], features[1])
        self.en3 = DoubleConv(features[1], features[2])
        self.en4 = DoubleConv(features[2], features[3])
        self.en5 = DoubleConv(features[3], features[4])

    def forward(self, x):
        # Contracting path
        e1 = self.en1(x)
        e2 = self.en2(self.pool(e1))
        e3 = self.en3(self.pool(e2))
        e4 = self.en4(self.pool(e3))
        e5 = self.en5(self.pool(e4))
        return e1, e2, e3, e4, e5


class Decoder(nn.Module):

    def __init__(self, features: list[int], out_channels: int, concat_channels: int):
        super(Decoder, self).__init__()
        # concat_features = number of channels per skip connection that will be concatenated

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # Decoder at level 4 (lowest)
        self.de4_en2 = SingleConv(features[1], concat_channels)
        self.de4_en4 = SingleConv(features[3], concat_channels)
        self.de4_de5 = SingleConv(features[4], concat_channels)
        self.de4 = ConvCBAM(3 * concat_channels, features[3])

        # Decoder at level 3
        self.de3_en1 = SingleConv(features[0], concat_channels)
        self.de3_en3 = SingleConv(features[2], concat_channels)
        self.de3_de4 = SingleConv(features[3], concat_channels)
        self.de3 = ConvCBAM(3 * concat_channels, features[2])

        # Decoder at level 2
        self.de2_en2 = SingleConv(features[1], concat_channels)
        self.de2_de3 = SingleConv(features[2], concat_channels)
        self.de2 = ConvCBAM(2 * concat_channels, features[1])

        # Decoder at level 1 (highest)
        self.de1_en1 = SingleConv(features[0], concat_channels)
        self.de1_de2 = SingleConv(features[1], concat_channels)
        self.de1_de3 = SingleConv(features[2], concat_channels)
        self.de1_de4 = SingleConv(features[3], concat_channels)
        self.de1 = ConvCBAM(4 * concat_channels, features[0])

        # Final convolution
        self.last = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, *x):
        e1, e2, e3, e4, e5 = x  # skip connections from encoder

        # Expanding path with skip connections
        d4 = self.de4(torch.cat((
            self.de4_en2(self.pool4(e2)),
            self.de4_en4(e4),
            self.de4_de5(self.up2(e5)),
        ), dim=1))
        d3 = self.de3(torch.cat((
            self.de3_en1(self.pool4(e1)),
            self.de3_en3(e3),
            self.de3_de4(self.up2(d4)),
        ), dim=1))
        d2 = self.de2(torch.cat((
            self.de2_en2(e2),
            self.de2_de3(self.up2(d3)),
        ), dim=1))
        d1 = self.de1(torch.cat((
            self.de1_en1(e1),
            self.de1_de2(self.up2(d2)),
            self.de1_de3(self.up4(d3)),
            self.de1_de4(self.up8(d4)),
        ), dim=1))

        # Final layer with 1x1 convolution
        return self.last(d1)


class RefUnet3PlusCBAM(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(RefUnet3PlusCBAM, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Refined U-Net 3+ with CBAM requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels, features[0])

        # initialize weights
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(*x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DualRefUnet3PlusCBAM(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(DualRefUnet3PlusCBAM, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Dual Refined U-Net 3+ with CBAM requires a list of 5 features'

        self.encoder = Encoder(in_channels, features)
        self.decoder1 = Decoder(features, out_channels, features[0])
        self.decoder2 = Decoder(features, out_channels, features[0])

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder1(*x)
        x2 = self.decoder2(*x)
        return x1, x2

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        RefUnet3PlusCBAM(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        DualRefUnet3PlusCBAM(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        if isinstance(predictions, tuple):
            for prediction in predictions:
                assert prediction.shape == (_batch_size, _out_channels, _height, _width)
        else:
            assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
