import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super(SingleConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):  # Conv3x3 -> BN -> ReLU (1x)
        return F.relu(self.dropout(self.bn(self.conv(x))), inplace=True)


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, dropout: float = 0.2):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):  # Conv3x3 -> BN -> ReLU (2x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.dropout(self.bn2(self.conv2(out))), inplace=True)


class ConvCBAM(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super(ConvCBAM, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x):  # Conv3x3 -> BN -> CBAM -> ReLU
        return F.relu(self.cbam(self.bn(self.conv(x))), inplace=True)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, features: list[int], multi_scale_input: bool = False):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.multi_scale_input = multi_scale_input

        if multi_scale_input:
            self.side1 = SingleConv(in_channels, features[0])
            self.side2 = SingleConv(in_channels, features[1])
            self.side3 = SingleConv(in_channels, features[2])

            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avgpool4 = nn.AvgPool2d(kernel_size=4, stride=4)
            self.avgpool8 = nn.AvgPool2d(kernel_size=8, stride=8)

        multiplier = 2 if multi_scale_input else 1

        # Backbone encoder
        self.en1 = DoubleConv(in_channels, features[0])
        self.en2 = DoubleConv(features[0] * multiplier, features[1])
        self.en3 = DoubleConv(features[1] * multiplier, features[2])
        self.en4 = DoubleConv(features[2] * multiplier, features[3])
        self.en5 = DoubleConv(features[3], features[4])

    def forward(self, x):
        # Contracting path
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

    def __init__(self, features: list[int], out_channels: int, concat_channels: int, dropout: float = 0.2):
        super(Decoder, self).__init__()
        # concat_features = number of channels per skip connection that will be concatenated

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # Decoder at level 4 (lowest)
        self.de4_en2 = SingleConv(features[1], concat_channels, dropout=dropout)
        self.de4_en4 = SingleConv(features[3], concat_channels, dropout=dropout)
        self.de4_de5 = SingleConv(features[4], concat_channels, dropout=dropout)
        self.de4 = ConvCBAM(3 * concat_channels, features[3])

        # Decoder at level 3
        self.de3_en1 = SingleConv(features[0], concat_channels, dropout=dropout)
        self.de3_en3 = SingleConv(features[2], concat_channels, dropout=dropout)
        self.de3_de4 = SingleConv(features[3], concat_channels, dropout=dropout)
        self.de3 = ConvCBAM(3 * concat_channels, features[2])

        # Decoder at level 2
        self.de2_en2 = SingleConv(features[1], concat_channels, dropout=dropout)
        self.de2_de3 = SingleConv(features[2], concat_channels, dropout=dropout)
        self.de2 = ConvCBAM(2 * concat_channels, features[1])

        # Decoder at level 1 (highest)
        self.de1_en1 = SingleConv(features[0], concat_channels, dropout=dropout)
        self.de1_de2 = SingleConv(features[1], concat_channels, dropout=dropout)
        self.de1_de3 = SingleConv(features[2], concat_channels, dropout=dropout)
        self.de1_de4 = SingleConv(features[3], concat_channels, dropout=dropout)
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
                 multi_scale_input: bool = False, dropout: float = 0.2):
        super(RefUnet3PlusCBAM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features or [32, 64, 128, 256, 512]
        assert len(self.features) == 5, 'Refined U-Net 3+ with CBAM requires a list of 5 features'

        self.encoder = Encoder(in_channels, features, multi_scale_input)
        self.decoder = Decoder(features, out_channels, features[0], dropout=dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(*x)
        return x


# Dual Network: model with two decoder branches
class DualRefUnet3PlusCBAM(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 multi_scale_input: bool = False, dropout: float = 0.2):
        super(DualRefUnet3PlusCBAM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features or [32, 64, 128, 256, 512]
        assert len(self.features) == 5, 'Dual Refined U-Net 3+ with CBAM requires a list of 5 features'

        self.encoder = Encoder(in_channels, features, multi_scale_input)
        self.decoder1 = Decoder(features, out_channels, features[0], dropout=dropout)
        self.decoder2 = Decoder(features, out_channels, features[0], dropout=dropout)

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder1(*x)
        x2 = self.decoder2(*x)
        return x1, x2


# Cascade Network: two models in series
class CascadeRefUnet3PlusCBAM(nn.Module):

    def __init__(self, first_model, second_model,
                 activation=torch.sigmoid, threshold: float = 0.5, post_processing_functions: list = None):
        super(CascadeRefUnet3PlusCBAM, self).__init__()

        self.model1 = torch.load(first_model) if isinstance(first_model, str) else first_model
        self.model2 = torch.load(second_model) if isinstance(second_model, str) else second_model

        # Parameters inbetween models
        self.activation = activation
        self.threshold = threshold
        self.post_processing = post_processing_functions or []

    def forward(self, x):
        # First encoder-decoder model
        self.model1.eval()
        with torch.no_grad():
            x1 = self.model1(x)
            # Create binary mask from first model's output
            cascade_mask = (self.activation(x1) > self.threshold).long()

        # Post-processing to improve mask quality
        for func in self.post_processing:
            cascade_mask = func(cascade_mask)

        # Apply output mask from first model to input image
        x = x * cascade_mask

        # Second encoder-decoder model
        x2 = self.model2(x)

        return x1, x2


if __name__ == '__main__':
    _batch_size = 4
    _in_channels, _out_channels = 3, 1
    _height, _width = 64, 64
    _layers = [16, 24, 32, 40, 48]

    _random_data = torch.randn((_batch_size, _in_channels, _height, _width))

    _model = RefUnet3PlusCBAM(_in_channels, _out_channels, _layers)
    _predictions = _model(_random_data)
    assert _predictions.shape == (_batch_size, _out_channels, _height, _width)

    _dual_model = DualRefUnet3PlusCBAM(_in_channels, _out_channels, _layers)
    _predictions1, _predictions2 = _dual_model(_random_data)
    assert _predictions1.shape == (_batch_size, _out_channels, _height, _width)
    assert _predictions2.shape == (_batch_size, _out_channels, _height, _width)

    _cascade_model = CascadeRefUnet3PlusCBAM(_model, RefUnet3PlusCBAM(_in_channels, _out_channels, _layers))
    _predictions1, _predictions2 = _cascade_model(_random_data)
    assert _predictions1.shape == (_batch_size, _out_channels, _height, _width)
    assert _predictions2.shape == (_batch_size, _out_channels, _height, _width)
