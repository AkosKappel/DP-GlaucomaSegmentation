import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['RUNet']


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


class RecurrentBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t: int = 2, bn: bool = True):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            *([nn.BatchNorm2d(out_channels), ] if bn else []),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            *([nn.BatchNorm2d(out_channels), ] if bn else []),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 1x1 convolution to set correct number of channels for recurrent blocks
        out = self.conv1x1(x)

        x = out
        # first recurrent block
        for i in range(self.t):
            # one pass is done before the recursion loop begins
            if i == 0:
                out = self.block1(x)
            out = self.block1(out + x)

        # reset input for second recurrent block
        x = out
        # second recurrent block
        for i in range(self.t):
            if i == 0:
                out = self.block2(x)
            out = self.block2(out + x)

        return out


class RUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 n_repeats: int = 2, init_weights: bool = True):
        super(RUNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Recurrent U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en1 = RecurrentBlock(in_channels, features[0], n_repeats)
        self.en2 = RecurrentBlock(features[0], features[1], n_repeats)
        self.en3 = RecurrentBlock(features[1], features[2], n_repeats)
        self.en4 = RecurrentBlock(features[2], features[3], n_repeats)
        self.en5 = RecurrentBlock(features[3], features[4], n_repeats)

        self.up1 = UpConv(features[4], features[3], scale_factor=2, mode='transpose')
        self.up2 = UpConv(features[3], features[2], scale_factor=2, mode='transpose')
        self.up3 = UpConv(features[2], features[1], scale_factor=2, mode='transpose')
        self.up4 = UpConv(features[1], features[0], scale_factor=2, mode='transpose')

        self.de1 = RecurrentBlock(features[4], features[3], n_repeats)
        self.de2 = RecurrentBlock(features[3], features[2], n_repeats)
        self.de3 = RecurrentBlock(features[2], features[1], n_repeats)
        self.de4 = RecurrentBlock(features[1], features[0], n_repeats)

        self.conv1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if init_weights:
            self.initialize_weights()

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

    def forward(self, x):
        # Contracting path
        e1 = self.en1(x)
        e2 = self.en2(self.pool(e1))
        e3 = self.en3(self.pool(e2))
        e4 = self.en4(self.pool(e3))
        e5 = self.en5(self.pool(e4))

        # Expanding path
        d1 = self.de1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.de2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.de3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.de4(torch.cat([self.up4(d3), e1], dim=1))

        # Output
        return self.conv1x1(d4)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _k = 2
    _layers = [16, 32, 64, 128, 256]
    _models = [
        RUNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, n_repeats=_k),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
