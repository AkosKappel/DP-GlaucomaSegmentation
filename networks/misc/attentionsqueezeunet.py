import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['AttentionSqueezeUnet']


class FireModule(nn.Module):

    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(squeeze_channels),
        )

        self.left = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.right = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super(AttentionBlock, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=inter_channels),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=inter_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        x = F.interpolate(x, size=g.size()[2:], mode='bilinear', align_corners=True)
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpsamplingBlock(nn.Module):

    def __init__(self, input_channels, skip_channels, gate_channels, squeeze_channels, expand_channels,
                 kernel_size, stride, att_channels):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, gate_channels, kernel_size=kernel_size, stride=stride)
        self.fire = FireModule(gate_channels + skip_channels, squeeze_channels, expand_channels)
        self.attention = AttentionBlock(gate_channels, skip_channels, att_channels)

    def forward(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        d = torch.concat([d, x], dim=1)
        x = self.fire(d)
        return x


# See:
# https://github.com/apennisi/att_squeeze_unet/blob/main/networks/att_squeeze_unet.py
# https://github.com/rubythalib33/Att-Squeeze-UNet-Pytorch/blob/main/models/att-_squeeze_unet.py
class AttentionSqueezeUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(AttentionSqueezeUnet, self).__init__()

        features = [64, 128, 256, 384, 512]
        squeeze_channels = [16, 16, 32, 48, 48]
        expand_channels = [32, 64, 128, 192, 256]
        attention_channels = [4, 16, 64, 96]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.fire1 = FireModule(features[0], squeeze_channels[1], expand_channels[1])
        self.fire2 = FireModule(features[1], squeeze_channels[1], expand_channels[1])
        self.fire3 = FireModule(features[1], squeeze_channels[2], expand_channels[2])
        self.fire4 = FireModule(features[2], squeeze_channels[2], expand_channels[2])
        self.fire5 = FireModule(features[2], squeeze_channels[3], expand_channels[3])
        self.fire6 = FireModule(features[3], squeeze_channels[3], expand_channels[3])
        self.fire7 = FireModule(features[3], squeeze_channels[4], expand_channels[4])
        self.fire8 = FireModule(features[4], squeeze_channels[4], expand_channels[4])

        self.upsampling1 = UpsamplingBlock(
            features[4], features[3], expand_channels[3], squeeze_channels=squeeze_channels[3],
            expand_channels=expand_channels[3], stride=2, kernel_size=3, att_channels=attention_channels[3])
        self.upsampling2 = UpsamplingBlock(
            features[3], features[2], expand_channels[2], squeeze_channels=squeeze_channels[2],
            expand_channels=expand_channels[2], stride=2, kernel_size=3, att_channels=attention_channels[2])
        self.upsampling3 = UpsamplingBlock(
            features[2], features[1], expand_channels[1], squeeze_channels=squeeze_channels[1],
            expand_channels=expand_channels[1], stride=2, kernel_size=3, att_channels=attention_channels[1])
        self.upsampling4 = UpsamplingBlock(
            features[1], features[0], expand_channels[0], squeeze_channels=squeeze_channels[0],
            expand_channels=expand_channels[0], stride=1, kernel_size=3, att_channels=attention_channels[0])

        self.conv_out = nn.Sequential(
            nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
        )

    def forward(self, x):
        en1 = self.conv_in(x)
        en2 = self.pool(en1)
        en3 = self.pool(self.fire2(self.fire1(en2)))
        en4 = self.pool(self.fire4(self.fire3(en3)))
        en5 = self.fire6(self.fire5(en4))
        en6 = self.fire8(self.fire7(en5))

        de5 = self.upsampling1(en6, en5)
        de4 = self.upsampling2(de5, en4)
        de3 = self.upsampling3(de4, en3)
        de2 = self.upsampling4(de3, en2)
        de2 = F.interpolate(de2, size=en1.size()[2:], mode='bilinear', align_corners=True)
        de1 = torch.concat([de2, en1], dim=1)

        out = self.conv_out(de1)
        return out


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _models = [
        AttentionSqueezeUnet(in_channels=_in_channels, out_channels=_out_channels),
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
