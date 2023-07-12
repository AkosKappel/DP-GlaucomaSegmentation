import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['AttentionSqueezeUnet']


class FireModule(nn.Module):

    def __init__(self, squeeze_channels: int, expand_channels: int):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(expand_channels),
        )

        self.left = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(expand_channels),
        )

        self.right = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(expand_channels),
        )


class AttentionSqueezeUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(AttentionSqueezeUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Attention Squeeze U-Net requires a list of 5 features'

    def forward(self, x):
        pass


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        AttentionSqueezeUnet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
