import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['SwinUnet']


# TODO: Implement Shifted Window Vision Transformer Unet
class SwinUnet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(SwinUnet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'SwinU-Net requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        pass

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
