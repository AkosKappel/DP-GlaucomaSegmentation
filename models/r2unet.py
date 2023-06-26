import torch
import torch.nn as nn
from torchsummary import summary
from models.blocks import RecurrentResidualBlock, UpConv

__all__ = ['R2UNet']


class R2UNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 n_repeats: int = 2, init_weights: bool = True):
        super(R2UNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Recurrent Residual U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = RecurrentResidualBlock(in_channels, features[0], n_repeats)
        self.encoder2 = RecurrentResidualBlock(features[0], features[1], n_repeats)
        self.encoder3 = RecurrentResidualBlock(features[1], features[2], n_repeats)
        self.encoder4 = RecurrentResidualBlock(features[2], features[3], n_repeats)
        self.encoder5 = RecurrentResidualBlock(features[3], features[4], n_repeats)

        self.up1 = UpConv(features[4], features[3], scale_factor=2)
        self.up2 = UpConv(features[3], features[2], scale_factor=2)
        self.up3 = UpConv(features[2], features[1], scale_factor=2)
        self.up4 = UpConv(features[1], features[0], scale_factor=2)

        self.decoder1 = RecurrentResidualBlock(features[4], features[3], n_repeats)
        self.decoder2 = RecurrentResidualBlock(features[3], features[2], n_repeats)
        self.decoder3 = RecurrentResidualBlock(features[2], features[1], n_repeats)
        self.decoder4 = RecurrentResidualBlock(features[1], features[0], n_repeats)

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
        en1 = self.encoder1(x)
        en2 = self.encoder2(self.pool(en1))
        en3 = self.encoder3(self.pool(en2))
        en4 = self.encoder4(self.pool(en3))
        en5 = self.encoder5(self.pool(en4))

        # Expanding path
        de1 = self.decoder1(torch.cat([self.up1(en5), en4], dim=1))
        de2 = self.decoder2(torch.cat([self.up2(de1), en3], dim=1))
        de3 = self.decoder3(torch.cat([self.up3(de2), en2], dim=1))
        de4 = self.decoder4(torch.cat([self.up4(de3), en1], dim=1))

        # Output
        return self.conv1x1(de4)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _k = 2
    _layers = [16, 32, 64, 128, 256]
    _models = [
        R2UNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers, n_repeats=_k),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
