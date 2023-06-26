import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchsummary import summary
from models.blocks import DoubleConv

__all__ = ['UNet', 'GenericUNet']


class UNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(UNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'U-Net requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(in_channels, features[0])
        self.conv2 = DoubleConv(features[0], features[1])
        self.conv3 = DoubleConv(features[1], features[2])
        self.conv4 = DoubleConv(features[2], features[3])

        self.bridge = DoubleConv(features[3], features[4])

        self.up1 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)

        self.conv6 = DoubleConv(features[4], features[3])
        self.conv7 = DoubleConv(features[3], features[2])
        self.conv8 = DoubleConv(features[2], features[1])
        self.conv9 = DoubleConv(features[1], features[0])

        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
        # Encoder
        en1 = self.conv1(x)
        en2 = self.conv2(self.pool(en1))
        en3 = self.conv3(self.pool(en2))
        en4 = self.conv4(self.pool(en3))

        # Bridge
        bridge = self.bridge(self.pool(en4))

        # Decoder
        de1 = self.conv6(torch.cat([self.up1(bridge), en4], dim=1))
        de2 = self.conv7(torch.cat([self.up2(de1), en3], dim=1))
        de3 = self.conv8(torch.cat([self.up3(de2), en2], dim=1))
        de4 = self.conv9(torch.cat([self.up4(de3), en1], dim=1))

        # Output
        return self.output(de4)


class GenericUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 init_weights: bool = True):
        super(GenericUNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features[:-1]:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bridge between encoder and decoder
        self.bottleneck = DoubleConv(features[-2], features[-1])

        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

        # Initialize weights
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
        skip_connections = []

        # Contracting path
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom part of the U-Net
        x = self.bottleneck(x)
        # Reverse the order of skip connections
        skip_connections = skip_connections[::-1]

        # Expanding path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]

            # Resize the skip connection if it is not the same size as the upsampled output
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate the skip connection with the upsampled output
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concat_skip)

        return self.output(x)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        UNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
        GenericUNet(in_channels=_in_channels, out_channels=_out_channels, features=_layers),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
