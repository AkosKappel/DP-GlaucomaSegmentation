import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['UNetPlusPlus', 'GenericUNetPlusPlus']


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class UNetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, init_weights: bool = True):
        super(UNetPlusPlus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'U-Net++ requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep_supervision = deep_supervision

        self.rows = nn.ModuleList()

        # Row 1
        row1 = nn.ModuleList([
            DoubleConv(in_channels, features[0]),
            DoubleConv(features[0] * 1 + features[1], features[0]),
            DoubleConv(features[0] * 2 + features[1], features[0]),
            DoubleConv(features[0] * 3 + features[1], features[0]),
            DoubleConv(features[0] * 4 + features[1], features[0]),
        ])
        self.rows.append(row1)

        # Row 2
        row2 = nn.ModuleList([
            DoubleConv(features[0], features[1]),
            DoubleConv(features[1] * 1 + features[2], features[1]),
            DoubleConv(features[1] * 2 + features[2], features[1]),
            DoubleConv(features[1] * 3 + features[2], features[1]),
        ])
        self.rows.append(row2)

        # Row 3
        row3 = nn.ModuleList([
            DoubleConv(features[1], features[2]),
            DoubleConv(features[2] * 1 + features[3], features[2]),
            DoubleConv(features[2] * 2 + features[3], features[2]),
        ])
        self.rows.append(row3)

        # Row 4
        row4 = nn.ModuleList([
            DoubleConv(features[2], features[3]),
            DoubleConv(features[3] * 1 + features[4], features[3]),
        ])
        self.rows.append(row4)

        # Row 5
        row5 = nn.ModuleList([
            DoubleConv(features[3], features[4]),
        ])
        self.rows.append(row5)

        # Output
        if self.deep_supervision:
            # Accurate mode (the final from all branches in top row are averaged to produce the final result)
            self.output = nn.ModuleList([
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Conv2d(features[0], out_channels, kernel_size=1),
            ])
        else:
            # Fast mode (only the final from the last branch in top row is used)
            self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
        x0_0 = self.rows[0][0](x)
        x1_0 = self.rows[1][0](self.pool(x0_0))
        x2_0 = self.rows[2][0](self.pool(x1_0))
        x3_0 = self.rows[3][0](self.pool(x2_0))
        x4_0 = self.rows[4][0](self.pool(x3_0))

        x0_1 = self.rows[0][1](torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.rows[1][1](torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.rows[2][1](torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.rows[3][1](torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.rows[0][2](torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.rows[1][2](torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.rows[2][2](torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.rows[0][3](torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.rows[1][3](torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.rows[0][4](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.output[0](x0_1)
            output2 = self.output[1](x0_2)
            output3 = self.output[2](x0_3)
            output4 = self.output[3](x0_4)
            return (output1 + output2 + output3 + output4) / 4
        else:
            return self.output(x0_4)


class GenericUNetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, init_weights: bool = True):
        super(GenericUNetPlusPlus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]

        self.n_rows = len(features)
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nested networks
        self.rows = nn.ModuleList()
        for i in range(self.n_rows):
            row = nn.ModuleList()
            for j in range(self.n_rows - i):
                if j == 0:
                    in_features = in_channels if i == 0 else features[i - 1]
                else:
                    in_features = features[i] * j + features[i + 1]
                out_features = features[i]
                row.append(DoubleConv(in_features, out_features))
            self.rows.append(row)

        # Output
        if self.deep_supervision:
            # Accurate mode (the final from all branches in top row are averaged to produce the final result)
            self.output = nn.ModuleList()
            for i in range(self.n_rows - 1):
                self.output.append(nn.Conv2d(features[0], out_channels, kernel_size=1))
        else:
            # Fast mode (only the final from the last branch in top row is used)
            self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
        x_values = [[None] * self.n_rows for _ in range(self.n_rows)]

        for i in range(self.n_rows):
            if i == 0:
                # Backbone
                for j in range(self.n_rows):
                    x_values[j][i] = self.rows[j][i](x) if j == 0 else self.rows[j][i](self.pool(x_values[j - 1][i]))
            else:
                # Nested and dense skip connections
                for j in range(self.n_rows - i):
                    skips = [x_values[j][k] for k in range(i)]
                    x = torch.cat([*skips, self.up(x_values[j + 1][i - 1])], dim=1)
                    x_values[j][i] = self.rows[j][i](x)

        if self.deep_supervision:
            n_outputs = self.n_rows - 1
            output = self.output[0](x_values[0][1])
            for i in range(1, n_outputs):
                output += self.output[i](x_values[0][i + 1])
            return output / n_outputs
        else:
            return self.output(x_values[0][self.n_rows - 1])


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        UNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                     deep_supervision=False),
        UNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                     deep_supervision=True),
        GenericUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                            deep_supervision=False),
        GenericUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                            deep_supervision=True),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
