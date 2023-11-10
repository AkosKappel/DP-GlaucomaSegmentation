import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['R2UnetPlusPlus']


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, bn: bool = True):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn),
            *([nn.BatchNorm2d(mid_channels), ] if bn else []),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn),
            *([nn.BatchNorm2d(out_channels), ] if bn else []),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mode: str = 'bilinear', with_conv: bool = True,
                 scale_factor: int = 2):
        super(UpConv, self).__init__()
        if mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                *([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
                  if with_conv else []),
            )

    def forward(self, x):
        return self.up(x)


class RecurrentConv(nn.Module):
    def __init__(self, out_channels: int, t: int = 2):
        super(RecurrentConv, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            x1 = self.conv(x + x1)
        return x1


class RecurrentResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t=2):
        super(RecurrentResidualConv, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.recursion = nn.Sequential(
            RecurrentConv(out_channels, t=t),
            RecurrentConv(out_channels, t=t),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.recursion(x1)
        return x1 + x2


class R2UnetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, up_mode: str = 'bilinear', up_conv: bool = True):
        super(R2UnetPlusPlus, self).__init__()

        # deep_supervision: switch between fast (no DS) and accurate mode (with DS)
        # up_mode: 'transpose' or one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        # up_conv: whether to use Conv2d->BN->ReLU after Upsample (it changes number of channels)
        #          has no effect if up_mode is 'transpose'

        if features is None:
            features = [32, 64, 128, 256, 512]
        if up_mode == 'transpose':
            up_conv = True
        assert len(features) == 5, 'Recurrent Residual U-Net++ requires a list of 5 features'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deep_supervision = deep_supervision

        self.rows = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Row 1
        self.rows.append(nn.ModuleList([
            RecurrentResidualConv(in_channels, features[0]),
            RecurrentResidualConv(features[0] * 1 + features[0 if up_conv else 1], features[0]),
            RecurrentResidualConv(features[0] * 2 + features[0 if up_conv else 1], features[0]),
            RecurrentResidualConv(features[0] * 3 + features[0 if up_conv else 1], features[0]),
            RecurrentResidualConv(features[0] * 4 + features[0 if up_conv else 1], features[0]),
        ]))
        # No upsampling in first row

        # Row 2
        self.rows.append(nn.ModuleList([
            RecurrentResidualConv(features[0], features[1]),
            RecurrentResidualConv(features[1] * 1 + features[1 if up_conv else 2], features[1]),
            RecurrentResidualConv(features[1] * 2 + features[1 if up_conv else 2], features[1]),
            RecurrentResidualConv(features[1] * 3 + features[1 if up_conv else 2], features[1]),
        ]))
        self.ups.append(nn.ModuleList([
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
        ]))

        # Row 3
        self.rows.append(nn.ModuleList([
            RecurrentResidualConv(features[1], features[2]),
            RecurrentResidualConv(features[2] * 1 + features[2 if up_conv else 3], features[2]),
            RecurrentResidualConv(features[2] * 2 + features[2 if up_conv else 3], features[2]),
        ]))
        self.ups.append(nn.ModuleList([
            UpConv(features[2], features[1], up_mode, up_conv),
            UpConv(features[2], features[1], up_mode, up_conv),
            UpConv(features[2], features[1], up_mode, up_conv),
        ]))

        # Row 4
        self.rows.append(nn.ModuleList([
            RecurrentResidualConv(features[2], features[3]),
            RecurrentResidualConv(features[3] * 1 + features[3 if up_conv else 4], features[3]),
        ]))
        self.ups.append(nn.ModuleList([
            UpConv(features[3], features[2], up_mode, up_conv),
            UpConv(features[3], features[2], up_mode, up_conv),
        ]))

        # Row 5
        self.rows.append(nn.ModuleList([
            RecurrentResidualConv(features[3], features[4]),
        ]))
        self.ups.append(nn.ModuleList([
            UpConv(features[4], features[3], up_mode, up_conv),
        ]))

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

    def forward(self, x):
        x0_0 = self.rows[0][0](x)
        x1_0 = self.rows[1][0](self.pool(x0_0))
        x2_0 = self.rows[2][0](self.pool(x1_0))
        x3_0 = self.rows[3][0](self.pool(x2_0))
        x4_0 = self.rows[4][0](self.pool(x3_0))

        x0_1 = self.rows[0][1](torch.cat([x0_0, self.ups[0][0](x1_0)], dim=1))
        x1_1 = self.rows[1][1](torch.cat([x1_0, self.ups[1][0](x2_0)], dim=1))
        x2_1 = self.rows[2][1](torch.cat([x2_0, self.ups[2][0](x3_0)], dim=1))
        x3_1 = self.rows[3][1](torch.cat([x3_0, self.ups[3][0](x4_0)], dim=1))

        x0_2 = self.rows[0][2](torch.cat([x0_0, x0_1, self.ups[0][1](x1_1)], dim=1))
        x1_2 = self.rows[1][2](torch.cat([x1_0, x1_1, self.ups[1][1](x2_1)], dim=1))
        x2_2 = self.rows[2][2](torch.cat([x2_0, x2_1, self.ups[2][1](x3_1)], dim=1))

        x0_3 = self.rows[0][3](torch.cat([x0_0, x0_1, x0_2, self.ups[0][2](x1_2)], dim=1))
        x1_3 = self.rows[1][3](torch.cat([x1_0, x1_1, x1_2, self.ups[1][2](x2_2)], dim=1))

        x0_4 = self.rows[0][4](torch.cat([x0_0, x0_1, x0_2, x0_3, self.ups[0][3](x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.output[0](x0_1)
            output2 = self.output[1](x0_2)
            output3 = self.output[2](x0_3)
            output4 = self.output[3](x0_4)
            return (output1 + output2 + output3 + output4) / 4
        else:
            return self.output(x0_4)


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=False, up_mode='transpose', up_conv=True),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=True, up_mode='transpose', up_conv=True),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=False, up_mode='transpose', up_conv=False),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=True, up_mode='transpose', up_conv=False),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=False, up_mode='bilinear', up_conv=True),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=True, up_mode='bilinear', up_conv=True),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=False, up_mode='bilinear', up_conv=False),
        R2UnetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                       deep_supervision=True, up_mode='bilinear', up_conv=False),
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
