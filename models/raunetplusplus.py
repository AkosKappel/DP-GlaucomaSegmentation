import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ['ResAttentionUNetPlusPlus']


class SingleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mode: str = 'transpose', with_conv: bool = True,
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


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class ResAttentionUNetPlusPlus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, init_weights: bool = True,
                 up_mode: str = 'bilinear', up_conv: bool = True):
        super(ResAttentionUNetPlusPlus, self).__init__()

        # features: list of channels for each layer
        # deep_supervision: switch between fast (no DS) and accurate mode (with DS)
        # up_mode: 'transpose' or one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        # up_conv: whether to use Conv2d->BN->ReLU after Upsample (it changes number of channels)
        #          has no effect if up_mode is 'transpose'

        # block typer for convolution blocks outside backbone blocks
        PlainConv = ResidualConv  # ResidualConv, DoubleConv or SingleConv

        if features is None:
            features = [32, 64, 128, 256, 512]

        if up_mode == 'transpose':
            up_conv = True

        assert len(features) == 5, 'Residual Attention U-Net++ requires a list of 5 features'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deep_supervision = deep_supervision

        self.rows = nn.ModuleList()
        self.up_convolutions = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # Row 1
        self.rows.append(nn.ModuleList([
            ResidualConv(in_channels, features[0]),
            PlainConv(features[0] * 1 + features[0 if up_conv else 1], features[0]),
            PlainConv(features[0] * 2 + features[0 if up_conv else 1], features[0]),
            PlainConv(features[0] * 3 + features[0 if up_conv else 1], features[0]),
            PlainConv(features[0] * 4 + features[0 if up_conv else 1], features[0]),
        ]))
        # No attention gates or upsampling in the first row

        # Row 2
        self.rows.append(nn.ModuleList([
            ResidualConv(features[0], features[1]),
            PlainConv(features[1] * 1 + features[1 if up_conv else 2], features[1]),
            PlainConv(features[1] * 2 + features[1 if up_conv else 2], features[1]),
            PlainConv(features[1] * 3 + features[1 if up_conv else 2], features[1]),
        ]))
        self.up_convolutions.append(nn.ModuleList([
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
            UpConv(features[1], features[0], up_mode, up_conv),
        ]))
        self.attention_gates.append(nn.ModuleList([
            AttentionGate(features[0], features[0 if up_conv else 1], features[0] // 2),
            AttentionGate(features[0], features[0 if up_conv else 1], features[0] // 2),
            AttentionGate(features[0], features[0 if up_conv else 1], features[0] // 2),
            AttentionGate(features[0], features[0 if up_conv else 1], features[0] // 2),
        ]))

        # Row 3
        self.rows.append(nn.ModuleList([
            ResidualConv(features[1], features[2]),
            PlainConv(features[2] * 1 + features[2 if up_conv else 3], features[2]),
            PlainConv(features[2] * 2 + features[2 if up_conv else 3], features[2]),
        ]))
        self.up_convolutions.append(nn.ModuleList([
            UpConv(features[2], features[1], up_mode, up_conv),
            UpConv(features[2], features[1], up_mode, up_conv),
            UpConv(features[2], features[1], up_mode, up_conv),
        ]))
        self.attention_gates.append(nn.ModuleList([
            AttentionGate(features[1], features[1 if up_conv else 2], features[1] // 2),
            AttentionGate(features[1], features[1 if up_conv else 2], features[1] // 2),
            AttentionGate(features[1], features[1 if up_conv else 2], features[1] // 2),
        ]))

        # Row 4
        self.rows.append(nn.ModuleList([
            ResidualConv(features[2], features[3]),
            PlainConv(features[3] * 1 + features[3 if up_conv else 4], features[3]),
        ]))
        self.up_convolutions.append(nn.ModuleList([
            UpConv(features[3], features[2], up_mode, up_conv),
            UpConv(features[3], features[2], up_mode, up_conv),
        ]))
        self.attention_gates.append(nn.ModuleList([
            AttentionGate(features[2], features[2 if up_conv else 3], features[2] // 2),
            AttentionGate(features[2], features[2 if up_conv else 3], features[2] // 2),
        ]))

        # Row 5
        self.rows.append(nn.ModuleList([
            ResidualConv(features[3], features[4]),
        ]))
        self.up_convolutions.append(nn.ModuleList([
            UpConv(features[4], features[3], up_mode, up_conv),
        ]))
        self.attention_gates.append(nn.ModuleList([
            AttentionGate(features[3], features[3 if up_conv else 4], features[3] // 2),
        ]))

        # Output
        if self.deep_supervision:
            # Accurate mode (the final from all branches in top row are averaged to produce the final result)
            self.output = nn.ModuleList([
                OutConv(features[0], out_channels),
                OutConv(features[0], out_channels),
                OutConv(features[0], out_channels),
                OutConv(features[0], out_channels),
            ])
        else:
            # Fast mode (only the final from the last branch in top row is used)
            self.output = OutConv(features[0], out_channels)

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
        # Backbone
        x0_0 = self.rows[0][0](x)
        x1_0 = self.rows[1][0](self.pool(x0_0))
        x2_0 = self.rows[2][0](self.pool(x1_0))
        x3_0 = self.rows[3][0](self.pool(x2_0))
        x4_0 = self.rows[4][0](self.pool(x3_0))

        # Upsampling backbone blocks
        up1_0 = self.up_convolutions[0][0](x1_0)
        up2_0 = self.up_convolutions[1][0](x2_0)
        up3_0 = self.up_convolutions[2][0](x3_0)
        up4_0 = self.up_convolutions[3][0](x4_0)
        # Attention gates above backbone blocks
        ag1_0 = self.attention_gates[0][0](up1_0, x0_0)
        ag2_0 = self.attention_gates[1][0](up2_0, x1_0)
        ag3_0 = self.attention_gates[2][0](up3_0, x2_0)
        ag4_0 = self.attention_gates[3][0](up4_0, x3_0)
        # Diagonal above backbone / second block in each row
        x0_1 = self.rows[0][1](torch.cat([ag1_0, up1_0], dim=1))
        x1_1 = self.rows[1][1](torch.cat([ag2_0, up2_0], dim=1))
        x2_1 = self.rows[2][1](torch.cat([ag3_0, up3_0], dim=1))
        x3_1 = self.rows[3][1](torch.cat([ag4_0, up4_0], dim=1))

        # Repeating previous 3 steps for each diagonal row
        up1_1 = self.up_convolutions[0][1](x1_1)
        up2_1 = self.up_convolutions[1][1](x2_1)
        up3_1 = self.up_convolutions[2][1](x3_1)
        ag1_1 = self.attention_gates[0][1](up1_1, x0_1)
        ag2_1 = self.attention_gates[1][1](up2_1, x1_1)
        ag3_1 = self.attention_gates[2][1](up3_1, x2_1)
        x0_2 = self.rows[0][2](torch.cat([x0_0, ag1_1, up1_1], dim=1))
        x1_2 = self.rows[1][2](torch.cat([x1_0, ag2_1, up2_1], dim=1))
        x2_2 = self.rows[2][2](torch.cat([x2_0, ag3_1, up3_1], dim=1))

        up1_2 = self.up_convolutions[0][2](x1_2)
        up2_2 = self.up_convolutions[1][2](x2_2)
        ag1_2 = self.attention_gates[0][2](up1_2, x0_2)
        ag2_2 = self.attention_gates[1][2](up2_2, x1_2)
        x0_3 = self.rows[0][3](torch.cat([x0_0, x0_1, ag1_2, up1_2], dim=1))
        x1_3 = self.rows[1][3](torch.cat([x1_0, x1_1, ag2_2, up2_2], dim=1))

        up1_3 = self.up_convolutions[0][3](x1_3)
        ag1_3 = self.attention_gates[0][3](up1_3, x0_3)
        x0_4 = self.rows[0][4](torch.cat([x0_0, x0_1, x0_2, ag1_3, up1_3], dim=1))

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
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=False, up_mode='transpose', up_conv=True),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=True, up_mode='transpose', up_conv=True),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=False, up_mode='transpose', up_conv=False),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=True, up_mode='transpose', up_conv=False),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=False, up_mode='bilinear', up_conv=True),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=True, up_mode='bilinear', up_conv=True),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=False, up_mode='bilinear', up_conv=False),
        ResAttentionUNetPlusPlus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                                 deep_supervision=True, up_mode='bilinear', up_conv=False),
    ]
    random_data = torch.randn((_batch_size, _in_channels, _height, _width))
    for model in _models:
        predictions = model(random_data)
        assert predictions.shape == (_batch_size, _out_channels, _height, _width)
        print(model)
        summary(model.cuda(), (_in_channels, _height, _width))
        print()
