import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ['Unet3Plus', 'GenericUnet3Plus']


def dot_product(x, cgm):
    B, N, H, W = x.size()
    x = x.view(B, N, H * W)
    y = torch.einsum('ijk,ij->ijk', [x, cgm])
    y = y.view(B, N, H, W)
    return y


class SingleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn)
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.act(out)
        return out


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, bn: bool = True):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels) if bn else None
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=not bn)
        self.batch_norm2 = nn.BatchNorm2d(out_channels) if bn else None
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm1:
            out = self.batch_norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        if self.batch_norm2:
            out = self.batch_norm2(out)
        out = self.act2(out)

        return out


class Unet3Plus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, cgm: bool = False, init_weights: bool = True):
        super(Unet3Plus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'U-Net 3+ requires a list of 5 features'

        self.deep_supervision = deep_supervision

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.pool8 = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # backbone (encoder + bridge)
        self.en1 = DoubleConv(in_channels, features[0])
        self.en2 = DoubleConv(features[0], features[1])
        self.en3 = DoubleConv(features[1], features[2])
        self.en4 = DoubleConv(features[2], features[3])
        self.en5 = DoubleConv(features[3], features[4])

        concat_channels = features[0]
        concat_blocks = len(features)
        up_channels = concat_channels * concat_blocks

        # decoder - level 4
        self.de4_1 = SingleConv(features[0], concat_channels)
        self.de4_2 = SingleConv(features[1], concat_channels)
        self.de4_3 = SingleConv(features[2], concat_channels)
        self.de4_4 = SingleConv(features[3], concat_channels)
        self.de4_5 = SingleConv(features[4], concat_channels)
        self.de4 = SingleConv(up_channels, up_channels)

        # decoder - level 3
        self.de3_1 = SingleConv(features[0], concat_channels)
        self.de3_2 = SingleConv(features[1], concat_channels)
        self.de3_3 = SingleConv(features[2], concat_channels)
        self.de3_4 = SingleConv(up_channels, concat_channels)
        self.de3_5 = SingleConv(features[4], concat_channels)
        self.de3 = SingleConv(up_channels, up_channels)

        # decoder - level 2
        self.de2_1 = SingleConv(features[0], concat_channels)
        self.de2_2 = SingleConv(features[1], concat_channels)
        self.de2_3 = SingleConv(up_channels, concat_channels)
        self.de2_4 = SingleConv(up_channels, concat_channels)
        self.de2_5 = SingleConv(features[4], concat_channels)
        self.de2 = SingleConv(up_channels, up_channels)

        # decoder - level 1
        self.de1_1 = SingleConv(features[0], concat_channels)
        self.de1_2 = SingleConv(up_channels, concat_channels)
        self.de1_3 = SingleConv(up_channels, concat_channels)
        self.de1_4 = SingleConv(up_channels, concat_channels)
        self.de1_5 = SingleConv(features[4], concat_channels)
        self.de1 = SingleConv(up_channels, up_channels)

        # deep supervision
        if self.deep_supervision:
            self.output = nn.ModuleList([
                nn.Conv2d(up_channels, out_channels, kernel_size=1),
                nn.Conv2d(up_channels, out_channels, kernel_size=1),
                nn.Conv2d(up_channels, out_channels, kernel_size=1),
                nn.Conv2d(up_channels, out_channels, kernel_size=1),
                nn.Conv2d(features[4], out_channels, kernel_size=1),
            ])
        else:
            self.output = nn.Conv2d(up_channels, out_channels, kernel_size=1)

        # classification-guided module
        if cgm:
            self.cgm = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(features[4], 2, kernel_size=1),
                nn.AdaptiveMaxPool2d(1),
                nn.Sigmoid(),
            )
        else:
            self.cgm = None

        # initialize weights
        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.en1(x)  # 320 x 320 x 64
        e2 = self.en2(self.pool2(e1))  # 160 x 160 x 128
        e3 = self.en3(self.pool2(e2))  # 80 x 80 x 256
        e4 = self.en4(self.pool2(e3))  # 40 x 40 x 512
        e5 = self.en5(self.pool2(e4))  # 20 x 20 x 1024

        # Decoder
        d4_1 = self.de4_1(self.pool8(e1))
        d4_2 = self.de4_2(self.pool4(e2))
        d4_3 = self.de4_3(self.pool2(e3))
        d4_4 = self.de4_4(e4)
        d4_5 = self.de4_5(self.up2(e5))
        d4 = self.de4(torch.cat((d4_1, d4_2, d4_3, d4_4, d4_5), dim=1))  # 40 x 40 x UpChannels

        d3_1 = self.de3_1(self.pool4(e1))
        d3_2 = self.de3_2(self.pool2(e2))
        d3_3 = self.de3_3(e3)
        d3_4 = self.de3_4(self.up2(d4))
        d3_5 = self.de3_5(self.up4(e5))
        d3 = self.de3(torch.cat((d3_1, d3_2, d3_3, d3_4, d3_5), dim=1))  # 80 x 80 x UpChannels

        d2_1 = self.de2_1(self.pool2(e1))
        d2_2 = self.de2_2(e2)
        d2_3 = self.de2_3(self.up2(d3))
        d2_4 = self.de2_4(self.up4(d4))
        d2_5 = self.de2_5(self.up8(e5))
        d2 = self.de2(torch.cat((d2_1, d2_2, d2_3, d2_4, d2_5), dim=1))  # 160 x 160 x UpChannels

        d1_1 = self.de1_1(e1)
        d1_2 = self.de1_2(self.up2(d2))
        d1_3 = self.de1_3(self.up4(d3))
        d1_4 = self.de1_4(self.up8(d4))
        d1_5 = self.de1_5(self.up16(e5))
        d1 = self.de1(torch.cat((d1_1, d1_2, d1_3, d1_4, d1_5), dim=1))  # 320 x 320 x UpChannels

        # Output (Deep Supervision & Classification-guided module)
        if self.deep_supervision:
            output1 = self.output[0](d1)
            output2 = self.output[1](self.up2(d2))
            output3 = self.output[2](self.up4(d3))
            output4 = self.output[3](self.up8(d4))
            output5 = self.output[4](self.up16(e5))

            if self.cgm is not None:
                cgm_branch = self.cgm(e5).squeeze(3).squeeze(2)
                cgm_branch_max = cgm_branch.argmax(dim=1).unsqueeze(1).float()

                output1 = dot_product(output1, cgm_branch_max)
                output2 = dot_product(output2, cgm_branch_max)
                output3 = dot_product(output3, cgm_branch_max)
                output4 = dot_product(output4, cgm_branch_max)
                output5 = dot_product(output5, cgm_branch_max)

            output = (output1 + output2 + output3 + output4 + output5) / 5
        else:
            output = self.output(d1)

            if self.cgm is not None:
                cgm_branch = self.cgm(e5).squeeze(3).squeeze(2)
                cgm_branch_max = cgm_branch.argmax(dim=1).unsqueeze(1).float()
                output = dot_product(output, cgm_branch_max)

        return output


class GenericUnet3Plus(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None,
                 deep_supervision: bool = False, cgm: bool = False, init_weights: bool = True):
        super(GenericUnet3Plus, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]

        self.n_features = len(features)
        self.deep_supervision = deep_supervision

        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2 ** i, stride=2 ** i, ceil_mode=True) for i in range(1, self.n_features - 1)
        ])
        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True) for i in range(1, self.n_features)
        ])

        # encoder (backbone)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        concat_channels = features[0]
        concat_blocks = len(features)
        up_channels = concat_channels * concat_blocks

        # skip connections
        self.levels = nn.ModuleList()
        for i in range(self.n_features - 1):
            level = nn.ModuleList()
            for j in range(self.n_features):
                if j < self.n_features - 1 - i or j == self.n_features - 1:
                    conv = SingleConv(features[j], concat_channels)
                else:
                    conv = SingleConv(up_channels, concat_channels)
                level.append(conv)
            self.levels.append(level)

        # decoder
        self.decoder = nn.ModuleList([SingleConv(up_channels, up_channels) for _ in range(self.n_features - 1)])

        # final
        if self.deep_supervision:
            self.output = nn.ModuleList()
            for _ in range(self.n_features - 1):
                self.output.append(nn.Conv2d(up_channels, out_channels, kernel_size=1))
            self.output.append(nn.Conv2d(features[-1], out_channels, kernel_size=1))
        else:
            self.output = nn.Conv2d(up_channels, out_channels, kernel_size=1)

        # classification-guided module
        if cgm:
            self.cgm = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(features[-1], 2, kernel_size=1),
                nn.AdaptiveMaxPool2d(1),
                nn.Sigmoid(),
            )
        else:
            self.cgm = None

        # initialize weights
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
        en = [self.encoder[0](x)]
        for j in range(1, self.n_features):
            en.append(self.encoder[j](self.pools[0](en[j - 1])))

        # Decoder
        n_pools = len(self.pools)
        de = [en[-1]]
        for i in range(self.n_features - 1):
            depth = self.n_features - i - 2
            # full-scale skip connections
            skips = []
            for j in range(self.n_features):
                if j < depth:  # downscale
                    skip = self.levels[i][j](self.pools[n_pools - 1 - j - i](en[j]))
                elif j == depth:  # no resize
                    skip = self.levels[i][j](en[j])
                else:  # upscale
                    skip = self.levels[i][j](self.ups[j - depth - 1](de[-(j - depth)]))
                skips.append(skip)
            d = self.decoder[i](torch.cat(skips, dim=1))
            de.append(d)

        # Deep supervision and Classification-guided module
        if self.deep_supervision:
            outputs = [self.output[0](de[-1])]
            for i, upscale in enumerate(self.ups):
                outputs.append(self.output[i + 1](upscale(de[-(i + 2)])))

            if self.cgm is not None:
                cgm_branch = self.cgm(de[0]).squeeze(3).squeeze(2)
                cgm_branch_max = cgm_branch.argmax(dim=1).unsqueeze(1).float()

                for i, o in enumerate(outputs):
                    outputs[i] = dot_product(o, cgm_branch_max)

            # average outputs
            output = outputs[0]
            for _, o in enumerate(outputs[1:]):
                output += o
            output /= self.n_features
        else:
            output = self.output(de[-1])

            if self.cgm is not None:
                cgm_branch = self.cgm(de[0]).squeeze(3).squeeze(2)
                cgm_branch_max = cgm_branch.argmax(dim=1).unsqueeze(1).float()
                output = dot_product(output, cgm_branch_max)

        return output


if __name__ == '__main__':
    _batch_size = 8
    _in_channels, _out_channels = 3, 1
    _height, _width = 128, 128
    _layers = [16, 32, 64, 128, 256]
    _models = [
        Unet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                  deep_supervision=False, cgm=False),
        Unet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                  deep_supervision=False, cgm=True),
        Unet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                  deep_supervision=True, cgm=False),
        Unet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                  deep_supervision=True, cgm=True),
        GenericUnet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                         deep_supervision=False, cgm=False),
        GenericUnet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                         deep_supervision=False, cgm=True),
        GenericUnet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                         deep_supervision=True, cgm=False),
        GenericUnet3Plus(in_channels=_in_channels, out_channels=_out_channels, features=_layers,
                         deep_supervision=True, cgm=True),
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
