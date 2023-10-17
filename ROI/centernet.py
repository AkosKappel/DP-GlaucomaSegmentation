import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
        else:
            x = x1
        x = self.conv(x)
        return x


class CenterNet(nn.Module):

    def __init__(self, n_classes: int = 1, model_name: str = 'resnet18', weights: str = None):
        super(CenterNet, self).__init__()

        # Backbone
        basemodel = torchvision.models.resnet18(weights=weights)
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        self.backbone = basemodel

        if model_name in ('resnet34', 'resnet18'):
            num_ch = 512
        else:
            num_ch = 2048

        # Neck
        self.up1 = Up(num_ch, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 256)

        # Head
        self.out_classification = nn.Conv2d(256, n_classes, 1)
        self.out_residue = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        x = self.backbone(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        classification = self.out_classification(x)
        residue = self.out_residue(x)

        return classification, residue


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f'Initialized {net.__class__.__name__} network parameters with {init_type} method.')
