import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return self.conv(x)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 is not None:
            diff_x = x2.size()[2] - x1.size()[2]
            diff_y = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)
        return x


class CenterNet(nn.Module):

    def __init__(self, n_classes: int = 1, model_name: str = 'resnet18'):
        super(CenterNet, self).__init__()

        # Backbone
        model_name = model_name.lower()
        if model_name == 'resnet18':
            basemodel = ResNet18()
        elif model_name == 'resnet34':
            basemodel = ResNet34()
        elif model_name == 'resnet50':
            basemodel = ResNet50()
        elif model_name == 'resnet101':
            basemodel = ResNet101()
        elif model_name == 'resnet152':
            basemodel = ResNet152()
        else:
            raise NotImplementedError(f'Model {model_name} is not implemented.')

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


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride: int = 1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks: list[int], num_classes: int = 5):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 512 -> 256
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 256 -> 128

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 128 -> 128
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 128 -> 64
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 64 -> 32
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 32 -> 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 512 * 1 -> 5

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 512 -> 256
        out = self.maxpool(out)  # 256 -> 128

        out = self.layer1(out)  # 128 -> 128
        out = self.layer2(out)  # 128 -> 64
        out = self.layer3(out)  # 64 -> 32
        out = self.layer4(out)  # 32 -> 16

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNet18():
    return ResNet(ResNetBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(ResNetBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(ResNetBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(ResNetBlock, [3, 4, 23, 3])


def ResNet152():
    return ResNet(ResNetBlock, [3, 8, 36, 3])


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
