import torch
import torch.nn as nn

__all__ = ['AttentionUNet']


class AttentionUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = None):
        super(AttentionUNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        assert len(features) == 5, 'Attention U-Net requires a list of 5 features'
