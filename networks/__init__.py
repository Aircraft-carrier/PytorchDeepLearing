from .Unet3d import UNet3d

from torch import nn


def initialize_weights(net):
    if isinstance(net, (nn.Conv3d, nn.Conv2d)):
        nn.init.kaiming_normal_(net.weight.data, nonlinearity='relu')
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, (nn.ConvTranspose3d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(net.weight.data, nonlinearity='relu')
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.constant_(net.weight.data, 1)
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, nn.Linear):
        nn.init.kaiming_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0)

