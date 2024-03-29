import torch
import torch.nn as nn
import torchvision.transforms.functional as F

# Differences from original Unet:
# 1) Add padding to preserve original input size
# 2) Add BatchNorm2d to improve my results


class UNET(nn.Module):
    def __init__(self, config):
        super(UNET, self).__init__()

        self.max_layer_size = config.model.max_layer_size
        self.min_layer_size = config.model.min_layer_size
        mls = config.model.max_layer_size
        in_channels = config.image.image_channels
        classes = config.image.mask_labels

        layers = [in_channels, config.model.min_layer_size]
        while mls > config.model.min_layer_size:
            layers.insert(2, mls)
            mls = int(mls * 0.5)

        self.layers = layers

        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])]
        )

        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
                for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])
            ]
        )

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]]
        )

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(self.min_layer_size, classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return conv

    def forward(self, x):
        # down layers
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        # up layers
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = F.resize(x, concat_layer.shape[2:], antialias=True)

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        return self.final_conv(x)
