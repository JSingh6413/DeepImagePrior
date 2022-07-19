import torch
from torch import nn

import torch.nn.functional as F

import numpy as np
from functools import partial


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 activation=nn.ReLU, norm=nn.BatchNorm2d):
        '''
        nn.BatchNorm2d and nn.InstanceNorm2d are supported
        '''
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            norm(out_channels) if norm else nn.Identity(), activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            norm(out_channels) if norm else nn.Identity(), activation()
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=nn.MaxPool2d, **block_config):
        '''
        nn.MaxPool2d and nn.AvgPool2d are supported
        '''
        super().__init__()
        self.block = BasicBlock(in_channels, out_channels, **block_config)
        self.pooling = pooling(2)
        
    def forward(self, x, skip=True):
        x = self.block(x)
        return (self.pooling(x), x) if skip else self.pooling(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear', **block_config):
        '''
        any <mode> from F.interpolate is supported
        '''
        super().__init__()
        self.block = BasicBlock(in_channels, out_channels, **block_config)
        self.upsampler = partial(F.interpolate, mode=mode)
    
    def forward(self, x, skip=None):
        *_, height, width = x.size()
        x = self.upsampler(x, size=(2 * height, 2 * width))
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class BasicHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.Identity):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            activation()
        )


class UNet(nn.Module):
    def __init__(self, encoders, center, decoders, head=nn.Identity(), skips=True):
        super().__init__()
        self.encoders = encoders
        self.center = center
        self.decoders = decoders
        self.head = head
        self.skips = skips

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            if self.skips:
                x, skip = encoder(x, skip=True)
            else:
                x = encoder(x, skip=False)
                skip = None
            skip_connections.insert(0, skip)

        x = self.center(x)
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip)

        return self.head(x)


def configure_channels(in_channels, out_channels, min_channels, max_channels, depth, skips=True):
    encoder_out = [
        min(min_channels * (2 ** power), max_channels)
        for power in range(depth)
    ]
    encoder_in = [in_channels] + encoder_out[:-1]

    center_out = min(2 * encoder_out[-1], max_channels)
    center_in = encoder_out[-1]

    decoder_out = encoder_out[::-1]
    decoder_in = [center_out] + decoder_out[:-1]

    if skips:
        decoder_in = list(np.asarray(decoder_in) + np.asarray(encoder_out[::-1]))

    return list(zip(encoder_in, encoder_out)), (center_in, center_out), list(zip(decoder_in, decoder_out))


def build_unet(in_channels=1, out_channels=1, depth=4, min_channels=16, max_channels=64, skips=True, 
               pooling=nn.MaxPool2d, interpolation_mode='bilinear', head_activation=nn.Identity, **block_config):
    encoders_config, center_config, decoders_config = configure_channels(
        in_channels, out_channels, 
        min_channels, max_channels, 
        depth, skips
    )

    encoders = nn.ModuleList([
        EncoderBlock(*config, pooling=pooling, **block_config)
        for config in encoders_config
    ])

    center = BasicBlock(*center_config, **block_config)

    decoders = nn.ModuleList([
        DecoderBlock(*config, mode=interpolation_mode, **block_config)
        for config in decoders_config
    ])

    if out_channels != min_channels:
        head = BasicHead(min_channels, out_channels, activation=head_activation)
    else:
        head = nn.Identity()

    return UNet(encoders, center, decoders, head, skips)
