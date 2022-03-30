import torch
from torch import nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 activation=nn.ReLU, batch_norm=True):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(
                out_channels) if batch_norm else nn.Identity(), activation(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, padding=padding),
            nn.BatchNorm2d(
                out_channels) if batch_norm else nn.Identity(), activation()
        )

    def forward(self, X):
        return self.stack(X)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_filters=64, depth=5,
                 downsampler=nn.MaxPool2d, upsampler=nn.ConvTranspose2d, batch_norm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.depth = depth
        self.batch_norm = batch_norm
        self.factor = 2  # predefined

        channels = [n_filters * (2 ** idx) for idx in range(depth)]
        self.input_block = UNetBlock(self.in_channels, channels[0])

        self.down_blocks = nn.ModuleList([
            UNetBlock(in_channels, out_channels, batch_norm=self.batch_norm)
            for in_channels, out_channels in zip(channels, channels[1:])
        ])
        self.downsampler = downsampler(self.factor)

        channels = channels[::-1]
        self.up_blocks = nn.ModuleList([
            UNetBlock(in_channels, out_channels, batch_norm=self.batch_norm)
            for in_channels, out_channels in zip(channels, channels[1:])
        ])
        self.upsamplers = nn.ModuleList([
            upsampler(
                in_channels, out_channels,
                kernel_size=self.factor, stride=self.factor
            ) for in_channels, out_channels in zip(channels, channels[1:])
        ])

        self.output_block = nn.Sequential(
            nn.Conv2d(channels[-1], self.out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.input_block(X)
        skip_connections = []

        for block in self.down_blocks:
            skip_connections.append(X)
            X = block(self.downsampler(X))

        skip_connections = skip_connections[::-1]
        for block, skip, upsampler in zip(self.up_blocks, skip_connections, self.upsamplers):
            X = block(torch.hstack([upsampler(X), skip]))

        return self.output_block(X)
