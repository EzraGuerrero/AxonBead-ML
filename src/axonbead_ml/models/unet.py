"""
A small U-Net for bead-heatmap prediction.

U-Net's defining idea: an encoder that progressively downsamples (learning
increasingly abstract features) paired with a decoder that upsamples back
to the original resolution, with "skip connections" carrying fine-grained
spatial detail directly across from encoder to decoder. Without those
skip connections, the decoder would have to reconstruct precise bead
locations from a heavily compressed representation alone — skip
connections are why U-Net is good at exactly this kind of
precise-localization task.

Deliberately small (base_channels=16, 3 downsampling levels) rather than a
textbook-sized U-Net (64+ base channels, 4-5 levels): with only 14 training
images, a larger model has enough capacity to simply memorize them instead
of learning generalizable bead appearance. Worth revisiting if you later
annotate more images or add heavy augmentation.
"""

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with ReLU — the basic repeated unit in U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()

        # Encoder: each level halves spatial resolution, doubles channels
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck: the most compressed representation
        self.bottleneck = conv_block(base_channels * 4, base_channels * 8)

        # Decoder: each level upsamples, then concatenates the matching
        # encoder output (the "skip connection") before convolving
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels * 2, base_channels)

        # Final 1x1 conv maps back to a single-channel heatmap;
        # sigmoid squashes output to [0, 1], matching our heatmap targets
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.sigmoid(self.out_conv(d1))