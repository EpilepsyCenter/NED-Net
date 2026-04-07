"""1D U-Net for per-sample seizure detection.

Input:  (batch, n_channels, n_samples) — multi-channel EEG at 250 Hz
Output: (batch, n_classes, n_samples)  — per-sample probability

n_classes=2 by default: channel 0 = seizure, channel 1 = convulsive.
Models trained with n_classes=1 (legacy) only predict seizure.

The architecture uses a standard encoder-decoder with skip connections.
The encoder progressively downsamples to capture patterns at multiple
timescales (spike morphology, rhythmicity, evolution over seconds).
The decoder upsamples back to full resolution for precise onset/offset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 1D convolutions with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample by 2x then apply ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample by 2x, concatenate skip, then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd-length inputs
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SeizureUNet(nn.Module):
    """1D U-Net for per-sample seizure segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (EEG channels + optional activity channels).
    base_filters : int
        Number of filters in the first layer. Doubles at each encoder level.
    depth : int
        Number of encoder/decoder levels (excluding the bottleneck).
    dropout : float
        Dropout probability applied before the final output layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.n_classes = n_classes

        # Encoder
        self.enc_input = ConvBlock(in_channels, base_filters)
        self.encoders = nn.ModuleList()
        ch = base_filters
        for i in range(depth):
            out_ch = ch * 2
            self.encoders.append(DownBlock(ch, out_ch))
            ch = out_ch

        # Bottleneck
        self.bottleneck = DownBlock(ch, ch * 2)
        ch = ch * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth + 1):
            out_ch = ch // 2
            self.decoders.append(UpBlock(ch, out_ch))
            ch = out_ch

        # Output — n_classes channels (seizure + convulsive)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(ch, n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, in_channels, n_samples)

        Returns
        -------
        (batch, n_classes, n_samples) — logits, use sigmoid for probabilities.
        Channel 0 = seizure, channel 1 = convulsive (if n_classes >= 2).
        """
        # Encoder path — collect skip connections
        skips = []

        x = self.enc_input(x)
        skips.append(x)

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path — use skip connections in reverse
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = dec(x, skip)

        x = self.dropout(x)
        x = self.out_conv(x)
        return x

    def predict_proba(self, x):
        """Forward pass with sigmoid applied.

        Returns
        -------
        (batch, n_classes, n_samples) — values in [0, 1]
        """
        return torch.sigmoid(self.forward(x))


def build_model(
    n_eeg_channels: int,
    include_activity: bool = False,
    n_activity_channels: int = 0,
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.2,
    n_classes: int = 2,
) -> SeizureUNet:
    """Create a SeizureUNet with the right number of input channels.

    Parameters
    ----------
    n_eeg_channels : number of EEG channels
    include_activity : whether activity channels are included
    n_activity_channels : number of activity channels (used if include_activity)
    base_filters : filters in first layer
    depth : encoder depth
    dropout : dropout rate
    n_classes : number of output channels (2 = seizure + convulsive)

    Returns
    -------
    SeizureUNet
    """
    in_ch = n_eeg_channels
    if include_activity:
        in_ch += n_activity_channels

    return SeizureUNet(
        in_channels=in_ch,
        base_filters=base_filters,
        depth=depth,
        dropout=dropout,
        n_classes=n_classes,
    )
