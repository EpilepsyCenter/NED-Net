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


class TemporalContext(nn.Module):
    """Long-range context at the U-Net bottleneck.

    The plain U-Net sees about 4 s of signal per output sample (its receptive
    field). This block lets every bottleneck step see the whole window, either
    through a bidirectional GRU or through self-attention. It is residual, so at
    initialisation it adds little to the conv features it receives.

    Parameters
    ----------
    channels : channels of the bottleneck feature map
    kind : "gru" or "transformer"
    dim : working width (projected down from ``channels`` and back)
    layers : GRU or transformer layers
    heads : attention heads (transformer only)
    dropout : dropout inside the block
    """

    def __init__(self, channels: int, kind: str, dim: int = 256,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        self.proj_in = nn.Conv1d(channels, dim, kernel_size=1)
        if kind == "gru":
            self.seq = nn.GRU(dim, dim // 2, num_layers=layers, batch_first=True,
                              bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                dropout=dropout, batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown context kind {kind!r}")
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv1d(dim, channels, kernel_size=1)

    @staticmethod
    def _positions(n: int, dim: int, device, dtype) -> torch.Tensor:
        """Sinusoidal positions (n, dim); works for any window length."""
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        ang = pos / torch.pow(10000.0, i / dim)
        pe = torch.zeros(n, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang)
        return pe.to(dtype)

    def forward(self, x):
        h = self.proj_in(x).transpose(1, 2)          # (B, L, dim)
        if self.kind == "gru":
            h, _ = self.seq(h)
        else:
            h = h + self._positions(h.size(1), h.size(2), h.device, h.dtype)
            h = self.seq(h)
        h = self.norm(h).transpose(1, 2)             # (B, dim, L)
        return x + self.proj_out(h)


BOTTLENECKS = ("conv", "gru", "transformer")


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
    bottleneck : str
        "conv" (the original U-Net), or "gru" / "transformer" to add a
        whole-window context block after the conv bottleneck.
    max_filters : int
        Cap on the filter count per level (0 = no cap). Lets a deeper U-Net
        widen its receptive field without the bottleneck reaching thousands of
        channels.
    context_dim, context_layers, context_heads : int
        Size of the context block (ignored for "conv").

    With the defaults the layers and their names are exactly those of models
    trained before these options existed, so their weights still load.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        n_classes: int = 2,
        bottleneck: str = "conv",
        max_filters: int = 0,
        context_dim: int = 256,
        context_layers: int = 2,
        context_heads: int = 4,
    ):
        super().__init__()
        if bottleneck not in BOTTLENECKS:
            raise ValueError(f"bottleneck must be one of {BOTTLENECKS}, not {bottleneck!r}")
        self.in_channels = in_channels
        self.depth = depth
        self.n_classes = n_classes
        self.bottleneck_kind = bottleneck

        def widen(c: int) -> int:
            return min(c * 2, max_filters) if max_filters > 0 else c * 2

        # Encoder
        self.enc_input = ConvBlock(in_channels, base_filters)
        self.encoders = nn.ModuleList()
        ch = base_filters
        skip_ch = [ch]
        for i in range(depth):
            out_ch = widen(ch)
            self.encoders.append(DownBlock(ch, out_ch))
            ch = out_ch
            skip_ch.append(ch)

        # Bottleneck
        out_ch = widen(ch)
        self.bottleneck = DownBlock(ch, out_ch)
        ch = out_ch
        self.context = (
            TemporalContext(ch, bottleneck, dim=context_dim,
                            layers=context_layers, heads=context_heads)
            if bottleneck != "conv" else None
        )

        # Decoder: each level upsamples to its skip connection's width
        self.decoders = nn.ModuleList()
        for skip in reversed(skip_ch):
            self.decoders.append(UpBlock(ch, skip))
            ch = skip

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

        # Bottleneck (+ whole-window context)
        x = self.bottleneck(x)
        if self.context is not None:
            x = self.context(x)

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
    bottleneck: str = "conv",
    max_filters: int = 0,
    context_dim: int = 256,
    context_layers: int = 2,
    context_heads: int = 4,
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
    bottleneck, max_filters, context_* : see SeizureUNet

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
        bottleneck=bottleneck,
        max_filters=max_filters,
        context_dim=context_dim,
        context_layers=context_layers,
        context_heads=context_heads,
    )
