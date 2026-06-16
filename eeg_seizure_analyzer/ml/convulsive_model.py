"""1D CNN classifier for the convulsive-vs-not decision on seizure crops.

This is Stage 2 of the seizure→convulsive cascade.  Stage 1 (``SeizureUNet``)
detects seizures; this classifier is run on each detected seizure crop to set
the convulsive flag, replacing the detector's ch1 signal.

Because every convulsive event is a seizure (convulsive ⊂ seizure), the task is
"given a seizure, is it convulsive?" — a balanced binary classification on
60 s @ 250 Hz crops, rather than the heavily imbalanced per-sample segmentation
the detector's ch1 has to solve.

Input:  (batch, n_channels, n_samples) — one EEG channel at 250 Hz
Output: (batch, 1)                     — a single logit (use sigmoid for prob)

The encoder reuses the detector's ``ConvBlock``/``DownBlock`` so the two models
share the same convolutional vocabulary; a global average pool collapses the
time axis and a linear head produces the scalar logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eeg_seizure_analyzer.ml.model import ConvBlock, DownBlock


class ConvulsiveClassifier(nn.Module):
    """1D CNN that maps a seizure crop to a convulsive logit.

    Parameters
    ----------
    in_channels : int
        Number of input channels (EEG channels; activity is not used).
    base_filters : int
        Filters in the first layer. Doubles at each downsampling stage.
    depth : int
        Number of downsampling stages.
    dropout : float
        Dropout probability before the linear head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth

        self.enc_input = ConvBlock(in_channels, base_filters)
        self.downs = nn.ModuleList()
        ch = base_filters
        for _ in range(depth):
            out_ch = ch * 2
            self.downs.append(DownBlock(ch, out_ch))
            ch = out_ch

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, in_channels, n_samples)

        Returns
        -------
        (batch, 1) — logits, use sigmoid for the convulsive probability.
        """
        x = self.enc_input(x)
        for down in self.downs:
            x = down(x)
        x = self.pool(x).squeeze(-1)  # (batch, ch)
        x = self.dropout(x)
        return self.head(x)  # (batch, 1)

    def predict_proba(self, x):
        """Forward pass with sigmoid applied. Returns (batch, 1) in [0, 1]."""
        return torch.sigmoid(self.forward(x))


def build_convulsive_classifier(
    n_eeg_channels: int = 1,
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.3,
) -> ConvulsiveClassifier:
    """Create a ConvulsiveClassifier.

    Parameters
    ----------
    n_eeg_channels : number of EEG input channels (activity is not used)
    base_filters : filters in first layer
    depth : number of downsampling stages
    dropout : dropout rate before the linear head

    Returns
    -------
    ConvulsiveClassifier
    """
    return ConvulsiveClassifier(
        in_channels=n_eeg_channels,
        base_filters=base_filters,
        depth=depth,
        dropout=dropout,
    )
