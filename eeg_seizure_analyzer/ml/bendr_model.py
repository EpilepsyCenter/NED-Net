"""BENDR-based seizure segmentation model.

Adapted from the BENDR architecture (Kostas et al., 2021, Frontiers in
Human Neuroscience) — a self-supervised EEG representation learner based
on wav2vec 2.0.  Original code: github.com/SPOClab-ca/BENDR (Apache 2.0).

Core components extracted from the DN3 framework and made standalone:

- **ConvEncoderBENDR**: 6-layer 1D CNN that compresses raw EEG by 96×
  (at 256 Hz → ~2.67 Hz effective rate, 375 ms per token).
- **BENDRContextualizer**: 8-layer transformer that learns temporal
  context across encoded tokens via masked prediction pre-training.
- **BENDRSegmentation**: Decoder that upsamples contextualised
  representations back to per-sample resolution for seizure onset/offset
  detection.  Output format matches ``SeizureUNet``:
  ``(batch, n_classes, n_samples)`` with sigmoid → probabilities.
- **BENDRPretrainModel**: Contrastive masked prediction wrapper for
  self-supervised pre-training on unlabelled EEG data.

Pre-training produces encoder + contextualizer weights.  Fine-tuning
attaches the segmentation decoder and trains on annotated seizure data,
using the same loss / metrics / dataset pipeline as the U-Net.

Input:  (batch, n_channels, n_samples) — EEG at 256 Hz
Output: (batch, n_classes, n_samples) — per-sample logits
"""

from __future__ import annotations

import copy
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility layers (replacing DN3 dependencies) ──────────────────────


class _Permute(nn.Module):
    """Permute tensor dimensions.  Replaces ``dn3.trainable.layers.Permute``."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class _Hax(nn.Module):
    """Identity layer — T-fixup removes self-attention LayerNorms."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Masking utilities ────────────────────────────────────────────────


def _make_span_from_seeds(
    seeds: np.ndarray, span: int, total: int | None = None,
) -> np.ndarray:
    """Expand seed indices into contiguous spans."""
    inds: list[int] = []
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            if i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(
    shape: tuple[int, int],
    p: float,
    total: int,
    span: int,
    allow_no_inds: bool = False,
) -> torch.Tensor:
    """Create a boolean mask with random contiguous spans."""
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)
    for i in range(shape[0]):
        mask_seeds: list[int] = []
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0].tolist()
        mask[i, _make_span_from_seeds(np.array(mask_seeds), span, total=total)] = True
    return mask


# ── Convolutional Encoder ────────────────────────────────────────────


class ConvEncoderBENDR(nn.Module):
    """6-block 1D CNN that compresses raw EEG into learned representations.

    Default configuration: kernel widths ``(3, 2, 2, 2, 2, 2)`` with
    matching strides → total downsampling factor of 96×.
    Each block: ``Conv1d → Dropout → GroupNorm → GELU``.

    Parameters
    ----------
    in_features : int
        Number of input EEG channels.
    encoder_h : int
        Hidden dimension (number of filters in every layer).
    enc_width : tuple[int, ...]
        Kernel widths for each convolutional block.
    enc_downsample : tuple[int, ...]
        Stride (downsampling factor) for each block.
    dropout : float
        Dropout probability after each convolution.
    projection_head : bool
        If True, append an extra 1×1 conv projection layer.
    """

    def __init__(
        self,
        in_features: int,
        encoder_h: int = 512,
        enc_width: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        enc_downsample: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
        projection_head: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h
        self._downsampling = list(enc_downsample)
        self._width = [w if w % 2 else w + 1 for w in enc_width]

        self.encoder = nn.Sequential()
        ch_in = in_features
        for i, (width, stride) in enumerate(zip(self._width, self._downsampling)):
            self.encoder.add_module(
                f"Encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(ch_in, encoder_h, width, stride=stride, padding=width // 2),
                    nn.Dropout1d(dropout),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            ch_in = encoder_h

        if projection_head:
            self.encoder.add_module(
                "projection-1",
                nn.Sequential(
                    nn.Conv1d(encoder_h, encoder_h, 1),
                    nn.Dropout1d(dropout * 2),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )

    @property
    def total_downsampling(self) -> int:
        """Total downsampling factor across all blocks."""
        result = 1
        for d in self._downsampling:
            result *= d
        return result

    def downsampling_factor(self, samples: int) -> int:
        """Compute number of output tokens for a given input length."""
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, samples) → (batch, encoder_h, encoded_samples)"""
        return self.encoder(x)

    def freeze(self, frozen: bool = True) -> None:
        """Freeze or unfreeze all encoder parameters."""
        for p in self.parameters():
            p.requires_grad = not frozen


# ── Encoding Augment (masking + positional encoding) ─────────────────


class EncodingAugment(nn.Module):
    """Temporal/channel masking and positional encoding for encoded features.

    Used during pre-training (heavy masking) and optionally during
    fine-tuning (light masking for regularisation).

    Parameters
    ----------
    in_features : int
        Encoder hidden dimension (``encoder_h``).
    mask_p_t, mask_p_c : float
        Probability of masking each temporal / channel position.
    mask_t_span, mask_c_span : int
        Length of contiguous masked spans.
    dropout : float
        Dropout after layer norm.
    position_encoder : int
        Kernel size for the grouped convolution positional encoder.
    """

    def __init__(
        self,
        in_features: int,
        mask_p_t: float = 0.1,
        mask_p_c: float = 0.01,
        mask_t_span: int = 6,
        mask_c_span: int = 64,
        dropout: float = 0.1,
        position_encoder: int = 25,
    ):
        super().__init__()
        self.mask_replacement = nn.Parameter(torch.zeros(in_features))
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(
            in_features, in_features, position_encoder,
            padding=position_encoder // 2, groups=16,
        )
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            _Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            _Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, seq, self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, feat, self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename: str | Path) -> None:
        """Load mask embedding and position encoder from contextualizer."""
        state_dict = torch.load(filename, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        for p in self.parameters():
            p.requires_grad = False


# ── Transformer Contextualizer ───────────────────────────────────────


class BENDRContextualizer(nn.Module):
    """Transformer encoder that learns temporal context across tokens.

    Uses T-fixup initialisation (removes LayerNorm from self-attention)
    and grouped-convolution positional encoding.

    Parameters
    ----------
    in_features : int
        Input dimension (``encoder_h`` from the CNN encoder).
    hidden_feedforward : int
        Feedforward dimension in transformer layers.
    heads : int
        Number of attention heads.
    layers : int
        Number of transformer layers.
    dropout : float
        Dropout in transformer layers.
    position_encoder : int
        Kernel size for positional encoding convolution.
    layer_drop : float
        Probability of dropping each layer during training.
    mask_p_t, mask_p_c : float
        Masking probabilities for fine-tuning regularisation.
    mask_t_span, mask_c_span : int
        Masking span lengths.
    start_token : float or None
        If not None, prepend a learnable start token.
    finetuning : bool
        If True, apply light masking during training for regularisation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_feedforward: int = 3076,
        heads: int = 8,
        layers: int = 8,
        dropout: float = 0.15,
        activation: str = "gelu",
        position_encoder: int = 25,
        layer_drop: float = 0.0,
        mask_p_t: float = 0.1,
        mask_p_c: float = 0.004,
        mask_t_span: int = 6,
        mask_c_span: int = 64,
        start_token: float | None = -5,
        finetuning: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        # Build transformer layers with T-fixup (norms replaced by identity)
        layer_template = nn.TransformerEncoderLayer(
            d_model=self._transformer_dim,
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        layer_template.norm1 = _Hax()
        layer_template.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(layer_template) for _ in range(layers)]
        )

        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Learnable mask replacement vector
        self.mask_replacement = nn.Parameter(
            torch.normal(0, in_features ** (-0.5), size=(in_features,)),
        )

        # Positional encoding via grouped convolution
        self.position_encoder = position_encoder > 0
        if self.position_encoder:
            conv = nn.Conv1d(
                in_features, in_features, position_encoder,
                padding=position_encoder // 2, groups=16,
            )
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        # Input conditioning: LayerNorm → dropout → project to transformer dim
        self.input_conditioning = nn.Sequential(
            _Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            _Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            _Permute([2, 0, 1]),  # → (seq, batch, dim) for transformer
        )

        # Output projection back to encoder dimension
        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)

        # T-fixup initialisation
        self.apply(self._init_bert_params)

    def _init_bert_params(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # T-fixup scaling
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )

    def forward(
        self,
        x: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_c: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """(batch, features, seq) → (batch, features, seq).

        If ``return_hidden`` is True, also return the list of per-layer
        transformer hidden states (each ``(seq[+start], batch, transformer_dim)``)
        — used by the data2vec pre-training objective for top-K target averaging.
        """
        bs, feat, seq = x.shape

        # Fine-tuning masking (light regularisation)
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, seq, self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, feat, self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        # Positional encoding + conditioning
        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)  # → (seq, batch, transformer_dim)

        # Optional start token
        if self.start_token is not None:
            token = (
                self.start_token
                * torch.ones((1, 1, 1), device=x.device)
                .expand(-1, *x.shape[1:])
            )
            x = torch.cat([token, x], dim=0)

        # Transformer layers with stochastic depth
        hidden_states: list[torch.Tensor] = []
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1).item() > self.layer_drop:
                x = layer(x)
            if return_hidden:
                hidden_states.append(x)

        # → (batch, features, seq) and project back to encoder_h
        out = self.output_layer(x.permute(1, 2, 0))
        if return_hidden:
            return out, hidden_states
        return out

    def freeze(self, frozen: bool = True) -> None:
        """Freeze or unfreeze all contextualizer parameters."""
        for p in self.parameters():
            p.requires_grad = not frozen
        if frozen and self.finetuning:
            self.mask_replacement.requires_grad = False


# ── Segmentation Decoder ─────────────────────────────────────────────


class BENDRSegmentationDecoder(nn.Module):
    """Upsample contextualised representations to per-sample predictions.

    Mirrors the encoder's downsampling factors in reverse using transposed
    convolutions, producing output at the original sampling rate.

    Output shape matches ``SeizureUNet``:
    ``(batch, n_classes, n_samples)`` — logits (apply sigmoid for probs).

    Parameters
    ----------
    encoder_h : int
        Hidden dimension from the encoder/contextualizer.
    n_classes : int
        Number of output classes (2 = seizure + convulsive).
    enc_downsample : tuple[int, ...]
        Downsampling factors from the encoder (reversed for upsampling).
    decoder_h : int
        Hidden dimension in decoder layers (gradually reduces).
    dropout : float
        Dropout before final output layer.
    """

    def __init__(
        self,
        encoder_h: int = 512,
        n_classes: int = 2,
        enc_downsample: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        decoder_h: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Build upsampling layers in reverse order of encoder strides
        upsample_factors = list(reversed(enc_downsample))

        layers: list[nn.Module] = []
        ch_in = encoder_h
        for i, factor in enumerate(upsample_factors):
            # Gradually reduce channels in later layers
            if i < len(upsample_factors) - 1:
                ch_out = max(decoder_h, encoder_h // (2 ** (i + 1)))
            else:
                ch_out = decoder_h // 2

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        ch_in, ch_out, kernel_size=factor * 2,
                        stride=factor, padding=factor // 2,
                    ),
                    nn.GroupNorm(max(1, ch_out // 8), ch_out),
                    nn.GELU(),
                )
            )
            ch_in = ch_out

        self.upsample = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(ch_in, n_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, target_length: int | None = None,
    ) -> torch.Tensor:
        """(batch, encoder_h, encoded_seq) → (batch, n_classes, n_samples)

        Parameters
        ----------
        x : torch.Tensor
            Contextualised encoder output.
        target_length : int, optional
            If given, pad/trim output to exactly this length.
        """
        for layer in self.upsample:
            x = layer(x)

        x = self.dropout(x)
        x = self.out_conv(x)

        # Trim or pad to exact target length if specified
        if target_length is not None:
            if x.size(2) > target_length:
                x = x[:, :, :target_length]
            elif x.size(2) < target_length:
                x = F.pad(x, (0, target_length - x.size(2)))

        return x


# ── Full Segmentation Model (for fine-tuning) ────────────────────────


class BENDRSeizureModel(nn.Module):
    """BENDR encoder + contextualizer + segmentation decoder.

    End-to-end model for per-sample seizure detection.  Architecture:

        raw EEG → ConvEncoder → Contextualizer → SegmentationDecoder → logits

    Output format is identical to ``SeizureUNet``:
    ``(batch, n_classes, n_samples)`` — apply sigmoid for probabilities.

    Parameters
    ----------
    in_channels : int
        Number of input EEG channels.
    encoder_h : int
        Hidden dimension for encoder and contextualizer.
    n_classes : int
        Output classes (2 = seizure + convulsive).
    enc_width : tuple
        Kernel widths for encoder blocks.
    enc_downsample : tuple
        Stride factors for encoder blocks.
    context_layers : int
        Number of transformer layers in contextualizer.
    context_heads : int
        Number of attention heads.
    context_feedforward : int
        Feedforward dimension in transformer.
    context_dropout : float
        Dropout in transformer layers.
    finetuning : bool
        Enable light masking in contextualizer for regularisation.
    decoder_h : int
        Hidden dimension in decoder layers.
    decoder_dropout : float
        Dropout before final decoder output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_h: int = 512,
        n_classes: int = 2,
        enc_width: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        enc_downsample: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        context_layers: int = 8,
        context_heads: int = 8,
        context_feedforward: int = 3076,
        context_dropout: float = 0.15,
        finetuning: bool = True,
        decoder_h: int = 256,
        decoder_dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.encoder_h = encoder_h
        self._enc_downsample = enc_downsample

        self.encoder = ConvEncoderBENDR(
            in_features=in_channels,
            encoder_h=encoder_h,
            enc_width=enc_width,
            enc_downsample=enc_downsample,
        )

        self.contextualizer = BENDRContextualizer(
            in_features=encoder_h,
            hidden_feedforward=context_feedforward,
            heads=context_heads,
            layers=context_layers,
            dropout=context_dropout,
            finetuning=finetuning,
            start_token=None,  # no start token for segmentation
        )

        self.decoder = BENDRSegmentationDecoder(
            encoder_h=encoder_h,
            n_classes=n_classes,
            enc_downsample=enc_downsample,
            decoder_h=decoder_h,
            dropout=decoder_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, in_channels, n_samples) → (batch, n_classes, n_samples)"""
        target_length = x.size(2)
        z = self.encoder(x)
        c = self.contextualizer(z)
        return self.decoder(c, target_length=target_length)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid applied → values in [0, 1]."""
        return torch.sigmoid(self.forward(x))

    def load_pretrained(
        self,
        encoder_path: str | Path,
        contextualizer_path: str | Path | None = None,
        freeze_encoder: bool = False,
        freeze_contextualizer: bool = False,
    ) -> None:
        """Load pre-trained encoder and contextualizer weights.

        Parameters
        ----------
        encoder_path : path
            Path to encoder state dict (``.pt``).
        contextualizer_path : path, optional
            Path to contextualizer state dict.  If None, only encoder is loaded.
        freeze_encoder : bool
            If True, freeze encoder weights (only train decoder).
        freeze_contextualizer : bool
            If True, freeze contextualizer weights.
        """
        enc_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(enc_state, strict=True)

        if contextualizer_path is not None:
            ctx_state = torch.load(
                contextualizer_path, map_location="cpu", weights_only=True,
            )
            self.contextualizer.load_state_dict(ctx_state, strict=True)

        if freeze_encoder:
            self.encoder.freeze(True)
        if freeze_contextualizer:
            self.contextualizer.freeze(True)

    def load_pretrained_combined(
        self,
        checkpoint_path: str | Path,
        freeze_encoder: bool = False,
        freeze_contextualizer: bool = False,
    ) -> None:
        """Load pre-trained weights from a combined checkpoint.

        The checkpoint should contain ``encoder`` and ``contextualizer``
        keys, each mapping to the respective state dict.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
        self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.contextualizer.load_state_dict(
            checkpoint["contextualizer"], strict=True,
        )

        if freeze_encoder:
            self.encoder.freeze(True)
        if freeze_contextualizer:
            self.contextualizer.freeze(True)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (encoder, contextualizer, decoder)."""
        self.encoder.freeze(False)
        self.contextualizer.freeze(False)
        for p in self.decoder.parameters():
            p.requires_grad = True


# ── Pre-training Model ───────────────────────────────────────────────


class BENDRPretrainModel(nn.Module):
    """Contrastive masked prediction for self-supervised pre-training.

    Wraps an encoder and contextualizer.  During training:

    1. Encode raw EEG → feature sequence ``z``
    2. Mask random spans of ``z``
    3. Pass masked ``z`` through contextualizer → ``c``
    4. Contrastive loss: predict correct ``z`` at masked positions
       against negative samples drawn from the same sequence

    This is the wav2vec 2.0 objective adapted for EEG.

    Parameters
    ----------
    encoder : ConvEncoderBENDR
        The convolutional encoder to pre-train.
    contextualizer : BENDRContextualizer
        The transformer contextualizer to pre-train.
    mask_rate : float
        Probability of masking each temporal position.
    mask_span : int
        Length of each contiguous mask span.
    temp : float
        Temperature for cosine similarity scaling.
    num_negatives : int
        Number of negative samples per masked position.
    enc_feat_l2 : float
        L2 regularisation weight on encoder features.
    """

    def __init__(
        self,
        encoder: ConvEncoderBENDR,
        contextualizer: BENDRContextualizer,
        mask_rate: float = 0.1,
        mask_span: int = 6,
        temp: float = 0.5,
        num_negatives: int = 100,
        enc_feat_l2: float = 0.001,
    ):
        super().__init__()
        self.encoder = encoder
        self.contextualizer = contextualizer
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.num_negatives = num_negatives
        self.enc_feat_l2 = enc_feat_l2
        self.loss_fn = nn.CrossEntropyLoss()

    def _generate_negatives(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample negative examples from the same batch."""
        batch_size, feat, full_len = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, feat)

        with torch.no_grad():
            negative_inds = torch.randint(
                0, full_len - 1,
                size=(batch_size, full_len * self.num_negatives),
            )
            for i in range(1, batch_size):
                negative_inds[i] += i * full_len

        z_neg = z_flat[negative_inds.view(-1)].view(
            batch_size, full_len, self.num_negatives, feat,
        )
        return z_neg, negative_inds

    def _compute_logits(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity logits for contrastive loss."""
        # c has start token → skip first position
        c = c[..., 1:].permute(0, 2, 1).unsqueeze(-2)
        z = z.permute(0, 2, 1).unsqueeze(-2)

        # A sampled negative is "false" when it coincides with the positive
        # target z (cos = 1 would penalise the model for what is actually a
        # correct guess). Negatives are drawn from z, so the comparison must
        # be against z. Comparing against the contextualizer output c (a
        # different representation space) never fires, so false negatives
        # stayed in the denominator and held the objective at chance.
        neg_in_target = (z == negatives).all(-1)
        targets = torch.cat([c, negatives], dim=-2)

        # Compute the contrastive similarity in fp32 regardless of the
        # surrounding autocast dtype: cosine_similarity's internal eps is
        # fp32-sized, so a reduced-precision near-zero-norm vector could
        # otherwise yield 0/0 = NaN.
        with torch.autocast(z.device.type, enabled=False):
            logits = F.cosine_similarity(z.float(), targets.float(), dim=-1) / self.temp
        if neg_in_target.any():
            # Positive match is column 0; mask only the negative columns
            # (last dim) — NOT logits[1:], which sliced the batch dim.
            logits[..., 1:][neg_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute contrastive logits, encoded features, and mask.

        Returns
        -------
        logits : (N, 1 + num_negatives)
            Cosine similarity logits.  Label 0 is always the correct match.
        z : (batch, encoder_h, encoded_len)
            Encoded features (before masking).
        mask : (batch, encoded_len)
            Boolean mask indicating which positions were masked.
        """
        z = self.encoder(x)
        unmasked_z = z.clone()

        batch_size, feat, samples = z.shape

        if self.training:
            mask = _make_mask(
                (batch_size, samples), self.mask_rate, samples, self.mask_span,
            )
        else:
            mask = torch.zeros(
                (batch_size, samples), requires_grad=False, dtype=torch.bool,
            )
            half_seeds = max(1, int(samples * self.mask_rate * 0.5))
            seed_positions = (
                (samples // half_seeds)
                * np.arange(half_seeds).astype(int)
            )
            mask[:, _make_span_from_seeds(seed_positions, self.mask_span)] = True

        # Apply mask to encoder output. Cast the (fp32) mask-replacement
        # parameter to z's dtype so the in-place assignment works under
        # autocast, where z is reduced-precision but params stay fp32.
        mask_device = mask.to(z.device)
        z.transpose(2, 1)[mask_device] = self.contextualizer.mask_replacement.to(z.dtype)

        c = self.contextualizer(z)

        # Stop-gradient on the prediction targets. The positive (unmasked_z)
        # and the negatives drawn from it are *targets*, not predictions —
        # gradient must not flow into them. Without detach the encoder
        # minimises the loss by collapsing every z to be mutually equidistant
        # (the positive parks just under the all-equal negatives → acc pinned
        # at chance forever), which is exactly the failure seen on Arrhenius.
        # The non-detached unmasked_z is still returned so the L2 feature
        # penalty in compute_loss can regularise the encoder. This restores
        # the standard wav2vec 2.0 / BENDR objective.
        target_z = unmasked_z.detach()
        negatives, _ = self._generate_negatives(target_z)
        logits = self._compute_logits(target_z, c, negatives)

        return logits, unmasked_z, mask

    @staticmethod
    def select_masked_logits(
        logits: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep only the logit rows at masked positions.

        ``logits`` is ``(batch * seq, 1 + num_negatives)``, flattened
        batch-major to match ``_compute_logits``; ``mask`` is
        ``(batch, seq)``. The contrastive objective is defined *only* at
        masked positions — scoring the unmasked ones (which the
        contextualizer can copy straight through from the input) swamps the
        signal with trivial examples and pins accuracy at chance. wav2vec 2.0
        and BENDR both compute the loss on masked positions alone.
        """
        sel = mask.reshape(-1).to(logits.device)
        return logits[sel]

    def compute_loss(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contrastive loss (masked positions only) + L2 feature penalty.

        ``mask`` is the per-position boolean mask returned by ``forward``.
        It is optional only for backward compatibility; always pass it —
        omitting it scores every timestep and the objective cannot learn.
        """
        if mask is not None:
            logits = self.select_masked_logits(logits, mask)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, labels) + self.enc_feat_l2 * z.pow(2).mean()


# ── data2vec Pre-training Model ──────────────────────────────────────


class BENDRData2VecPretrainModel(nn.Module):
    """data2vec-style self-supervised pre-training (EMA target encoder).

    The contrastive objective (``BENDRPretrainModel``) collapses on continuous
    EEG targets: even with detached targets and masked-only loss, the
    representation degrades after an early peak (held-out accuracy returns to
    chance). data2vec (Baevski et al., 2022) is the principled cure and is
    essentially "BENDR done right" — no negatives, no quantization.

    A **student** (the trainable encoder + contextualizer) sees the *masked*
    feature sequence and, at the masked positions, regresses to targets produced
    by a **teacher** — an exponential-moving-average (EMA) copy of the same
    networks — run on the *unmasked* sequence. The teacher is never trained by
    gradient (only EMA-tracked), and its targets are the average of the top-K
    transformer layers, instance-normalised over time then layer-normed over
    features. The EMA lag + target normalisation are what structurally prevent
    the representation collapse.

    Parameters
    ----------
    encoder, contextualizer
        The student networks to pre-train (also EMA-copied into the teacher).
    mask_rate, mask_span
        Temporal masking, as in the contrastive model.
    ema_decay, ema_end_decay, ema_anneal_steps
        EMA momentum schedule: ``ema_decay`` at step 0, linearly annealed to
        ``ema_end_decay`` over ``ema_anneal_steps`` optimizer steps, then held.
    top_k_layers
        Number of final teacher transformer layers averaged into the target.
    """

    def __init__(
        self,
        encoder: ConvEncoderBENDR,
        contextualizer: BENDRContextualizer,
        mask_rate: float = 0.15,
        mask_span: int = 6,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_steps: int = 2000,
        top_k_layers: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.contextualizer = contextualizer
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_steps = ema_anneal_steps
        self.top_k_layers = top_k_layers

        transformer_dim = contextualizer._transformer_dim
        self.predictor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
        )

        # EMA teacher: frozen deep copies, updated only via update_ema().
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_contextualizer = copy.deepcopy(contextualizer)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        for p in self.ema_contextualizer.parameters():
            p.requires_grad_(False)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _current_decay(self) -> float:
        s = int(self.ema_step.item())
        if s >= self.ema_anneal_steps:
            return self.ema_end_decay
        frac = s / max(1, self.ema_anneal_steps)
        return self.ema_decay + (self.ema_end_decay - self.ema_decay) * frac

    @torch.no_grad()
    def update_ema(self) -> None:
        """EMA-track the teacher toward the student. Call after optimizer.step()."""
        d = self._current_decay()
        for pe, ps in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            pe.mul_(d).add_(ps.detach(), alpha=1 - d)
        for pe, ps in zip(
            self.ema_contextualizer.parameters(), self.contextualizer.parameters(),
        ):
            pe.mul_(d).add_(ps.detach(), alpha=1 - d)
        # Buffers (norm stats etc.) are copied outright, not averaged.
        for be, bs in zip(self.ema_encoder.buffers(), self.encoder.buffers()):
            be.copy_(bs)
        for be, bs in zip(
            self.ema_contextualizer.buffers(), self.contextualizer.buffers(),
        ):
            be.copy_(bs)
        self.ema_step += 1

    def _strip_start(self, h: torch.Tensor) -> torch.Tensor:
        """(seq[+start], batch, dim) → (batch, seq, dim), dropping the start token."""
        if self.contextualizer.start_token is not None:
            h = h[1:]
        return h.permute(1, 0, 2)

    def _make_mask_for(self, batch_size: int, samples: int) -> torch.Tensor:
        if self.training:
            return _make_mask(
                (batch_size, samples), self.mask_rate, samples, self.mask_span,
            )
        mask = torch.zeros((batch_size, samples), dtype=torch.bool)
        half_seeds = max(1, int(samples * self.mask_rate * 0.5))
        seed_positions = (samples // half_seeds) * np.arange(half_seeds).astype(int)
        mask[:, _make_span_from_seeds(seed_positions, self.mask_span)] = True
        return mask

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(pred, target, mask)``.

        pred, target : (batch, seq, transformer_dim)
            Student prediction and (stop-grad) teacher target.
        mask : (batch, seq) bool
            Masked positions — the loss is computed only here.
        """
        z = self.encoder(x)
        batch_size, feat, samples = z.shape

        mask = self._make_mask_for(batch_size, samples)
        mask_device = mask.to(z.device)

        z_masked = z.clone()
        z_masked.transpose(2, 1)[mask_device] = (
            self.contextualizer.mask_replacement.to(z.dtype)
        )

        # Student: masked input → top transformer hidden → predictor.
        _, student_hidden = self.contextualizer(z_masked, return_hidden=True)
        student_h = self._strip_start(student_hidden[-1])
        pred = self.predictor(student_h)

        # Teacher: unmasked input → averaged, normalised top-K targets (no grad).
        with torch.no_grad():
            self.ema_encoder.eval()
            self.ema_contextualizer.eval()
            z_teacher = self.ema_encoder(x)
            _, teacher_hidden = self.ema_contextualizer(z_teacher, return_hidden=True)
            top_k = teacher_hidden[-self.top_k_layers:]
            # Build the targets in fp32. Under bf16 autocast the normalisation
            # statistics (mean/std/division) lose all precision — a near-flat
            # feature gives a tiny std and the division explodes to huge or
            # non-finite targets (observed: ~79% of batches skipped on the A100).
            with torch.autocast(x.device.type, enabled=False):
                normed = []
                for h in top_k:
                    h = self._strip_start(h).float()  # (B, S, D), fp32
                    # Instance-normalise over time (per batch, per feature).
                    h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-4)
                    normed.append(h)
                target = sum(normed) / len(normed)
                target = F.layer_norm(target, (target.shape[-1],))

        return pred, target, mask_device

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth-L1 regression loss at masked positions only.

        Computed in fp32 (targets are fp32; pred is cast up) — the regression
        against normalised targets is numerically fragile under bf16 autocast.
        """
        with torch.autocast(pred.device.type, enabled=False):
            return F.smooth_l1_loss(pred[mask].float(), target[mask].float())


# ── Factory functions ────────────────────────────────────────────────


def build_bendr_model(
    n_eeg_channels: int = 1,
    encoder_h: int = 512,
    n_classes: int = 2,
    context_layers: int = 8,
    context_heads: int = 8,
    pretrained_path: str | Path | None = None,
    freeze_encoder: bool = False,
    freeze_contextualizer: bool = False,
    finetuning: bool = True,
    decoder_dropout: float = 0.2,
) -> BENDRSeizureModel:
    """Create a BENDR segmentation model, optionally loading pre-trained weights.

    Parameters
    ----------
    n_eeg_channels : int
        Number of input EEG channels.
    encoder_h : int
        Hidden dimension for encoder (512 = BENDR default).
    n_classes : int
        Output classes (2 = seizure + convulsive).
    context_layers : int
        Transformer layers (8 = BENDR default).
    context_heads : int
        Attention heads (8 = BENDR default).
    pretrained_path : path, optional
        Path to combined checkpoint with encoder + contextualizer weights.
    freeze_encoder : bool
        Freeze encoder during fine-tuning.
    freeze_contextualizer : bool
        Freeze contextualizer during fine-tuning.
    finetuning : bool
        Enable light masking regularisation in contextualizer.
    decoder_dropout : float
        Dropout in segmentation decoder.

    Returns
    -------
    BENDRSeizureModel
    """
    model = BENDRSeizureModel(
        in_channels=n_eeg_channels,
        encoder_h=encoder_h,
        n_classes=n_classes,
        context_layers=context_layers,
        context_heads=context_heads,
        finetuning=finetuning,
        decoder_dropout=decoder_dropout,
    )

    if pretrained_path is not None:
        model.load_pretrained_combined(
            pretrained_path,
            freeze_encoder=freeze_encoder,
            freeze_contextualizer=freeze_contextualizer,
        )

    return model


def build_pretrain_model(
    n_eeg_channels: int = 1,
    encoder_h: int = 512,
    context_layers: int = 8,
    context_heads: int = 8,
    mask_rate: float = 0.1,
    mask_span: int = 6,
    temp: float = 0.5,
    num_negatives: int = 100,
) -> BENDRPretrainModel:
    """Create a BENDR model configured for self-supervised pre-training.

    Parameters
    ----------
    n_eeg_channels : int
        Number of input EEG channels.
    encoder_h : int
        Hidden dimension.
    context_layers, context_heads : int
        Transformer configuration.
    mask_rate : float
        Probability of masking each temporal position.
    mask_span : int
        Length of contiguous mask spans.
    temp : float
        Contrastive loss temperature.
    num_negatives : int
        Number of negative samples per position.

    Returns
    -------
    BENDRPretrainModel
    """
    encoder = ConvEncoderBENDR(
        in_features=n_eeg_channels,
        encoder_h=encoder_h,
    )
    contextualizer = BENDRContextualizer(
        in_features=encoder_h,
        hidden_feedforward=3076,
        heads=context_heads,
        layers=context_layers,
        finetuning=False,
        start_token=-5,  # use start token during pre-training
    )
    return BENDRPretrainModel(
        encoder=encoder,
        contextualizer=contextualizer,
        mask_rate=mask_rate,
        mask_span=mask_span,
        temp=temp,
        num_negatives=num_negatives,
    )


def build_data2vec_pretrain_model(
    n_eeg_channels: int = 1,
    encoder_h: int = 512,
    context_layers: int = 8,
    context_heads: int = 8,
    mask_rate: float = 0.15,
    mask_span: int = 6,
    ema_decay: float = 0.999,
    ema_end_decay: float = 0.9999,
    ema_anneal_steps: int = 2000,
    top_k_layers: int = 4,
) -> BENDRData2VecPretrainModel:
    """Create a BENDR model configured for data2vec (EMA-teacher) pre-training.

    Drop-in alternative to :func:`build_pretrain_model` that avoids the
    continuous-target contrastive collapse. The encoder/contextualizer produced
    here are weight-compatible with :class:`BENDRSeizureModel`, so the resulting
    checkpoint fine-tunes through the same ``load_pretrained_combined`` path.

    Parameters
    ----------
    n_eeg_channels, encoder_h, context_layers, context_heads
        Architecture, identical to :func:`build_pretrain_model`.
    mask_rate, mask_span
        Temporal masking.
    ema_decay, ema_end_decay, ema_anneal_steps
        EMA teacher momentum schedule (see :class:`BENDRData2VecPretrainModel`).
    top_k_layers
        Number of final teacher layers averaged into the target.
    """
    encoder = ConvEncoderBENDR(
        in_features=n_eeg_channels,
        encoder_h=encoder_h,
    )
    contextualizer = BENDRContextualizer(
        in_features=encoder_h,
        hidden_feedforward=3076,
        heads=context_heads,
        layers=context_layers,
        finetuning=False,
        start_token=-5,
    )
    return BENDRData2VecPretrainModel(
        encoder=encoder,
        contextualizer=contextualizer,
        mask_rate=mask_rate,
        mask_span=mask_span,
        ema_decay=ema_decay,
        ema_end_decay=ema_end_decay,
        ema_anneal_steps=ema_anneal_steps,
        top_k_layers=top_k_layers,
    )
