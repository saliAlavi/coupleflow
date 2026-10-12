"""VLAAI — PyTorch port of the official TensorFlow implementation.

Upstream: ``upstream/vlaai/model.py`` (github.com/exporl/vlaai, commit
recorded in ``upstream/vlaai/GIT_COMMIT``), accompanying

    Accou, Vanthornhout, Van hamme & Francart (2023).
    "Decoding of the speech envelope from EEG using the VLAAI deep neural
    network." *Scientific Reports* 13, 812.

The port is layer-for-layer identical to the Keras graph.  Three details that
a naive rewrite gets wrong and that are reproduced here:

1. **Weight sharing.** ``vlaai()`` builds *one* ``extractor`` model and *one*
   ``output_context`` model and calls each of them ``nb_blocks`` times; a
   Keras ``Model`` called repeatedly shares its weights.  Only the
   ``Dense(64)`` between them is a fresh layer per block.  Verified against
   the released ``pretrained_models/vlaai.h5``, which contains exactly five
   extractor convolutions, one output-context convolution and four dense
   layers -> 1 740 289 parameters.  (Re-implementations that give every block
   its own extractor land at ~6.9 M.)
2. **Normalisation axis.** ``tf.keras.layers.LayerNormalization()`` defaults to
   ``axis=-1``, i.e. it normalises across *channels at each time step* on a
   ``(batch, time, channels)`` tensor — not across channels *and* time.  Its
   default ``epsilon`` is ``1e-3``, not PyTorch's ``1e-5``.
3. **LeakyReLU slope.** Keras defaults to ``alpha=0.3``; PyTorch defaults to
   ``0.01``.

Padding follows the Keras graph: the extractor convolutions are ``valid`` and
followed by ``ZeroPadding1D((0, k - 1))`` (right pad, anti-causal), while the
output-context convolution is preceded by ``ZeroPadding1D((k - 1, 0))``
(left pad, causal).  Both keep the time axis at its input length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_LN_EPS = 1e-3
KERAS_LEAKY_SLOPE = 0.3


class ChannelLayerNorm(nn.Module):
    """``tf.keras.layers.LayerNormalization()`` for ``(B, C, T)`` tensors.

    Normalises over the channel axis at every time step, matching Keras'
    ``axis=-1`` default on ``(batch, time, channels)`` inputs.
    """

    def __init__(self, n_channels: int, eps: float = KERAS_LN_EPS):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(B, C, T)``. Returns: ``(B, C, T)``."""
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Extractor(nn.Module):
    """The shared CNN stack (``extractor`` in ``model.py``).

    Args:
        filters: Output channels per convolution.
        kernels: Kernel size per convolution.
        input_channels: Channels of the incoming EEG-shaped tensor.
    """

    def __init__(
        self,
        filters: tuple[int, ...] = (256, 256, 256, 128, 128),
        kernels: tuple[int, ...] = (8,) * 5,
        input_channels: int = 64,
    ):
        super().__init__()
        if len(filters) != len(kernels):
            raise ValueError("'filters' and 'kernels' must have the same length")
        self.kernels = tuple(kernels)
        convs, norms = [], []
        in_ch = input_channels
        for f, k in zip(filters, kernels):
            convs.append(nn.Conv1d(in_ch, f, kernel_size=k))
            norms.append(ChannelLayerNorm(f))
            in_ch = f
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(B, C, T)``. Returns: ``(B, filters[-1], T)``."""
        for conv, norm, k in zip(self.convs, self.norms, self.kernels):
            x = conv(x)                                   # valid -> T - k + 1
            x = norm(x)
            x = F.leaky_relu(x, KERAS_LEAKY_SLOPE)
            x = F.pad(x, (0, k - 1))                      # ZeroPadding1D((0,k-1))
        return x


class OutputContext(nn.Module):
    """The shared causal output-context convolution.

    Args:
        filter_: Output channels.
        kernel: Kernel size (receptive field into the past).
        input_channels: Input channels.
    """

    def __init__(self, filter_: int = 64, kernel: int = 32,
                 input_channels: int = 64):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(input_channels, filter_, kernel_size=kernel)
        self.norm = ChannelLayerNorm(filter_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(B, C, T)``. Returns: ``(B, filter_, T)``."""
        x = F.pad(x, (self.kernel - 1, 0))                # ZeroPadding1D((k-1,0))
        x = self.conv(x)
        x = self.norm(x)
        return F.leaky_relu(x, KERAS_LEAKY_SLOPE)


class VLAAI(nn.Module):
    """The VLAAI decoder.

    Args:
        nb_blocks: Repeated blocks (paper: N = 4).
        input_channels: EEG channels.
        output_dim: Output channels (1 = broadband envelope).
        use_skip: Feed ``eeg + x`` to the extractor instead of ``x``.
        extractor_filters: Filters of the CNN stack.
        extractor_kernels: Kernels of the CNN stack.
        context_filter: Output-context filters.
        context_kernel: Output-context kernel.
        share_blocks: Reuse one extractor / output-context across blocks, as
            the official code and released checkpoint do.
    """

    def __init__(
        self,
        nb_blocks: int = 4,
        input_channels: int = 64,
        output_dim: int = 1,
        use_skip: bool = True,
        extractor_filters: tuple[int, ...] = (256, 256, 256, 128, 128),
        extractor_kernels: tuple[int, ...] = (8,) * 5,
        context_filter: int = 64,
        context_kernel: int = 32,
        share_blocks: bool = True,
    ):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.use_skip = use_skip
        self.share_blocks = share_blocks

        def _extractor() -> Extractor:
            return Extractor(extractor_filters, extractor_kernels, input_channels)

        def _context() -> OutputContext:
            return OutputContext(context_filter, context_kernel, input_channels)

        n_unique = 1 if share_blocks else nb_blocks
        self.extractors = nn.ModuleList([_extractor() for _ in range(n_unique)])
        self.output_contexts = nn.ModuleList([_context() for _ in range(n_unique)])
        # Dense(input_channels) is a *fresh* Keras layer inside every block.
        self.dense_layers = nn.ModuleList([
            nn.Linear(extractor_filters[-1], input_channels)
            for _ in range(nb_blocks)
        ])
        self.final_dense = nn.Linear(context_filter, output_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Decode the speech envelope from EEG.

        Args:
            eeg: ``(B, T, input_channels)`` — same layout as the Keras model.

        Returns:
            ``(B, T, output_dim)`` envelope estimate.
        """
        e = eeg.transpose(1, 2)                            # (B, C, T)
        x = torch.zeros_like(e) if self.use_skip else e
        for i in range(self.nb_blocks):
            j = 0 if self.share_blocks else i
            h = self.extractors[j](e + x if self.use_skip else x)
            h = self.dense_layers[i](h.transpose(1, 2)).transpose(1, 2)
            x = self.output_contexts[j](h)
        return self.final_dense(x.transpose(1, 2))


def build_vlaai(**kwargs) -> VLAAI:
    """Instantiate VLAAI with the paper's defaults."""
    return VLAAI(**kwargs)


__all__ = ["VLAAI", "Extractor", "OutputContext", "ChannelLayerNorm",
           "build_vlaai"]
