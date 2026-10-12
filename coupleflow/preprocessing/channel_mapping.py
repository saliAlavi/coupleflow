"""Reorder and zero-pad EEG data to the canonical 64-channel layout.

Datasets with fewer than 64 channels are zero-padded in the missing
positions.  Extra channels that are not in the standard set (e.g. EXG,
Status) are silently dropped.
"""

from __future__ import annotations

import logging

import torch

from coupleflow.preprocessing.channels import CHANNEL_TO_INDEX, NUM_STANDARD_CHANNELS

logger = logging.getLogger(__name__)

# Common channel name aliases across different EEG systems.
# Maps variant names → canonical standard names.
_CHANNEL_ALIASES: dict[str, str] = {
    "P9": "TP9", "P10": "TP10",    # BioSemi vs 10-20 naming
    "T3": "T7", "T4": "T8",       # old 10-20 vs extended 10-20
    "T5": "P7", "T6": "P8",       # old 10-20 vs extended 10-20
    "FT9": "FT7", "FT10": "FT8",  # NJU 10-5 inferior → nearest 10-20
    "PO9": "PO7", "PO10": "PO8",  # NJU 10-5 inferior → nearest 10-20
}


def map_to_standard(
    eeg_data: torch.Tensor,
    source_channels: list[str],
    standard_channels: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder *eeg_data* from an arbitrary channel order to the canonical layout.

    Args:
        eeg_data: Raw EEG tensor of shape ``(n_source_channels, T)``.
        source_channels: Channel names in the order they appear in *eeg_data*.
            Names that are not part of the canonical 64-ch set are dropped.
        standard_channels: Optional override for the canonical channel list.
            When *None*, :data:`coupleflow.preprocessing.channels.STANDARD_64_CHANNELS` is used.

    Returns:
        A tuple ``(eeg_64ch, channel_mask)`` where

        * ``eeg_64ch`` has shape ``(64, T)`` with zeros in positions that
          have no matching source channel.
        * ``channel_mask`` is a boolean tensor of shape ``(64,)`` that is
          *True* for positions filled with real data.
    """
    if standard_channels is not None:
        ch_to_idx = {ch: i for i, ch in enumerate(standard_channels)}
        n_std = len(standard_channels)
    else:
        ch_to_idx = CHANNEL_TO_INDEX
        n_std = NUM_STANDARD_CHANNELS

    n_timepoints = eeg_data.shape[-1]
    eeg_64ch = torch.zeros(n_std, n_timepoints, dtype=eeg_data.dtype)
    channel_mask = torch.zeros(n_std, dtype=torch.bool)

    mapped_count = 0
    dropped: list[str] = []

    for src_idx, ch_name in enumerate(source_channels):
        # Try direct match first, then alias.
        resolved = ch_name if ch_name in ch_to_idx else _CHANNEL_ALIASES.get(ch_name)
        if resolved is not None and resolved in ch_to_idx:
            std_idx = ch_to_idx[resolved]
            eeg_64ch[std_idx] = eeg_data[src_idx]
            channel_mask[std_idx] = True
            mapped_count += 1
        else:
            dropped.append(ch_name)

    if dropped:
        logger.debug(
            "Dropped %d channel(s) not in standard layout: %s",
            len(dropped),
            dropped,
        )
    logger.debug(
        "Mapped %d / %d source channels to %d-ch layout.",
        mapped_count,
        len(source_channels),
        n_std,
    )

    return eeg_64ch, channel_mask
