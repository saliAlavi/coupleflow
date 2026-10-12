"""Canonical 64-channel layout and index mapping.

Every dataset is re-ordered to this standard layout so that all downstream
modules receive a consistent (64, T) tensor regardless of the source
recording system.
"""

from __future__ import annotations

STANDARD_64_CHANNELS: list[str] = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "Iz",
    "TP9", "TP10",
]
"""The canonical 64-channel order used throughout the pipeline."""

CHANNEL_TO_INDEX: dict[str, int] = {
    ch: idx for idx, ch in enumerate(STANDARD_64_CHANNELS)
}
"""Maps each canonical channel name to its 0-based position."""

NUM_STANDARD_CHANNELS: int = len(STANDARD_64_CHANNELS)
"""Number of channels in the canonical layout (64)."""
