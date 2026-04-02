"""Shared utility functions for hockey skating analysis."""

import os


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def format_timestamp(frame_idx: int, fps: float) -> str:
    """Convert frame index to MM:SS.f timestamp string.

    Args:
        frame_idx: Frame number (0-based).
        fps: Video frames per second.

    Returns:
        Formatted timestamp string like '01:23.4'.
    """
    total_sec = frame_idx / fps if fps > 0 else 0
    minutes = int(total_sec // 60)
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:05.2f}"
