"""Video I/O utilities for hockey skating analysis.

Provides memory-efficient frame generation and video writing.
Adapted from CS6476 PS4 read_video() but uses a generator pattern
to avoid loading entire videos into RAM.
"""

import cv2
import numpy as np
from typing import Iterator, Optional
from contextlib import contextmanager


def get_video_metadata(video_path: str) -> dict:
    """Get video metadata without reading frames.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps, frame_count, width, height, duration_sec.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    metadata["duration_sec"] = (
        metadata["frame_count"] / metadata["fps"] if metadata["fps"] > 0 else 0
    )
    cap.release()
    return metadata


def frame_generator(video_path: str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) from a video file.

    Memory-efficient: only one frame in memory at a time.

    Args:
        video_path: Path to the video file.

    Yields:
        Tuple of (frame_index, bgr_frame as numpy array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


@contextmanager
def video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
):
    """Context manager for writing video frames.

    Usage:
        with video_writer("out.mp4", 30.0, 1920, 1080) as writer:
            writer.write(frame)

    Args:
        output_path: Output video file path.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string (default: 'mp4v').
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    try:
        yield writer
    finally:
        writer.release()
