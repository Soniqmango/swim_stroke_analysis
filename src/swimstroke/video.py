"""Video input/output.

Reading: yields frames together with their *real* presentation timestamps.
Phone cameras record variable frame rate (VFR) video, so the time of frame i
is not exactly i / fps. Every timing metric (splits, turn time, stroke rate)
depends on these timestamps, so we take them from the container instead of
assuming a constant rate.

Writing: pipes frames into the ffmpeg binary bundled with imageio-ffmpeg to
produce H.264 MP4, which (unlike OpenCV's default 'mp4v') plays in browsers,
Streamlit and GitHub.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float  # nominal (average) frame rate reported by the container
    n_frames: int  # container estimate; can be off by a few frames for VFR video

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass
class Frame:
    idx: int
    t: float  # seconds from start of video
    image: np.ndarray  # BGR, H x W x 3, uint8


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {path}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    """Read video metadata without decoding frames."""
    cap = _open(path)
    try:
        return VideoInfo(
            path=Path(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield frames in order with timestamps in seconds.

    Timestamps come from CAP_PROP_POS_MSEC (the container's presentation
    timestamp for the frame just decoded). If the backend returns timestamps
    that are unusable (not strictly increasing), we fall back to idx / fps
    and warn once, so bad timing never passes silently.

    Note: OpenCV applies the phone's rotation metadata automatically, so
    portrait/landscape clips come out the right way up.
    """
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_t = -1.0
    use_fallback = False
    idx = 0
    try:
        while max_frames is None or idx < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not use_fallback and t <= prev_t:
                warnings.warn(
                    f"Non-increasing timestamp at frame {idx} ({t:.4f}s <= {prev_t:.4f}s); "
                    f"falling back to idx/fps ({fps:.2f}) for the rest of the video.",
                    stacklevel=2,
                )
                use_fallback = True
            if use_fallback:
                t = idx / fps
            prev_t = t
            yield Frame(idx=idx, t=t, image=image)
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Write BGR frames to a browser-playable H.264 MP4.

    Usage:
        with VideoWriter("out.mp4", fps=60, size=(w, h)) as vw:
            vw.write(frame)

    The output uses a constant frame rate. That is fine because the annotated
    video is for viewing only; all measurements use the real timestamps.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size  # (width, height)
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",  # OpenCV's native channel order, so no conversion
            pix_fmt_out="yuv420p",  # most compatible H.264 pixel format
            macro_block_size=2,  # yuv420p needs even dimensions; avoids resizing
            ffmpeg_log_level="error",
        )
        self._gen.send(None)  # start the ffmpeg process

    def write(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"Frame size {(w, h)} != writer size {self.size}")
        self._gen.send(np.ascontiguousarray(image))

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
