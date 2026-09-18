"""ffmpeg helpers: probe media duration and extract 16 kHz mono PCM WAV.

senko strictly requires 16 kHz mono 16-bit PCM WAV input, and we feed the same
file to the ASR so both engines share an identical timeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    pass


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioError(
            f"{name} not found on PATH. Install it first, e.g. `brew install ffmpeg`."
        )
    return path


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds."""
    ffprobe = _require_tool("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise AudioError(f"could not read duration of {path}") from exc


def extract_wav(src: str | Path, dst: str | Path) -> Path:
    """Extract a 16 kHz mono 16-bit PCM WAV from any ffmpeg-readable file."""
    ffmpeg = _require_tool("ffmpeg")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-v", "error",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to extract audio from {src}: {proc.stderr.strip()}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioError(f"ffmpeg produced no audio for {src} (no audio stream?)")
    return dst
