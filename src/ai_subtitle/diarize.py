"""senko wrapper: CoreML speaker diarization (VAD + CAM++ embeddings + clustering).

On macOS the whole pipeline runs through CoreML (Apple Neural Engine / CPU) and
the models ship inside the senko wheel, so nothing is downloaded at runtime.
Input must be a 16 kHz mono 16-bit PCM WAV — see ``audio.extract_wav``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import senko

from .models import DiarizationResult, Turn


class Diarizer:
    """Lazily-initialised senko Diarizer (CoreML on macOS).

    ``merge_threshold`` maps to senko's ``mer_cos``: after clustering, clusters
    whose centroid cosine similarity is >= the threshold are merged. Lower it
    when too many speakers are detected, raise it when too few.
    """

    def __init__(
        self,
        *,
        merge_threshold: Optional[float] = None,
        accurate: Optional[bool] = None,
        device: str = "auto",
        quiet: bool = True,
    ) -> None:
        self._kwargs: dict = dict(device=device, quiet=quiet, warmup=True)
        if merge_threshold is not None:
            self._kwargs["mer_cos"] = merge_threshold
        self.accurate = accurate
        self._diarizer: Optional[senko.Diarizer] = None

    def _get(self) -> senko.Diarizer:
        if self._diarizer is None:
            self._diarizer = senko.Diarizer(**self._kwargs)
        return self._diarizer

    def diarize(self, wav_path: str | Path) -> DiarizationResult:
        data = self._get().diarize(
            str(wav_path), accurate=self.accurate, generate_colors=False
        )
        turns = [
            Turn(start=float(seg["start"]), end=float(seg["end"]), speaker=str(seg["speaker"]))
            for seg in (data.get("merged_segments") or [])
        ]
        centroids = {
            str(spk): [float(x) for x in vec]
            for spk, vec in (data.get("speaker_centroids") or {}).items()
        }
        vad = [(float(s), float(e)) for s, e in (data.get("vad") or [])]
        return DiarizationResult(turns=turns, centroids=centroids, vad=vad)
