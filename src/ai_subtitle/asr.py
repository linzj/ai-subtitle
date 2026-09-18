"""Qwen3-ASR wrapper (pure MLX, on-device) with word-level timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from mlx_qwen3_asr import Session

from .models import TranscriptResult, Word

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
FAST_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

ProgressCallback = Callable[[dict], None]


class Transcriber:
    """Lazily-initialised Qwen3-ASR session.

    Timestamps are always requested: the forced aligner emits word-level
    segments, which the fusion stage needs for speaker assignment.
    """

    def __init__(self, model: str = DEFAULT_ASR_MODEL) -> None:
        self.model = model
        self._session: Optional[Session] = None

    def _get_session(self) -> Session:
        if self._session is None:
            self._session = Session(model=self.model)
        return self._session

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TranscriptResult:
        result = self._get_session().transcribe(
            str(audio_path),
            language=language,
            return_timestamps=True,
            on_progress=on_progress,
        )
        words = [
            Word(start=float(item["start"]), end=float(item["end"]), text=str(item["text"]))
            for item in (result.segments or [])
        ]
        return TranscriptResult(
            text=result.text,
            language=result.language,
            words=words,
        )
