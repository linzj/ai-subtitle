"""End-to-end pipeline: media file -> speaker-labeled subtitle cues."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .asr import DEFAULT_ASR_MODEL, ProgressCallback, Transcriber
from .audio import extract_wav, probe_duration
from .diarize import Diarizer
from .fuse import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_GAP,
    build_cues,
    match_speaker_names,
    order_speakers,
    speaker_infos,
)
from .models import Cue, SpeakerInfo, Turn

StageCallback = Callable[[str], None]


@dataclass
class SubtitleResult:
    cues: list[Cue]
    speakers: list[SpeakerInfo]
    names: dict[str, str]
    language: str
    duration: float
    transcript: str
    turns: list[Turn] = field(default_factory=list)


class SubtitlePipeline:
    """Reusable pipeline; models stay loaded across calls."""

    def __init__(
        self,
        *,
        asr_model: str = DEFAULT_ASR_MODEL,
        language: Optional[str] = None,
        diarize: bool = True,
        speaker_names: Optional[Sequence[str]] = None,
        merge_threshold: Optional[float] = None,
        accurate: Optional[bool] = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_duration: float = DEFAULT_MAX_DURATION,
        max_gap: float = DEFAULT_MAX_GAP,
        vad_filter: bool = True,
        on_stage: Optional[StageCallback] = None,
        on_asr_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.asr_model = asr_model
        self.language = language
        self.diarize = diarize
        self.speaker_names = list(speaker_names or [])
        self.merge_threshold = merge_threshold
        self.accurate = accurate
        self.max_chars = max_chars
        self.max_duration = max_duration
        self.max_gap = max_gap
        self.vad_filter = vad_filter
        self.on_stage = on_stage or (lambda stage: None)
        self.on_asr_progress = on_asr_progress
        self._transcriber: Optional[Transcriber] = None
        self._diarizer: Optional[Diarizer] = None

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(self.asr_model)
        return self._transcriber

    def _get_diarizer(self) -> Diarizer:
        if self._diarizer is None:
            self._diarizer = Diarizer(
                merge_threshold=self.merge_threshold, accurate=self.accurate
            )
        return self._diarizer

    def process(self, input_path: str | Path, *, workdir: Optional[str | Path] = None) -> SubtitleResult:
        if workdir is not None:
            return self._process(Path(input_path), Path(workdir))
        with tempfile.TemporaryDirectory(prefix="ai-subtitle-") as tmp:
            return self._process(Path(input_path), Path(tmp))

    def _process(self, input_path: Path, workdir: Path) -> SubtitleResult:
        workdir.mkdir(parents=True, exist_ok=True)

        self.on_stage("extract")
        duration = probe_duration(input_path)
        wav_path = extract_wav(input_path, workdir / "audio.wav")

        turns: list[Turn] = []
        centroids: dict[str, list[float]] = {}
        vad: Optional[list[tuple[float, float]]] = None
        if self.diarize:
            self.on_stage("diarize")
            diarization = self._get_diarizer().diarize(wav_path)
            turns = diarization.turns
            centroids = diarization.centroids
            if self.vad_filter:
                vad = diarization.vad
            (workdir / "diarization.json").write_text(
                json.dumps(
                    [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        self.on_stage("transcribe")
        transcript = self._get_transcriber().transcribe(
            wav_path, language=self.language, on_progress=self.on_asr_progress
        )

        self.on_stage("fuse")
        cues = build_cues(
            transcript.words,
            transcript_text=transcript.text,
            language=transcript.language,
            turns=turns,
            vad=vad,
            max_chars=self.max_chars,
            max_duration=self.max_duration,
            max_gap=self.max_gap,
        )
        names = match_speaker_names(order_speakers(turns), self.speaker_names)
        infos = speaker_infos(turns, cues, centroids=centroids, names=names)
        return SubtitleResult(
            cues=cues,
            speakers=infos,
            names=names,
            language=transcript.language,
            duration=duration,
            transcript=transcript.text,
            turns=turns,
        )


def transcribe_file(
    input_path: str | Path,
    *,
    workdir: Optional[str | Path] = None,
    **pipeline_kwargs,
) -> SubtitleResult:
    """One-shot convenience wrapper around :class:`SubtitlePipeline`."""
    return SubtitlePipeline(**pipeline_kwargs).process(input_path, workdir=workdir)
