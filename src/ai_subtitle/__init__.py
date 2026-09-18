"""On-device AI subtitle generator with speaker diarization (MLX + CoreML).

Submodules are loaded lazily (PEP 562) so that ``import ai_subtitle`` does not
pull in huggingface_hub: its endpoint is frozen at import time, so the CLI must
be able to set ``HF_ENDPOINT`` before any model code is imported.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_EXPORTS = {
    "DEFAULT_ASR_MODEL": ".asr",
    "FAST_ASR_MODEL": ".asr",
    "Transcriber": ".asr",
    "AudioError": ".audio",
    "extract_wav": ".audio",
    "probe_duration": ".audio",
    "Diarizer": ".diarize",
    "build_cues": ".fuse",
    "match_speaker_names": ".fuse",
    "order_speakers": ".fuse",
    "speaker_for_interval": ".fuse",
    "speaker_infos": ".fuse",
    "Cue": ".models",
    "DiarizationResult": ".models",
    "SpeakerInfo": ".models",
    "TranscriptResult": ".models",
    "Turn": ".models",
    "Word": ".models",
    "SubtitlePipeline": ".pipeline",
    "SubtitleResult": ".pipeline",
    "transcribe_file": ".pipeline",
    "render_json": ".subtitles",
    "render_srt": ".subtitles",
    "render_txt": ".subtitles",
    "render_vtt": ".subtitles",
    "write_outputs": ".subtitles",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from .asr import DEFAULT_ASR_MODEL, FAST_ASR_MODEL, Transcriber
    from .audio import AudioError, extract_wav, probe_duration
    from .diarize import Diarizer
    from .fuse import (
        build_cues,
        match_speaker_names,
        order_speakers,
        speaker_for_interval,
        speaker_infos,
    )
    from .models import (
        Cue,
        DiarizationResult,
        SpeakerInfo,
        TranscriptResult,
        Turn,
        Word,
    )
    from .pipeline import SubtitlePipeline, SubtitleResult, transcribe_file
    from .subtitles import (
        render_json,
        render_srt,
        render_txt,
        render_vtt,
        write_outputs,
    )
