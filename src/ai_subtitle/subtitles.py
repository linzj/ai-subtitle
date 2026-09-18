"""Render subtitle files: SRT, VTT, TXT and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .models import Cue, SpeakerInfo

DEFAULT_SPEAKER_FORMAT = "{name}: {text}"
ALL_FORMATS = ("srt", "vtt", "txt", "json")


def format_timestamp_srt(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def cue_text(cue: Cue, names: Mapping[str, str], speaker_format: str) -> str:
    if not cue.speaker:
        return cue.text
    name = names.get(cue.speaker, cue.speaker)
    return speaker_format.format(name=name, text=cue.text)


def render_srt(
    cues: Sequence[Cue],
    *,
    names: Mapping[str, str] | None = None,
    speaker_format: str = DEFAULT_SPEAKER_FORMAT,
) -> str:
    names = names or {}
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(
            f"{i}\n"
            f"{format_timestamp_srt(cue.start)} --> {format_timestamp_srt(cue.end)}\n"
            f"{cue_text(cue, names, speaker_format)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(
    cues: Sequence[Cue],
    *,
    names: Mapping[str, str] | None = None,
    speaker_format: str = DEFAULT_SPEAKER_FORMAT,
) -> str:
    names = names or {}
    blocks = ["WEBVTT"]
    for cue in cues:
        blocks.append(
            f"{format_timestamp_vtt(cue.start)} --> {format_timestamp_vtt(cue.end)}\n"
            f"{cue_text(cue, names, speaker_format)}"
        )
    return "\n\n".join(blocks) + "\n"


def render_txt(
    cues: Sequence[Cue],
    *,
    names: Mapping[str, str] | None = None,
    speaker_format: str = DEFAULT_SPEAKER_FORMAT,
) -> str:
    names = names or {}
    lines = [
        f"[{format_timestamp_srt(cue.start)}] {cue_text(cue, names, speaker_format)}"
        for cue in cues
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_json(
    cues: Sequence[Cue],
    speakers: Sequence[SpeakerInfo],
    *,
    source: str,
    language: str,
    duration: float,
    names: Mapping[str, str] | None = None,
) -> str:
    names = names or {}
    payload = {
        "source": source,
        "language": language,
        "duration": round(float(duration), 3),
        "speakers": [
            {
                "id": info.id,
                "name": info.name,
                "total_speech": round(info.total_speech, 3),
                "n_segments": info.n_segments,
                "centroid": [round(float(x), 6) for x in info.centroid]
                if info.centroid is not None
                else None,
            }
            for info in speakers
        ],
        "cues": [
            {
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "speaker": cue.speaker or None,
                "name": names.get(cue.speaker, cue.speaker) if cue.speaker else None,
                "text": cue.text,
            }
            for cue in cues
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_outputs(
    cues: Sequence[Cue],
    speakers: Sequence[SpeakerInfo],
    *,
    out_dir: str | Path,
    stem: str,
    formats: Sequence[str],
    source: str,
    language: str,
    duration: float,
    names: Mapping[str, str] | None = None,
    speaker_format: str = DEFAULT_SPEAKER_FORMAT,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "srt": lambda: render_srt(cues, names=names, speaker_format=speaker_format),
        "vtt": lambda: render_vtt(cues, names=names, speaker_format=speaker_format),
        "txt": lambda: render_txt(cues, names=names, speaker_format=speaker_format),
        "json": lambda: render_json(
            cues, speakers, source=source, language=language, duration=duration, names=names
        ),
    }
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt not in renderers:
            raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(ALL_FORMATS)}")
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(renderers[fmt](), encoding="utf-8")
        written.append(path)
    return written
