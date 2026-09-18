"""Fuse ASR words with diarization turns into speaker-labeled subtitle cues.

Word-to-speaker assignment and cue grouping reuse mlx-qwen3-asr's subtitle
machinery (max-overlap assignment, punctuation restoration, CJK-aware cue
splitting) so the output matches its subtitle formatting exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mlx_qwen3_asr.diarization import diarize_word_segments
from mlx_qwen3_asr.writers import group_subtitle_segments

from .models import Cue, SpeakerInfo, Turn, Word

DEFAULT_MAX_CHARS = 42
DEFAULT_MAX_DURATION = 6.0
DEFAULT_MAX_GAP = 0.8
MIN_CUE_DURATION = 0.5
VAD_TOLERANCE = 0.3


def turns_to_dicts(turns: Sequence[Turn]) -> list[dict]:
    return [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]


def _in_speech(mid: float, vad: Sequence[tuple[float, float]]) -> bool:
    for start, end in vad:
        if start - VAD_TOLERANCE <= mid <= end + VAD_TOLERANCE:
            return True
    return False


def _clamp_min_duration(grouped: list[dict]) -> None:
    """Stretch degenerate/too-short cues up to MIN_CUE_DURATION, capped by the
    next cue so they never overlap."""
    for i, cue in enumerate(grouped):
        start = float(cue["start"])
        end = float(cue["end"])
        next_start = float(grouped[i + 1]["start"]) if i + 1 < len(grouped) else float("inf")
        target = min(start + MIN_CUE_DURATION, next_start)
        if end < target:
            cue["end"] = target


def speaker_for_interval(start: float, end: float, turns: Sequence[Turn]) -> str:
    """Return the speaker with maximum temporal overlap; nearest turn as fallback."""
    if not turns:
        return ""
    end = max(end, start)
    best_speaker = turns[0].speaker
    best_overlap = -1.0
    for t in turns:
        overlap = max(0.0, min(end, t.end) - max(start, t.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = t.speaker
    if best_overlap > 0.0:
        return best_speaker

    mid = 0.5 * (start + end)
    best_dist = float("inf")
    for t in turns:
        dist = abs(mid - 0.5 * (t.start + t.end))
        if dist < best_dist:
            best_dist = dist
            best_speaker = t.speaker
    return best_speaker


def build_cues(
    words: Sequence[Word],
    *,
    transcript_text: str,
    language: str,
    turns: Sequence[Turn],
    vad: Optional[Sequence[tuple[float, float]]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_gap: float = DEFAULT_MAX_GAP,
) -> list[Cue]:
    """Build speaker-labeled subtitle cues from word timestamps and diarization.

    When ``vad`` speech regions are given, words whose midpoint falls outside
    every region are dropped: on music/sound-effect passages the ASR model
    hallucinates text that would otherwise leak into the subtitles. An empty
    ``vad`` list is treated as "no information" and disables filtering.
    """
    if not words:
        return []

    if vad:
        words = [w for w in words if _in_speech(0.5 * (w.start + w.end), vad)]
        if not words:
            return []

    word_dicts: list[dict] = [
        {"text": w.text, "start": w.start, "end": w.end} for w in words
    ]
    has_speakers = bool(turns)
    if has_speakers:
        word_dicts = diarize_word_segments(
            word_dicts, speaker_turns=turns_to_dicts(turns)
        )

    grouped = group_subtitle_segments(
        word_dicts,
        language=language,
        text=transcript_text,
        max_chars=max_chars,
        max_duration_sec=max_duration,
        max_gap_sec=max_gap,
    )
    _clamp_min_duration(grouped)
    return [
        Cue(
            start=float(g["start"]),
            end=float(g["end"]),
            speaker=speaker_for_interval(float(g["start"]), float(g["end"]), turns)
            if has_speakers
            else "",
            text=str(g["text"]),
        )
        for g in grouped
    ]


def order_speakers(turns: Sequence[Turn]) -> list[str]:
    """Speaker ids in order of first appearance."""
    seen: dict[str, None] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        seen.setdefault(turn.speaker, None)
    return list(seen)


def speaker_infos(
    turns: Sequence[Turn],
    cues: Sequence[Cue],
    centroids: Optional[dict[str, list[float]]] = None,
    names: Optional[dict[str, str]] = None,
) -> list[SpeakerInfo]:
    """Per-speaker statistics (speaking time from turns, cue count from cues)."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + max(0.0, turn.end - turn.start)
    counts: dict[str, int] = {}
    for cue in cues:
        if cue.speaker:
            counts[cue.speaker] = counts.get(cue.speaker, 0) + 1
    centroids = centroids or {}
    names = names or {}
    return [
        SpeakerInfo(
            id=speaker,
            name=names.get(speaker, speaker),
            total_speech=totals.get(speaker, 0.0),
            n_segments=counts.get(speaker, 0),
            centroid=centroids.get(speaker),
        )
        for speaker in order_speakers(turns)
    ]


def match_speaker_names(speakers: Sequence[str], names: Sequence[str]) -> dict[str, str]:
    """Map raw speaker ids to names in order of first appearance."""
    return {
        speaker: names[i] if i < len(names) else speaker
        for i, speaker in enumerate(speakers)
    }
