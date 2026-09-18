"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Cue:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class SpeakerInfo:
    id: str
    name: str
    total_speech: float
    n_segments: int
    centroid: list[float] | None = None


@dataclass
class DiarizationResult:
    turns: list[Turn]
    centroids: dict[str, list[float]] = field(default_factory=dict)
    vad: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TranscriptResult:
    text: str
    language: str
    words: list[Word]
