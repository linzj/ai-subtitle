import pytest

from ai_subtitle.fuse import (
    build_cues,
    match_speaker_names,
    order_speakers,
    speaker_for_interval,
    speaker_infos,
)
from ai_subtitle.models import Turn, Word


def _words(specs):
    return [Word(start=s, end=e, text=t) for s, e, t in specs]


class TestSpeakerForInterval:
    def test_empty_turns(self):
        assert speaker_for_interval(0.0, 1.0, []) == ""

    def test_max_overlap_wins(self):
        turns = [Turn(0.0, 1.0, "A"), Turn(1.0, 4.0, "B")]
        assert speaker_for_interval(0.8, 2.0, turns) == "B"

    def test_no_overlap_falls_back_to_nearest(self):
        turns = [Turn(0.0, 1.0, "A"), Turn(5.0, 6.0, "B")]
        assert speaker_for_interval(4.4, 4.6, turns) == "B"
        assert speaker_for_interval(1.2, 1.4, turns) == "A"

    def test_exact_boundary_prefers_first_turn(self):
        turns = [Turn(0.0, 1.0, "A"), Turn(1.0, 2.0, "B")]
        assert speaker_for_interval(1.0, 1.0, turns) == "A"


class TestOrdering:
    def test_order_by_first_appearance(self):
        turns = [Turn(5.0, 6.0, "B"), Turn(0.0, 1.0, "A"), Turn(7.0, 8.0, "A")]
        assert order_speakers(turns) == ["A", "B"]

    def test_match_names_in_order(self):
        assert match_speaker_names(["A", "B"], ["主持人", "嘉宾"]) == {
            "A": "主持人",
            "B": "嘉宾",
        }

    def test_match_names_partial(self):
        assert match_speaker_names(["A", "B"], ["主持人"]) == {
            "A": "主持人",
            "B": "B",
        }


class TestBuildCues:
    def test_no_turns_has_empty_speaker(self):
        words = _words([(0.0, 0.5, "你"), (0.5, 1.0, "好")])
        cues = build_cues(words, transcript_text="你好", language="Chinese", turns=[])
        assert len(cues) == 1
        assert cues[0].text == "你好"
        assert cues[0].speaker == ""

    def test_no_words(self):
        assert build_cues([], transcript_text="", language="Chinese", turns=[]) == []

    def test_speaker_change_splits_cues(self):
        words = _words(
            [(0.0, 0.5, "你"), (0.5, 1.0, "好"), (2.0, 2.5, "嗨"), (2.5, 3.0, "呀")]
        )
        turns = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 4.0, "SPEAKER_01")]
        cues = build_cues(words, transcript_text="你好嗨呀", language="Chinese", turns=turns)
        assert [(c.speaker, c.text) for c in cues] == [
            ("SPEAKER_00", "你好"),
            ("SPEAKER_01", "嗨呀"),
        ]

    def test_english_words_joined_with_space(self):
        words = _words([(0.0, 0.4, "Hello"), (0.4, 0.8, "there")])
        cues = build_cues(
            words, transcript_text="Hello there", language="English", turns=[]
        )
        assert cues[0].text == "Hello there"

    def test_long_gap_splits_cues(self):
        words = _words([(0.0, 0.5, "嗯"), (3.0, 3.5, "啊")])
        cues = build_cues(
            words, transcript_text="嗯啊", language="Chinese", turns=[], max_gap=0.8
        )
        assert len(cues) == 2

    def test_max_duration_splits_cues(self):
        words = _words([(i * 0.5, i * 0.5 + 0.5, "字") for i in range(12)])
        cues = build_cues(
            words,
            transcript_text="字" * 12,
            language="Chinese",
            turns=[],
            max_duration=2.0,
        )
        assert len(cues) == 3
        for cue in cues:
            assert cue.end - cue.start <= 2.0 + 1e-6

    def test_sentence_boundary_splits_cues(self):
        words = _words([(0.0, 0.5, "你"), (0.5, 1.0, "好"), (1.0, 1.5, "世"), (1.5, 2.0, "界")])
        cues = build_cues(
            words, transcript_text="你好。世界", language="Chinese", turns=[]
        )
        assert [c.text for c in cues] == ["你好。", "世界"]

    def test_cue_timestamps_are_ordered(self):
        words = _words([(i * 0.5, i * 0.5 + 0.5, "字") for i in range(20)])
        turns = [
            Turn(0.0, 3.0, "A"),
            Turn(3.0, 6.0, "B"),
            Turn(6.0, 10.0, "A"),
        ]
        cues = build_cues(words, transcript_text="字" * 20, language="Chinese", turns=turns)
        for prev, cur in zip(cues, cues[1:]):
            assert cur.start >= prev.start
        assert {c.speaker for c in cues} == {"A", "B"}


class TestVadFilter:
    def test_words_in_silence_dropped(self):
        words = _words([(0.0, 0.5, "你"), (5.0, 5.5, "魔"), (6.0, 6.5, "好")])
        vad = [(0.0, 1.0), (5.7, 7.0)]
        cues = build_cues(
            words, transcript_text="你魔好", language="Chinese", turns=[], vad=vad
        )
        text = "".join(c.text for c in cues)
        assert "你" in text and "好" in text and "魔" not in text

    def test_vad_tolerance_keeps_edge_words(self):
        # Midpoints (0.2, 0.6) sit just outside the VAD span but within tolerance.
        words = _words([(0.0, 0.4, "开"), (0.4, 0.8, "始")])
        vad = [(0.29, 0.51)]
        cues = build_cues(
            words, transcript_text="开始", language="Chinese", turns=[], vad=vad
        )
        assert "".join(c.text for c in cues) == "开始"

    def test_none_vad_disables_filtering(self):
        words = _words([(5.0, 5.5, "嗯")])
        cues = build_cues(
            words, transcript_text="嗯", language="Chinese", turns=[], vad=None
        )
        assert len(cues) == 1

    def test_empty_vad_disables_filtering(self):
        words = _words([(5.0, 5.5, "嗯")])
        cues = build_cues(
            words, transcript_text="嗯", language="Chinese", turns=[], vad=[]
        )
        assert len(cues) == 1

    def test_everything_filtered_returns_empty(self):
        words = _words([(5.0, 5.5, "魔")])
        cues = build_cues(
            words, transcript_text="魔", language="Chinese", turns=[], vad=[(0.0, 1.0)]
        )
        assert cues == []


class TestMinDuration:
    def test_degenerate_cue_stretched(self):
        words = _words([(8.0, 8.0, "啊")])
        cues = build_cues(words, transcript_text="啊", language="Chinese", turns=[])
        assert cues[0].end - cues[0].start == pytest.approx(0.5)

    def test_short_cue_stretched(self):
        words = _words([(1.0, 1.1, "啊")])
        cues = build_cues(words, transcript_text="啊", language="Chinese", turns=[])
        assert cues[0].end - cues[0].start == pytest.approx(0.5)

    def test_stretch_never_overlaps_next_cue(self):
        words = _words([(8.0, 8.0, "啊"), (8.2, 8.2, "哦")])
        cues = build_cues(
            words, transcript_text="啊哦", language="Chinese", turns=[], max_gap=0.05
        )
        assert len(cues) == 2
        assert cues[0].end <= cues[1].start

    def test_normal_cue_untouched(self):
        words = _words([(0.0, 2.0, "字")])
        cues = build_cues(words, transcript_text="字", language="Chinese", turns=[])
        assert cues[0].start == 0.0 and cues[0].end == 2.0


class TestSpeakerInfos:
    def test_totals_and_counts(self):
        turns = [Turn(0.0, 2.0, "A"), Turn(2.0, 3.0, "B"), Turn(3.0, 5.0, "A")]
        words = _words([(0.0, 0.5, "嗯"), (2.0, 2.5, "啊")])
        cues = build_cues(words, transcript_text="嗯啊", language="Chinese", turns=turns)
        infos = speaker_infos(turns, cues, centroids={"A": [0.1, 0.2]}, names={"A": "主持"})
        by_id = {i.id: i for i in infos}
        assert by_id["A"].total_speech == 4.0
        assert by_id["B"].total_speech == 1.0
        assert by_id["A"].name == "主持"
        assert by_id["A"].centroid == [0.1, 0.2]
        assert by_id["B"].name == "B"
        assert sum(i.n_segments for i in infos) == len(cues)
