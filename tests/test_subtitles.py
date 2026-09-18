import json

from ai_subtitle.models import Cue, SpeakerInfo
from ai_subtitle.subtitles import (
    format_timestamp_srt,
    format_timestamp_vtt,
    render_json,
    render_srt,
    render_txt,
    render_vtt,
    write_outputs,
)

CUES = [
    Cue(start=0.0, end=2.5, speaker="SPEAKER_00", text="你好。"),
    Cue(start=2.5, end=5.0, speaker="SPEAKER_01", text="Hello there"),
]
NAMES = {"SPEAKER_00": "主持人", "SPEAKER_01": "嘉宾"}
SPEAKERS = [
    SpeakerInfo(id="SPEAKER_00", name="主持人", total_speech=2.5, n_segments=1, centroid=[0.1, 0.2]),
    SpeakerInfo(id="SPEAKER_01", name="嘉宾", total_speech=2.5, n_segments=1),
]


class TestTimestamps:
    def test_srt_zero(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_srt_hours(self):
        assert format_timestamp_srt(3661.5) == "01:01:01,500"

    def test_srt_rounds_millis(self):
        assert format_timestamp_srt(1.0006) == "00:00:01,001"

    def test_vtt_dot_separator(self):
        assert format_timestamp_vtt(3661.5) == "01:01:01.500"

    def test_negative_clamped(self):
        assert format_timestamp_srt(-1.0) == "00:00:00,000"


class TestRenderSrt:
    def test_structure_and_prefix(self):
        out = render_srt(CUES, names=NAMES)
        assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\n主持人: 你好。\n\n")
        assert "2\n00:00:02,500 --> 00:00:05,000\n嘉宾: Hello there\n" in out

    def test_empty_speaker_has_no_prefix(self):
        cues = [Cue(start=0.0, end=1.0, speaker="", text="无角色")]
        assert render_srt(cues).strip().endswith("无角色")

    def test_custom_format_template(self):
        out = render_srt(CUES[:1], names=NAMES, speaker_format="[{name}] {text}")
        assert "[主持人] 你好。" in out

    def test_no_cues(self):
        assert render_srt([]) == ""


class TestRenderVtt:
    def test_header_and_timestamps(self):
        out = render_vtt(CUES, names=NAMES)
        assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n主持人: 你好。")


class TestRenderTxt:
    def test_timestamped_lines(self):
        out = render_txt(CUES, names=NAMES)
        lines = out.strip().splitlines()
        assert lines[0] == "[00:00:00,000] 主持人: 你好。"
        assert lines[1] == "[00:00:02,500] 嘉宾: Hello there"


class TestRenderJson:
    def test_payload(self):
        data = json.loads(
            render_json(
                CUES,
                SPEAKERS,
                source="a.mp4",
                language="Chinese",
                duration=5.0,
                names=NAMES,
            )
        )
        assert data["source"] == "a.mp4"
        assert data["language"] == "Chinese"
        assert data["speakers"][0]["name"] == "主持人"
        assert data["speakers"][0]["centroid"] == [0.1, 0.2]
        assert data["speakers"][1]["centroid"] is None
        assert data["cues"][0]["speaker"] == "SPEAKER_00"
        assert data["cues"][0]["name"] == "主持人"
        assert data["cues"][0]["text"] == "你好。"

    def test_unnamed_speaker(self):
        cues = [Cue(start=0.0, end=1.0, speaker="SPEAKER_00", text="x")]
        data = json.loads(
            render_json(cues, [], source="a", language="English", duration=1.0)
        )
        assert data["speakers"] == []
        assert data["cues"][0]["name"] == "SPEAKER_00"


class TestWriteOutputs:
    def test_writes_all_formats(self, tmp_path):
        written = write_outputs(
            CUES,
            SPEAKERS,
            out_dir=tmp_path,
            stem="demo",
            formats=["srt", "json"],
            source="demo.mp4",
            language="Chinese",
            duration=5.0,
            names=NAMES,
        )
        assert [p.name for p in written] == ["demo.srt", "demo.json"]
        assert (tmp_path / "demo.srt").read_text(encoding="utf-8").startswith("1\n")
        assert json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))["cues"]

    def test_rejects_unknown_format(self, tmp_path):
        try:
            write_outputs(
                CUES, SPEAKERS, out_dir=tmp_path, stem="x", formats=["ass"],
                source="x", language="Chinese", duration=1.0,
            )
        except ValueError as exc:
            assert "ass" in str(exc)
        else:
            raise AssertionError("expected ValueError")
