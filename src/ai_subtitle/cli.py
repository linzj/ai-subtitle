"""Command-line interface for ai-subtitle."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

# Must run before anything imports huggingface_hub: it freezes HF_ENDPOINT into
# its URL templates at import time. Documents/model downloads otherwise try
# huggingface.co directly, which is unreachable on blocked networks.
DEFAULT_HF_MIRROR = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_MIRROR)

from .asr import DEFAULT_ASR_MODEL, FAST_ASR_MODEL  # noqa: E402
from .audio import AudioError  # noqa: E402
from .pipeline import SubtitlePipeline, SubtitleResult  # noqa: E402
from .subtitles import ALL_FORMATS, DEFAULT_SPEAKER_FORMAT, write_outputs  # noqa: E402

STAGE_LABELS = {
    "extract": "提取 16kHz 音频",
    "diarize": "说话人分离 (CoreML)",
    "transcribe": "语音转写 (MLX)",
    "fuse": "对齐说话人与字幕",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-subtitle",
        description="端侧 AI 字幕：MLX 语音识别 + CoreML 说话人分离，输出分角色字幕。",
    )
    parser.add_argument("inputs", nargs="+", help="输入音视频文件（可多个）")
    parser.add_argument(
        "-o", "--output-dir",
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "-f", "--format", dest="formats", default="srt",
        help=f"输出格式，逗号分隔：{','.join(ALL_FORMATS)}，或 all（默认 srt）",
    )
    parser.add_argument(
        "--asr-model", default=DEFAULT_ASR_MODEL,
        help=f"Qwen3-ASR 模型（默认 {DEFAULT_ASR_MODEL}，可换 {FAST_ASR_MODEL} 提速）",
    )
    parser.add_argument(
        "--language", default=None,
        help="强制识别语言（如 Chinese / English），默认自动检测",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="跳过说话人分离，只做普通转写",
    )
    parser.add_argument(
        "--speaker-names", default=None,
        help="说话人姓名，逗号分隔，按首次出现顺序对应（如 主持人,嘉宾）",
    )
    parser.add_argument(
        "--speaker-format", default=DEFAULT_SPEAKER_FORMAT,
        help=f"字幕行模板（默认 {DEFAULT_SPEAKER_FORMAT!r}，可用 {{name}} 和 {{text}}）",
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=None,
        help="senko 聚类合并阈值 mer_cos（默认 0.875；相似度≥该值的聚类会被合并，"
             "分离不足/说话人偏少时调大，说话人偏多时调小）",
    )
    parser.add_argument(
        "--accurate", action="store_true",
        help="senko 高精度模式（更细粒度，稍慢）",
    )
    parser.add_argument(
        "--no-vad-filter", action="store_true",
        help="关闭静音区过滤（默认开启：丢弃无语音区间的识别结果，抑制音乐/音效上的幻觉）",
    )
    parser.add_argument("--max-chars", type=int, default=42, help="单条字幕最大显示宽度（CJK 记 2，默认 42）")
    parser.add_argument("--max-duration", type=float, default=6.0, help="单条字幕最长秒数（默认 6）")
    parser.add_argument("--max-gap", type=float, default=0.8, help="超过该静音秒数则断句（默认 0.8）")
    parser.add_argument(
        "--workdir", default=None,
        help="保留中间文件的目录（audio.wav / diarization.json）",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，只输出结果文件路径")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def _parse_formats(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_FORMATS)
    formats = [f.strip().lower() for f in raw.split(",") if f.strip()]
    if not formats:
        raise ValueError("--format 不能为空")
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(f"未知格式: {', '.join(unknown)}（可选 {', '.join(ALL_FORMATS)} 或 all）")
    return formats


def _parse_speaker_names(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [name.strip() for name in re.split(r"[,，、]", raw) if name.strip()]


class _ProgressPrinter:
    def __init__(self, enabled: bool, prefix: str) -> None:
        self.enabled = enabled
        self.prefix = prefix
        self._last_decile = -1

    def stage(self, stage: str) -> None:
        if self.enabled:
            print(f"{self.prefix}→ {STAGE_LABELS.get(stage, stage)}", file=sys.stderr, flush=True)

    def asr_event(self, event: dict) -> None:
        if not self.enabled:
            return
        if event.get("event") != "chunk_completed":
            return
        progress = event.get("progress")
        if progress is None:
            return
        decile = min(10, int(float(progress) * 10))
        if decile > self._last_decile:
            self._last_decile = decile
            print(f"{self.prefix}  转写进度 {decile * 10}%", file=sys.stderr, flush=True)


def _print_summary(result: SubtitleResult, paths: Sequence[Path]) -> None:
    for path in paths:
        print(path)
    print(f"  语言: {result.language} | 时长: {result.duration:.1f}s | 字幕: {len(result.cues)} 条", file=sys.stderr)
    if result.speakers:
        parts = [
            f"{info.name} ({info.total_speech:.1f}s, {info.n_segments} 段)"
            for info in result.speakers
        ]
        print(f"  说话人: {', '.join(parts)}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        formats = _parse_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))

    inputs = [Path(p) for p in args.inputs]
    missing = [p for p in inputs if not p.is_file()]
    if missing:
        for path in missing:
            print(f"错误: 文件不存在: {path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"HuggingFace 端点: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}"
            "（可设 HF_ENDPOINT 环境变量覆盖）",
            file=sys.stderr,
        )

    pipeline = SubtitlePipeline(
        asr_model=args.asr_model,
        language=args.language,
        diarize=not args.no_diarize,
        speaker_names=_parse_speaker_names(args.speaker_names),
        merge_threshold=args.merge_threshold,
        accurate=True if args.accurate else None,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        max_gap=args.max_gap,
        vad_filter=not args.no_vad_filter,
    )

    failed = False
    for index, input_path in enumerate(inputs, 1):
        header = f"[{index}/{len(inputs)}] {input_path.name}"
        if not args.quiet:
            print(header, file=sys.stderr, flush=True)
        printer = _ProgressPrinter(enabled=not args.quiet, prefix="  ")
        pipeline.on_stage = printer.stage
        pipeline.on_asr_progress = printer.asr_event

        out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        workdir = Path(args.workdir) if args.workdir else None
        try:
            result = pipeline.process(input_path, workdir=workdir)
            paths = write_outputs(
                result.cues,
                result.speakers,
                out_dir=out_dir,
                stem=input_path.stem,
                formats=formats,
                source=str(input_path),
                language=result.language,
                duration=result.duration,
                names=result.names,
                speaker_format=args.speaker_format,
            )
        except (AudioError, ValueError) as exc:
            print(f"  失败: {exc}", file=sys.stderr)
            failed = True
            continue

        if args.quiet:
            for path in paths:
                print(path)
        else:
            _print_summary(result, paths)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        raise SystemExit(130)
