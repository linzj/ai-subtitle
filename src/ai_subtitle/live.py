"""Live microphone transcription with speaker labels (分角色).

Capture goes through ffmpeg/avfoundation (the native macOS stack also used by
the system Voice Memos app). PortAudio/sounddevice was measured to deliver
noise-like garbage from the Bluetooth mic on this machine (zero-crossing rate
~0.4, spectral centroid ~4 kHz for actual speech), which the ASR then
"transcribed" as garbage; ffmpeg's capture of the same device is clean.

Decoding is not streaming: the upstream streaming decoder re-decodes
overlapping context per 2 s chunk and cannot keep up with real time on the
1.7B model. The *offline* batch path is ~10x faster than real time on the same
model, so this runner cuts the pending audio at silence roughly every
``--window-sec`` seconds and transcribes each segment offline.

Speaker labels come from a rolling window (last ``ROLLING_SEC`` seconds) that
is re-diarized with senko (CoreML, ~0.1 s per window) whenever a segment is
produced. Window-local clusters are matched to a cross-segment registry by
192-dim voiceprint centroid similarity, so labels stay stable across segments.

Usage: ``uv run python -m ai_subtitle.live`` or via ``./start.sh``.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Must be set before huggingface_hub is imported (it freezes the endpoint).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TARGET_RATE = 16000
CALLBACK_BLOCK_SEC = 0.5
BYTES_PER_SAMPLE = 2
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
SILENT_SEGMENT_RMS = 0.003
ROLLING_SEC = 60.0
SPEAKER_MATCH_THRESHOLD = 0.875
CONTEXT_TAIL_CHARS = 120


def _list_avfoundation_devices() -> list[tuple[int, str]]:
    """[(index, name)] of avfoundation audio input devices."""
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if in_audio:
            m = re.search(r"\[(\d+)\]\s+(.*?)\s*$", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


def _resolve_device(spec: Optional[str]) -> tuple[int, str]:
    devices = _list_avfoundation_devices()
    if not devices:
        raise RuntimeError("未找到可用的音频输入设备（ffmpeg avfoundation）")
    names = ", ".join(f"[{i}] {name}" for i, name in devices)
    if spec is not None and spec.strip().isdigit():
        index = int(spec)
        for i, name in devices:
            if i == index:
                return index, name
        raise RuntimeError(f"设备序号 {index} 不存在。可选: {names}")
    if spec is not None:
        for i, name in devices:
            if name == spec:
                return i, name
        for i, name in devices:
            if spec.lower() in name.lower():
                return i, name
        raise RuntimeError(f"未找到设备 {spec!r}。可选: {names}")
    # Default: match the system default input device by name.
    try:
        import sounddevice as sd

        default_name = str(sd.query_devices(sd.default.device[0])["name"])
        for i, name in devices:
            if name == default_name or name in default_name or default_name in name:
                return i, name
    except Exception:
        pass
    return devices[0]


def _find_cut(samples: np.ndarray, min_cut: float, max_cut: float) -> Optional[float]:
    """Return a cut point in seconds at the end of a silence run inside
    [min_cut, max_cut], or None when no run of >= 250 ms silence exists."""
    frame = int(0.025 * TARGET_RATE)
    hop = int(0.010 * TARGET_RATE)
    if len(samples) < frame + hop:
        return None
    n_frames = (len(samples) - frame) // hop + 1
    idx = np.arange(n_frames)[:, None] * hop + np.arange(frame)[None, :]
    rms = np.sqrt(np.mean(samples[idx].astype(np.float32) ** 2, axis=1) + 1e-12)
    # Silence is relative to the buffer's median level: speech dominates the
    # window, so quieter-than-a-quarter-of-median frames mark pauses.
    silent = rms < max(float(np.median(rms)) * 0.25, 0.002)

    lo = int(min_cut * TARGET_RATE / hop)
    hi = min(int(max_cut * TARGET_RATE / hop), n_frames)
    best_run = 0
    best_end = -1
    run = 0
    for i in range(max(lo, 0), hi):
        if silent[i]:
            run += 1
        else:
            if run > best_run:
                best_run, best_end = run, i
            run = 0
    if run > best_run:
        best_run, best_end = run, hi
    if best_run * hop / TARGET_RATE >= 0.25:
        return best_end * hop / TARGET_RATE
    return None


class _SpeakerRegistry:
    """Maps per-window senko clusters to stable 说话人N labels via voiceprints.

    A single observation below the match threshold must NOT create a speaker:
    the same voice drifts across windows (volume, distance from the mic,
    calling out loud), and one-off mis-matches otherwise produce ghost
    speakers. Rules:

    - score >= MATCH: confident match; the stored voiceprint is updated (EMA)
      so it tracks the speaker's drift.
    - gray zone: the voiceprint is held as *pending* and the segment is
      stickily attributed to the nearest known speaker; a second observation
      matching the pending print promotes it to a new 说话人N.
    - below FLOOR: too different to attribute; segment stays unlabeled.
    """

    def __init__(
        self,
        match_threshold: float = SPEAKER_MATCH_THRESHOLD,
        confirm_threshold: float = 0.85,
        floor: float = 0.6,
        max_speakers: Optional[int] = None,
    ) -> None:
        import senko

        self._similarity = senko.speaker_similarity
        self._match = match_threshold
        self._confirm = confirm_threshold
        self._floor = floor
        self._max = max_speakers
        self._speakers: list[list] = []  # [[label, centroid], ...]
        self._pending: Optional[np.ndarray] = None

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self._speakers]

    def assign(self, centroid: np.ndarray) -> str:
        centroid = np.asarray(centroid, dtype=np.float32)
        if not self._speakers:
            self._speakers.append(["说话人1", centroid])
            return "说话人1"

        best_label = ""
        best_score = -1.0
        best_ref = None
        for entry in self._speakers:
            score = float(self._similarity(centroid, entry[1]))
            if score > best_score:
                best_label, best_score, best_ref = entry[0], score, entry

        if best_score >= self._match:
            assert best_ref is not None
            best_ref[1] = 0.7 * best_ref[1] + 0.3 * centroid  # track voice drift
            return best_label

        # With a known speaker count, never exceed it: drifted same-voice
        # observations stay stickily attributed instead of spawning ghosts.
        at_cap = self._max is not None and len(self._speakers) >= self._max
        if at_cap:
            return best_label

        if self._pending is not None and float(self._similarity(centroid, self._pending)) >= self._confirm:
            label = f"说话人{len(self._speakers) + 1}"
            self._speakers.append([label, 0.5 * self._pending + 0.5 * centroid])
            self._pending = None
            return label

        self._pending = centroid
        return best_label if best_score >= self._floor else ""


def _run(args: argparse.Namespace) -> int:
    from mlx_qwen3_asr import Session

    index, dev_name = _resolve_device(args.device)
    print(f"设备: {dev_name} (avfoundation #{index})", file=sys.stderr)
    print("加载模型（期间不占用麦克风）...", file=sys.stderr, flush=True)
    t0 = time.time()
    session = Session(model=args.model)
    diarizer = None
    if not args.no_diarize:
        import senko

        diarizer = senko.Diarizer(device="auto", quiet=True, warmup=True)
        print(
            f"✅ 就绪（语音识别 + 分角色，耗时 {time.time() - t0:.0f}s）| "
            "现在开始说话，Ctrl+C 结束",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"✅ 就绪（耗时 {time.time() - t0:.0f}s）| 现在开始说话，Ctrl+C 结束",
            file=sys.stderr,
            flush=True,
        )
    registry = (
        _SpeakerRegistry(max_speakers=args.speakers) if diarizer is not None else None
    )

    language = None if args.language == "auto" else args.language
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    txt_path = out_dir / f"recording-{stamp}.txt"
    wav_path = out_dir / f"recording-{stamp}.wav"
    wav_file = wave.open(str(wav_path), "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(TARGET_RATE)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    dropped = 0
    stop_flag = threading.Event()

    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "avfoundation", "-i", f":{index}",
            "-ac", "1", "-ar", str(TARGET_RATE),
            "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    block_bytes = int(CALLBACK_BLOCK_SEC * TARGET_RATE) * BYTES_PER_SAMPLE

    def reader() -> None:  # ffmpeg stdout thread
        nonlocal dropped
        assert proc.stdout is not None
        # Buffered read: blocks until exactly block_bytes or EOF. Unbuffered
        # pipes can return short reads, which are not end-of-stream.
        while not stop_flag.is_set():
            raw = proc.stdout.read(block_bytes)
            if not raw or len(raw) < block_bytes:
                break
            mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                audio_q.put_nowait(mono)
            except queue.Full:
                dropped += 1

    time.sleep(0.3)
    if proc.poll() is not None:
        err = (proc.stderr.read() or b"").decode(errors="replace").strip()
        print(f"错误: ffmpeg 采集启动失败: {err}", file=sys.stderr)
        wav_file.close()
        return 1
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    recent: list[np.ndarray] = []
    recent_sec = 0.0
    pending: list[np.ndarray] = []
    pending_sec = 0.0
    stream_sec = 0.0
    decode_sec = 0.0
    segments = 0
    lines: list[str] = []
    detected_language = args.language if language else "auto"
    use_context = not args.no_context
    context_tail = ""  # tail of previous transcript, fed back as decode context

    def assign_speaker(start_sec: float, end_sec: float) -> str:
        if diarizer is None or registry is None or not recent:
            return ""
        window = np.concatenate(recent)
        window_sec = len(window) / TARGET_RATE
        if window_sec < 1.0:
            return ""
        data = diarizer.diarize_samples(
            window, sample_rate=TARGET_RATE, source_name="live"
        )
        if not data:  # no speech found in window (e.g. silence)
            return ""
        centroids = data.get("speaker_centroids") or {}
        window_start = stream_sec - window_sec
        best_speaker = ""
        best_overlap = 0.0
        for turn in data.get("merged_segments") or []:
            turn_start = window_start + float(turn["start"])
            turn_end = window_start + float(turn["end"])
            overlap = max(0.0, min(end_sec, turn_end) - max(start_sec, turn_start))
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, str(turn["speaker"])
        centroid = centroids.get(best_speaker)
        if centroid is None or best_overlap <= 0.05:
            return ""
        return registry.assign(np.asarray(centroid, dtype=np.float32))

    def transcribe_segment(samples: np.ndarray, start_sec: float, end_sec: float) -> None:
        nonlocal decode_sec, detected_language, segments, context_tail
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2) + 1e-12))
        if rms < SILENT_SEGMENT_RMS:
            return
        t1 = time.time()
        result = session.transcribe(
            (samples, TARGET_RATE),
            language=language,
            context=context_tail if use_context else "",
        )
        decode_sec += time.time() - t1
        segments += 1
        text = str(result.text or "").strip()
        if result.language:
            detected_language = result.language
        if not text:
            return
        if use_context:
            context_tail = (context_tail + text)[-CONTEXT_TAIL_CHARS:]
        speaker = assign_speaker(start_sec, end_sec)
        mm, ss = divmod(int(start_sec), 60)
        time_label = f"[{mm:02d}:{ss:02d}]"
        line = f"{time_label} {speaker}: {text}" if speaker else f"{time_label} {text}"
        lines.append(line)
        print(line, flush=True)

    started = time.time()
    try:
        try:
            while True:
                if args.duration_sec is not None and time.time() - started >= args.duration_sec:
                    break
                try:
                    block = audio_q.get(timeout=0.2)
                except queue.Empty:
                    if proc.poll() is not None:
                        print("采集进程意外退出", file=sys.stderr)
                        break
                    continue
                wav_file.writeframes(
                    (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                )
                recent.append(block)
                recent_sec += len(block) / TARGET_RATE
                while recent_sec > ROLLING_SEC and len(recent) > 1:
                    recent_sec -= len(recent.pop(0)) / TARGET_RATE
                pending.append(block)
                pending_sec += len(block) / TARGET_RATE
                stream_sec += len(block) / TARGET_RATE
                if pending_sec < args.window_sec:
                    continue
                seg = np.concatenate(pending)
                seg_start = stream_sec - len(seg) / TARGET_RATE
                cut = _find_cut(seg, min_cut=args.window_sec * 0.6, max_cut=args.max_window_sec)
                if cut is None:
                    if pending_sec < args.max_window_sec:
                        continue
                    cut = args.max_window_sec
                idx = int(cut * TARGET_RATE)
                piece = seg[:idx]
                transcribe_segment(piece, seg_start, seg_start + len(piece) / TARGET_RATE)
                rest = seg[idx:]
                pending = [rest] if len(rest) else []
                pending_sec = len(rest) / TARGET_RATE
        except KeyboardInterrupt:
            print("\n停止采集...", file=sys.stderr)
    finally:
        stop_flag.set()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        thread.join(timeout=2)
        wav_file.close()
        if pending_sec >= 0.3:
            print("处理剩余音频...", file=sys.stderr, flush=True)
            tail = np.concatenate(pending)
            tail_start = stream_sec - len(tail) / TARGET_RATE
            transcribe_segment(tail, tail_start, stream_sec)

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print()
    if dropped:
        print(f"⚠ 采集丢弃了 {dropped} 块音频", file=sys.stderr)
    speaker_note = ""
    if registry is not None and registry.labels:
        speaker_note = f" | 说话人: {len(registry.labels)} 个（{', '.join(registry.labels)}）"
    print(
        f"时长 {stream_sec:.1f}s | {segments} 段 | 解码共 {decode_sec:.1f}s"
        f"（{decode_sec / max(stream_sec, 0.1):.2f}x 实时）{speaker_note}",
        file=sys.stderr,
    )
    print(f"转录稿: {txt_path} | 录音: {wav_path}", file=sys.stderr)
    print(
        f"提示: 可在会后用离线管线对录音精修（更准的分角色与断句）: "
        f"uv run ai-subtitle {wav_path} -o {out_dir}/",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-subtitle-live", description="实时麦克风转写 + 分角色（Ctrl+C 结束）"
    )
    parser.add_argument(
        "--device", default=None,
        help="输入设备名或序号（ffmpeg avfoundation，默认跟随系统默认输入设备）",
    )
    parser.add_argument("--language", default="Chinese", help="识别语言，默认 Chinese；auto 为自动检测")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Qwen3-ASR 模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument(
        "--window-sec", type=float, default=6.0,
        help="攒够多少秒音频就送识别（默认 6，即文字延迟约几秒）",
    )
    parser.add_argument(
        "--max-window-sec", type=float, default=15.0,
        help="连续说话无静音时的强制切段上限（默认 15 秒）",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="关闭实时分角色（默认开启，说话人标签与文字同延迟）",
    )
    parser.add_argument(
        "--no-context", action="store_true",
        help="关闭跨段文本上下文接力（默认开启：把前文尾部作为提示词传给下一段解码，改善专名/术语一致性）",
    )
    parser.add_argument(
        "--speakers", type=int, default=None,
        help="已知说话人数量（独白=1、双人对谈=2；不填则自动判断。声纹漂移大的场景建议填写）",
    )
    parser.add_argument("--duration-sec", type=float, default=None, help="录制时长（默认直到 Ctrl+C）")
    parser.add_argument("-o", "--output-dir", default="recordings", help="转录稿与录音的保存目录")
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
