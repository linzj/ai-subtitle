#!/usr/bin/env bash
# Generate a synthetic two-speaker dialogue WAV (16 kHz mono) using macOS `say`.
# The two speakers are chosen to sound clearly different (default: a male and a
# female Chinese voice) so diarization has something real to separate.
#
# Usage: scripts/make_test_audio.sh [output.wav]
# Env:   VOICE_A / VOICE_B to override the two `say` voices.
set -euo pipefail

OUT="${1:-/tmp/ai_subtitle_test.wav}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pick_voice() {
  local available line
  available="$(say -v '?' 2>/dev/null)"
  for candidate in "$@"; do
    if grep -qF -- "$candidate" <<<"$available"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  echo "error: none of the candidate voices found. Run 'say -v ?' to list voices." >&2
  return 1
}

VOICE_A="${VOICE_A:-$(pick_voice 'Eddy (中文（中国大陆）)' 'Reed (中文（中国大陆）)' 'Eddy (中文（台灣）)' 'Tingting' 'Sinji')}"
VOICE_B="${VOICE_B:-$(pick_voice 'Flo (中文（中国大陆）)' 'Sandy (中文（中国大陆）)' 'Grandma (中文（中国大陆）)' 'Meijia' 'Mei-Jia')}"
echo "voice A: $VOICE_A"
echo "voice B: $VOICE_B"

# Speaker A = odd lines, speaker B = even lines.
LINES=(
  "你好，欢迎收听今天的播客节目。"
  "大家好，很高兴又和大家见面了。"
  "今天我们聊聊端侧人工智能这个话题。"
  "我最近测试了 MLX 框架，速度非常快。"
  "而且完全不用联网，隐私也有保障。"
  "没错，Apple Silicon 的性能确实很强。"
)

: > "$WORK/list.txt"
ffmpeg -y -v error -f lavfi -i anullsrc=r=16000:cl=mono -t 0.7 -c:a pcm_s16le "$WORK/silence.wav"
echo "file '$WORK/silence.wav'" >> "$WORK/list.txt"

for i in "${!LINES[@]}"; do
  n=$((i + 1))
  if (( n % 2 == 1 )); then voice="$VOICE_A"; else voice="$VOICE_B"; fi
  say -v "$voice" -o "$WORK/line${n}.aiff" "${LINES[$i]}"
  ffmpeg -y -v error -i "$WORK/line${n}.aiff" -ac 1 -ar 16000 -c:a pcm_s16le "$WORK/line${n}.wav"
  echo "file '$WORK/line${n}.wav'" >> "$WORK/list.txt"
  echo "file '$WORK/silence.wav'" >> "$WORK/list.txt"
done

ffmpeg -y -v error -f concat -safe 0 -i "$WORK/list.txt" -c copy "$OUT"
echo "written: $OUT"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT"
