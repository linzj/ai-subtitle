#!/usr/bin/env bash
# 实时麦克风转写（Ctrl+C 结束并保存转录稿 + 录音）
# 用法: ./start.sh [麦克风设备名]   设备列表: uv run python -c "import sounddevice; print(sounddevice.query_devices())"
# 首次运行需在系统弹窗中授予终端麦克风权限（系统设置 → 隐私与安全性 → 麦克风）
# 输出保存在 recordings/ 目录（txt 转录稿 + wav 录音，wav 可再用 ai-subtitle 做分角色精修）
set -euo pipefail
cd "$(dirname "$0")"

run() {
  uv run python -m ai_subtitle.live "$@"
}

if [[ $# -ge 1 ]]; then
  run --device "$1"
else
  run
fi
