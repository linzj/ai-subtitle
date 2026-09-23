#!/usr/bin/env bash
# 实时麦克风转写 + 分角色（Ctrl+C 结束并保存转录稿 + 录音）
# 用法: ./start.sh [麦克风设备名] [额外参数...]
#   例: ./start.sh --speakers 1          # 独白场景（已知只有一人，避免声纹漂移被拆成多个说话人）
#       ./start.sh --speakers 2          # 双人对谈
#       ./start.sh "STAX SPIRIT S3"      # 指定麦克风设备
# 首次运行需在系统弹窗中授予终端麦克风权限（系统设置 → 隐私与安全性 → 麦克风）
# 输出保存在 recordings/ 目录（txt 转录稿 + wav 录音，wav 可再用 ai-subtitle 做分角色精修）
set -euo pipefail
cd "$(dirname "$0")"

run() {
  uv run python -m ai_subtitle.live "$@"
}

# 第一个参数不是选项时按麦克风设备名处理，其余参数原样透传
if [[ $# -ge 1 && "$1" != -* ]]; then
  run --device "$@"
else
  run "$@"
fi
