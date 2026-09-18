# ai-subtitle

端侧 AI 字幕工具：**MLX 语音识别 + CoreML 说话人分离**，把任意音视频变成带说话人标注的字幕（分角色）。全程本地推理，不联网、不上传（仅首次下载模型需要网络）。

```
音视频文件
  │ ffmpeg → 16kHz 单声道 WAV
  ├─ 分角色：senko（CoreML，ANE/CPU）→ 说话人时间段 + VAD 语音区间
  ├─ 转写：Qwen3-ASR-1.7B（MLX，Metal GPU）→ 词级时间戳
  └─ 融合：VAD 过滤幻觉词 + 按时间重叠对齐 → SRT / VTT / TXT / JSON
```

**静音区过滤（默认开启）**：ASR 模型在音乐、音效段落会产生幻觉文字。管线用 senko 的 VAD 语音区间丢弃落在无语音区的识别结果（实测动漫片段片头音乐幻觉 38 条 → 19 条）。可用 `--no-vad-filter` 关闭。

| 环节 | 引擎 | 位置 |
| --- | --- | --- |
| 说话人分离 | [senko](https://github.com/narcotic-sh/senko)（pyannote segmentation-3.0 + CAM++ 声纹 + 聚类） | CoreML（模型随包附带，无需下载） |
| 语音识别 | [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr)（Qwen3-ASR，30 种语言 + 中文方言） | MLX / Metal GPU |
| 词级对齐 | Qwen3-ForcedAligner-0.6B | MLX |

## 安装

要求：Apple Silicon Mac、[uv](https://docs.astral.sh/uv/)、ffmpeg。

```bash
brew install uv ffmpeg
git clone <this-repo> && cd ai_subtitle
uv sync          # 自动装 Python 3.11（senko 仅提供 cp311 wheel）
```

### 模型下载与网络

首次运行会自动下载 `Qwen/Qwen3-ASR-1.7B`（约 3.4GB）和 `Qwen/Qwen3-ForcedAligner-0.6B` 到 `~/.cache/huggingface`。

CLI 在未设置 `HF_ENDPOINT` 时**默认使用镜像 `https://hf-mirror.com`**（huggingface.co 直连超时的网络下必需）。需要官方源时：

```bash
HF_ENDPOINT=https://huggingface.co uv run ai-subtitle ...   # 或在有代理的环境下直接运行
```

## 使用

```bash
# 基础：输出 SRT（与输入同目录）
uv run ai-subtitle 访谈.mp4 -o out/

# 常用组合
uv run ai-subtitle 访谈.mp4 -o out/ \
  --speaker-names "主持人,嘉宾"      # 按首次出现顺序给说话人命名
  --format srt,vtt,txt,json          # 多格式，或 --format all
  --asr-model Qwen/Qwen3-ASR-0.6B    # 更快的识别模型（精度略降）

# 只转写不分角色
uv run ai-subtitle 讲座.m4a --no-diarize

# 分离过度/不足时调参（默认 0.875，相似度 ≥ 该值的声纹聚类会被合并）
uv run ai-subtitle 会议.wav --merge-threshold 0.90   # 调大 → 更难合并 → 说话人更多
uv run ai-subtitle 会议.wav --merge-threshold 0.85   # 调小 → 更易合并 → 说话人更少
```

常用参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--speaker-names` | 说话人姓名，逗号分隔（中英文逗号均可） | `SPEAKER_01` 等 |
| `--speaker-format` | 字幕行模板，支持 `{name}` `{text}` | `{name}: {text}` |
| `--format` | `srt,vtt,txt,json` 或 `all` | `srt` |
| `--language` | 强制语言（`Chinese`/`English`…） | 自动检测 |
| `--merge-threshold` | senko 聚类合并阈值 `mer_cos` | 0.875 |
| `--accurate` | senko 更细粒度（略慢） | 关 |
| `--no-vad-filter` | 关闭静音区过滤（默认开启，见下） | 关（即过滤开启） |
| `--max-chars` / `--max-duration` / `--max-gap` | 字幕断句：宽度（CJK 记 2）/ 最长秒数 / 静音断句阈值 | 42 / 6.0 / 0.8 |
| `--workdir` | 保留中间文件（`audio.wav`、`diarization.json`） | 临时目录 |

输出示例（SRT）：

```
1
00:00:00,700 --> 00:00:03,700
Eddy: 你好，欢迎收听今天的播客节目。

2
00:00:04,020 --> 00:00:07,560
Flo: 大家好，很高兴又和大家见面了。
```

`json` 输出包含完整结构化数据：每条字幕（时间/文字/说话人）、说话人统计与 **192 维声纹 centroid**（可配合 `senko.speaker_similarity` 做跨文件身份比对）。

## 作为 Python 库

```python
from ai_subtitle import SubtitlePipeline, write_outputs

pipeline = SubtitlePipeline(speaker_names=["主持人", "嘉宾"])
result = pipeline.process("访谈.mp4")          # 可反复调用，模型常驻
print(result.language, [(s.name, round(s.total_speech, 1)) for s in result.speakers])
for cue in result.cues:
    print(f"[{cue.start:6.1f}] {result.names.get(cue.speaker, cue.speaker)}: {cue.text}")

write_outputs(result.cues, result.speakers, out_dir="out", stem="访谈",
              formats=["srt", "json"], source="访谈.mp4",
              language=result.language, duration=result.duration, names=result.names)
```

库使用时如需镜像，请**在 import 之前**设置环境变量：`os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`（huggingface_hub 在导入时固化 endpoint）。

## 已知限制

- senko 同一时刻只标一个说话人（重叠抢话时取主导者），不支持重叠标注
- 不能指定说话人数量，只能通过 `--merge-threshold` 间接调整
- 低信噪比、背景音乐、同人不同麦克风会降低分离质量
- 唱歌、集体喊叫等被 VAD 判为语音的段落，仍可能出现识别幻觉（VAD 过滤无法覆盖）
- 首次运行需下载模型，之后完全离线
- 暂无跨文件声纹识别（`json` 输出已保留 centroid，便于后续实现）

## 开发

```bash
uv run pytest                          # 纯逻辑单测（不加载模型）
scripts/make_test_audio.sh /tmp/t.wav  # 用 macOS say 生成双说话人测试音频
uv run ai-subtitle /tmp/t.wav -o /tmp/out --format all
```
