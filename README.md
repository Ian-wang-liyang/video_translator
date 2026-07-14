# Offline Japanese-to-Chinese subtitle pipeline

[![CI](https://github.com/Ian-wang-liyang/video_translator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ian-wang-liyang/video_translator/actions/workflows/ci.yml)

A local, resumable pipeline that transcribes Japanese video and produces
Japanese, Simplified Chinese, and optional stacked bilingual SRT subtitles.
Source video is never rewritten, and inference runs offline after the one-time
bootstrap downloads pinned models and dependencies.

> [!IMPORTANT]
> This is a quality-gated automation tool, not a substitute for a fluent human
> reviewer. Sample and first-video approvals are deliberately required before a
> collection can run unattended.

## Why this project exists

- **Private by default:** videos, subtitles, models, logs, and runtime state are
  excluded from Git.
- **Safe to resume:** work is checkpointed per five-minute audio chunk and final
  files are written atomically.
- **Quality gated:** transcription and translation outputs are checked for
  malformed cues, stale provenance, repetition, misalignment, untranslated
  text, and other common model failures.
- **Human controlled:** a reviewed sample and one reviewed full video are bound
  to artifact hashes. Changing the artifacts or pipeline fingerprint invalidates
  approval.
- **Cross-platform:** Apple Silicon uses MLX; Windows and Linux/WSL use
  faster-whisper and llama.cpp with CUDA when supported, otherwise CPU.
- **Machine friendly:** inspection commands expose stable JSON and a
  `next_action` field for terminal automation.

## Requirements

- Python 3.12 (the supported range is `>=3.12,<3.13`)
- FFmpeg and FFprobe on `PATH`
- Enough free disk space for the pinned speech and translation models
- A writable video directory, because generated SRT sidecars are placed beside
  each immutable source video

GPU acceleration is optional. NVIDIA auto-detection requires a driver reporting
CUDA 12.1 or newer. See [Backends](#backends) for platform details.

## Quick start

From a fresh clone:

```bash
python scripts/bootstrap.py --backend auto --non-interactive
```

Bootstrap creates `.venv`, installs the selected backend, and downloads pinned
models into `.subtitle-tools/models`. Interrupted downloads are resumable. This
is the only phase that requires network access.

Put supported videos directly in `videos/`, or copy `config.example.toml` to the
ignored `config.toml` and set `paths.videos` to a repository-relative or absolute
directory. Videos must be top-level files in that directory.

Use the virtual-environment Python for all remaining commands:

```powershell
# Windows PowerShell
.venv\Scripts\python -m subtitle_pipeline --json doctor
.venv\Scripts\python -m subtitle_pipeline --json inventory
.venv\Scripts\python -m subtitle_pipeline --json status
.venv\Scripts\python -m subtitle_pipeline sample --minutes 5
```

```bash
# macOS, Linux, and WSL
.venv/bin/python -m subtitle_pipeline --json doctor
.venv/bin/python -m subtitle_pipeline --json inventory
.venv/bin/python -m subtitle_pipeline --json status
.venv/bin/python -m subtitle_pipeline sample --minutes 5
```

Inspect both files in `.subtitle-tools/sample`. Only after a human linguistic
review, approve the exact artifacts and proceed:

```text
python -m subtitle_pipeline approve-sample --note "Japanese and Chinese sample reviewed"
python -m subtitle_pipeline process
python -m subtitle_pipeline approve-video-gate --note "First complete video reviewed"
python -m subtitle_pipeline process
python -m subtitle_pipeline validate
python -m subtitle_pipeline bilingual
```

The first `process` run stops after one complete video. Review its Japanese and
Chinese SRT files before approving the full-video gate. The second run processes
the remaining collection. `bilingual` requires a current successful full
validation.

## Workflow at a glance

```text
bootstrap -> doctor -> inventory -> sample -> human review
    -> approve sample -> process one video -> human review
    -> approve video -> process collection -> validate -> bilingual
```

At any point, run:

```text
python -m subtitle_pipeline --json status
```

Follow `next_action`; do not infer activity from a PID file alone. See
[AI_OPERATIONS.md](AI_OPERATIONS.md) for the complete automation contract and
failure-handling rules.

## Command reference

| Command | Purpose |
| --- | --- |
| `doctor` | Check FFmpeg, model files, directories, and backend modules. |
| `inventory` | Count supported source videos and existing monolingual outputs. |
| `status` | Report lock/gate state and the safe `next_action`. |
| `effective-config` | Show resolved paths, backends, models, and processing values. |
| `sample` | Generate a review sample; defaults to five minutes. |
| `approve-sample` | Record explicit human approval of the current sample hashes. |
| `process` | Resume safe one-video-at-a-time collection processing. |
| `approve-video-gate` | Approve the current first full-video artifacts. |
| `validate` | Validate every expected monolingual output and its provenance. |
| `bilingual` | Stack aligned Chinese-over-Japanese cues after validation. |
| `clean --dry-run` | List incomplete `.partial` files without deleting them. |

`--json` is a global option and must appear before the subcommand. Commands use
exit codes `0` (success), `10` (validation), `20` (configuration), `21` (missing
dependency/model), `30` (runner conflict), `40` (inference), and `130`
(interrupted).

`sample --minutes` must be positive. An explicit `--video` must match a
supported top-level inventory entry. The escape hatch
`process --video PATH --skip-sample-gate` is only for an explicitly authorized
quality migration; it processes exactly one inventory video and still stops at
the full-video review gate.

## Outputs and private runtime data

For `videos/example.mkv`, the pipeline may create:

```text
videos/example.ja.srt
videos/example.zh-Hans.srt
videos/example.ja-zh-Hans.srt
```

The bilingual file places Simplified Chinese above Japanese. Video titles are
never translated or renamed. Full validation writes
`.subtitle-tools/reports/validation.tsv`; decode warnings and audio-stream
inventory are stored under `.subtitle-tools/reports/decode-warnings/`.

All models, checkpoints, logs, reports, approvals, caches, samples, quarantine,
and provenance records live under ignored `.subtitle-tools/`. Suspect generated
outputs are preserved in `.subtitle-tools/quarantine/` before replacement.
Original videos are never filtered, rewritten, renamed, or deleted.

## Configuration

Defaults are documented in [config.example.toml](config.example.toml). Copy it
to ignored `config.toml` only when overriding them. Confirm resolved values with:

```text
python -m subtitle_pipeline --json effective-config
```

Runtime paths must stay inside the repository. Model paths are relative to the
runtime model directory. The video directory may be absolute and external.
Processing thresholds are quality-sensitive: changing backend, model, prompt,
chunking, or related fingerprinted settings requires a new sample and
full-video review.

## Backends

- **Apple Silicon:** MLX/Metal with the pinned MLX Whisper and Qwen models.
- **Windows:** faster-whisper and llama.cpp; CUDA is selected when compatible,
  otherwise CPU. CUDA bootstrap installs pinned NVIDIA cuBLAS/runtime packages
  and exposes their DLL directories to the inference libraries.
- **Linux/WSL:** faster-whisper and llama.cpp with CUDA or CPU. A
  hardware-isolated environment may use `--backend linux-cuda --cuda-wheel
  cu125` when host `nvidia-smi` confirms a CUDA 12.5-or-newer driver.

The Windows/Linux quality-first defaults use Whisper `large-v3`, the
CPU/float32 `kotoba-whisper-v2.0-faster` Japanese specialist, and Qwen3 8B
Q4_K_M. The default 4096-token translation context targets 6 GB-class GPUs;
lower `translation_gpu_layers` if full offload does not fit.

## Reliability model

Long media is decoded sequentially to temporary five-minute mono PCM chunks.
Both adjacent source chunks and faster-whisper's 30-second decode windows overlap
while non-overlapping ownership regions emit each moment once. A PCM foreground
gate can retain quieter speech when acoustic and ASR confidence are both strong.
Loud uncovered intervals and low-confidence cues are retried without no-speech
rejection and checked against the isolated Japanese specialist. De-clipping,
when needed, applies only to temporary rescue audio. Slightly sub-threshold
specialist agreement is accepted only when the audio is clearly loud and both
models are strongly confident.

Validated checkpoints and outputs are reused only when their provenance matches
the exact video, clip/output, model, backend, source subtitle, and pipeline
revisions. On Windows, locked sample and processing runs request system-sleep
inhibition for their lifetime without changing the display or persistent power
plan.

## Limitations

- Linguistic quality still requires a fluent human reviewer.
- The pipeline targets Japanese speech and Simplified Chinese output; it is not a
  general-purpose subtitle translator.
- Bootstrap needs network access and substantial disk space; normal processing
  is offline once dependencies and models are present.
- Performance and memory use vary significantly by media, backend, and hardware.
- Decode warnings or an unintended audio stream can still require manual
  diagnosis. A warning immediately after an input seek does not by itself prove
  corruption; reports preserve the evidence without modifying the source.

## Development and project policy

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the repository and
[SECURITY.md](SECURITY.md) before reporting a vulnerability. Maintainer and AI
execution rules live in [AGENTS.md](AGENTS.md); change history lives in
[CHANGELOG.md](CHANGELOG.md). CI runs linting, tests, CLI smoke checks, and a
private-data audit on Windows, macOS, and Linux without downloading models or
processing private media.

No license is currently granted for redistribution or reuse. Before publishing
or accepting external distribution, the repository owner should select and add
an explicit license.
