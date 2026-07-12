# Offline Japanese-to-Chinese subtitles

Local, resumable subtitle generation for Japanese videos. The pipeline creates
Japanese, Simplified Chinese, and optional stacked bilingual SRT files while
keeping videos, models, outputs, and runtime state out of Git.

## Fresh clone

Requirements: Python 3.12, FFmpeg, FFprobe, and enough disk space for models.

```bash
python scripts/bootstrap.py --backend auto --non-interactive
```

The bootstrap creates `.venv`, installs the platform backend, and downloads
models into ignored `.subtitle-tools/models`. After that, normal processing can
run offline. Copy `config.example.toml` to `config.toml` only when overriding
defaults.

Put source videos directly in `videos/`, then use the virtual-environment Python:

```bash
# Windows
.venv\Scripts\python -m subtitle_pipeline --json doctor
.venv\Scripts\python -m subtitle_pipeline --json inventory
.venv\Scripts\python -m subtitle_pipeline --json status
.venv\Scripts\python -m subtitle_pipeline sample --minutes 5

# macOS
.venv/bin/python -m subtitle_pipeline --json doctor
.venv/bin/python -m subtitle_pipeline --json inventory
.venv/bin/python -m subtitle_pipeline --json status
.venv/bin/python -m subtitle_pipeline sample --minutes 5
```

Inspect the sample files under `.subtitle-tools/sample`. Approve that exact
pipeline fingerprint only after a human spot check:

```bash
python -m subtitle_pipeline approve-sample --note "Japanese and Chinese sample reviewed"
python -m subtitle_pipeline process
python -m subtitle_pipeline approve-video-gate --note "First complete video reviewed"
python -m subtitle_pipeline process
python -m subtitle_pipeline validate
python -m subtitle_pipeline bilingual
```

Run these commands through `.venv` as shown above. The pipeline always finishes
one video—Japanese transcription, Chinese translation, and title checkpoint—
before moving to the next. The first `process` invocation stops after one video;
after that output is reviewed and approved, the second invocation processes the
remaining collection.

Sample and complete-video approvals are invalidated when configured backend,
device, model path, chunk duration, or explicit transcription/translation prompt
revisions change. Japanese outputs are reused only with matching video,
output/clip, model, backend, and transcription provenance. Chinese outputs also
require provenance for the exact Japanese source content and current translation
pipeline.

## Backends

- Apple Silicon defaults to MLX/Metal.
- Windows defaults to faster-whisper and llama.cpp, using CUDA when a supported
  NVIDIA/CUDA 12.1–12.5 environment is detected and CPU otherwise.
- Override detection in ignored `config.toml` or pass an explicit bootstrap
  backend: `mac`, `windows-cuda`, or `windows-cpu`.

Use `python -m subtitle_pipeline --json status` for stable machine-readable
progress and a `next_action`. See [AI_OPERATIONS.md](AI_OPERATIONS.md) for the
generic AI-terminal operating contract.

## Outputs

For `videos/example.mkv`:

- `example.ja.srt`
- `example.zh-Hans.srt`
- `example.ja-zh-Hans.srt` — Simplified Chinese above Japanese

These sidecars are written beside the source under `videos/`. Suggested Chinese
titles are checkpointed in `.subtitle-tools/reports/title-mapping.csv`, and
`validate` writes `.subtitle-tools/reports/validation.tsv`.

Original videos are never rewritten. Generated files and all runtime data are
ignored by Git.
