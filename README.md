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
run offline. Interrupted model downloads are safely resumable. Copy
`config.example.toml` to `config.toml` only when overriding defaults.

Put source videos directly in `videos/`, or set `paths.videos` in ignored
`config.toml` to a repository-relative or absolute directory. The configured
directory must be writable because generated subtitle sidecars are stored beside
the immutable source videos. Then use the virtual-environment Python:

```bash
# Windows
.venv\Scripts\python -m subtitle_pipeline --json doctor
.venv\Scripts\python -m subtitle_pipeline --json inventory
.venv\Scripts\python -m subtitle_pipeline --json status
.venv\Scripts\python -m subtitle_pipeline sample --minutes 5

# macOS and Linux/WSL
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

`sample --minutes` requires a positive duration. If `--video` is supplied, it
must select a supported top-level video listed by `inventory`.

Sample and complete-video approvals are invalidated when configured backend,
device, model path, chunk duration, or explicit transcription/translation prompt
revisions change. Japanese outputs are reused only with matching video,
output/clip, model, backend, and transcription provenance. Chinese outputs also
require provenance for the exact Japanese source content and current translation
pipeline. Full validation rejects outputs with missing, malformed, or stale
provenance, including before bilingual generation.
Blank Whisper decode windows do not interrupt repetition-burst detection; exact
loops spanning those blanks are filtered before subtitle validation. Detection
also runs after final cue whitespace normalization so formatting differences
cannot conceal a loop.
If a five-minute faster-whisper chunk becomes empty after hallucination
filtering, the pipeline retries that same PCM audio in independent 30-second
decode windows. The combined result must still pass the normal quality gates;
empty outputs remain failures.

## Backends

- Apple Silicon defaults to MLX/Metal.
- Linux/WSL and Windows default to faster-whisper and llama.cpp, using CUDA when
  an NVIDIA driver compatible with CUDA 12.1 or newer is detected and CPU
  otherwise. Newer drivers use the packaged CUDA 12.5 wheel through NVIDIA
  backward compatibility. Windows CUDA installs pinned NVIDIA cuBLAS and CUDA
  runtimes and exposes their DLLs to faster-whisper and llama.cpp automatically.
- Override detection in ignored `config.toml` or pass an explicit bootstrap
  backend: `mac`, `linux-cuda`, `linux-cpu`, `windows-cuda`, or `windows-cpu`.
  If WSL or a container hides `nvidia-smi` from the bootstrap process, select
  the driver-compatible packaged wheel explicitly, for example
  `--backend linux-cuda --cuda-wheel cu125` for a CUDA 12.5-or-newer driver.

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

Configured video paths may be absolute and external. Runtime paths must remain
inside the repository, and model paths must remain inside the runtime model
directory. Rejected or stale subtitle outputs are preserved under
`.subtitle-tools/quarantine/` before replacement.
