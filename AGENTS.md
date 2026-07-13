# AGENTS.md — Offline subtitle pipeline

## Mission and safety

Maintain a reusable local Japanese-to-Simplified-Chinese subtitle pipeline.
Original videos are immutable. Never delete or rename videos, models, quarantine
artifacts, or completed subtitles without explicit user approval.

Treat this file as maintained repository documentation. Update it in the same
change whenever a durable workflow, command, layout, or constraint changes.

## Documentation authority and synchronization

- `AGENTS.md` is the source of truth for repository mission, safety rules,
  processing invariants, change discipline, and completion requirements. Future
  agents must read and follow the current version before taking action.
- Keep `AGENTS.md`, `README.md`, `AI_OPERATIONS.md`, `config.example.toml`, CLI
  help/behavior, and relevant tests synchronized. When implementation or
  workflow changes make any of them inaccurate, update every affected document
  and test in the same change.
- Do not leave durable decisions only in chat, commit messages, pull request
  descriptions, or runtime logs. Record them in `AGENTS.md` and the appropriate
  user/operator documentation.
- Before reporting or publishing a change, compare the documentation with the
  implemented commands, defaults, paths, gates, and safety behavior. Treat a
  mismatch as incomplete work.

## Repository and private data

- Tracked source lives in `src/subtitle_pipeline`; setup utilities live in
  `scripts`; tests live in `tests`.
- The configured video directory contains private inputs and generated sidecars;
  it may be repository-relative or an absolute external path. Only
  `videos/.gitkeep` may be tracked from the default directory.
- `.subtitle-tools/`, `.venv/`, `config.toml`, models, caches, logs, reports,
  state, checkpoints, samples, and quarantine are local runtime data.
- Before commits, inspect `git status --ignored` and `git ls-files`; never stage
  private inputs, generated outputs, runtime state, or unexpectedly large files.

## Canonical AI-terminal workflow

Use platform-neutral Python commands and prefer `--json` for inspection:

```text
python -m subtitle_pipeline --json doctor
python -m subtitle_pipeline --json inventory
python -m subtitle_pipeline --json status
python -m subtitle_pipeline sample --minutes 5
python -m subtitle_pipeline approve-sample --note "human review summary"
python -m subtitle_pipeline process
python -m subtitle_pipeline approve-video-gate --note "human review summary"
python -m subtitle_pipeline process
python -m subtitle_pipeline validate
python -m subtitle_pipeline bilingual
```

Sample duration must be positive, and an explicit sample input must be a
supported top-level video reported by inventory.

Bootstrap a fresh clone with `python scripts/bootstrap.py --backend auto
--non-interactive`. Bootstrap is the only network-required phase; processing
after dependencies and models are installed is offline.
Interrupted model downloads must remain resumable and must not be treated as
complete until their pinned snapshot or model file is fully materialized.
On Linux/WSL and Windows, auto-detection selects CUDA for an NVIDIA driver
reporting CUDA 12.1 or newer; newer drivers use the project's CUDA 12.5 wheels.
The Windows CUDA bootstrap also installs the pinned NVIDIA cuBLAS and CUDA
runtimes and adds their DLL directories before loading faster-whisper or
llama.cpp.
Hardware-isolated WSL/container terminals may pass `--backend linux-cuda
--cuda-wheel cu125` when host `nvidia-smi` confirms a CUDA 12.5-or-newer driver.

## Processing invariants

- Process one video at a time: transcribe, validate Japanese, translate, validate
  Chinese, checkpoint title, then advance.
- Decode long media sequentially into temporary five-minute mono PCM chunks.
  Validate and checkpoint each chunk, then delete temporary audio.
- Use temperature fallback and `condition_on_previous_text=False` for MLX
  Whisper. Use faster-whisper with VAD on Linux/WSL and Windows.
- Reject empty, malformed, stale, misaligned, untranslated, internally
  repetitive, cross-cue repetitive, or implausibly long outputs.
- Ignore empty Whisper decode windows when detecting consecutive repetition
  bursts so blank segments cannot split and conceal a hallucination loop.
- Repeat burst detection must also run after final cue whitespace normalization,
  before SRT rendering and checkpoint validation.
- If filtering leaves a five-minute faster-whisper chunk with no valid cues,
  retry that same PCM chunk as independent 30-second decode windows, combine
  and validate the recovered cues, and still reject a genuinely empty result.
- Reuse Japanese outputs only when their provenance fingerprint matches the
  current video, output/clip, backend, model, and transcription revision. Reuse
  Chinese outputs only when their fingerprint matches the exact Japanese
  source, backend, model, and translation/prompt revisions.
- Write outputs atomically via `.partial`. Preserve all existing artifacts under
  `.subtitle-tools/quarantine/`; quarantine suspect outputs before manual
  replacement.
- A bilingual cue places Simplified Chinese above Japanese and is generated only
  from exactly aligned monolingual SRTs.
- Bind sample and first-video approvals to hashes of the reviewed artifacts;
  invalidate approval whenever those artifacts change. Require current full
  validation immediately before bilingual generation. Full validation must
  reject missing, malformed, or stale transcription and translation provenance.
- Keep configured runtime paths inside the repository and model paths inside the
  runtime model directory. Video paths may be external; treat every source video
  as immutable and write only generated sidecars beside it.

## Change and execution discipline

- Read `README.md`, `AI_OPERATIONS.md`, configuration, entrypoints, and relevant
  tests before editing or launching work.
- Confirm no runner is active. The atomic runner lock must reject duplicates;
  status must report absent, active, stale, or unreadable lock ownership.
  Unreadable ownership must fail closed rather than be automatically removed.
- After edits run `py_compile`, Ruff, pytest, CLI JSON smoke tests, and Git ignore
  audits with `python scripts/audit_repo.py`. CI must never download models or
  process private media.
- On a new backend, model, prompt, or transcription revision, require a reviewed
  five-minute sample and then one complete-video gate before advancing.
- Automated checks establish structural integrity, not human-level linguistic
  quality. Never claim subtitles are perfect or human-verified without review.
- Do not automatically remove models as recovery. Report disk pressure and stop
  safely between checkpoints.

## Completion reporting

Only report completion after validation succeeds. Include video count/duration,
Japanese and Chinese success/failure counts, unresolved warnings, report paths,
elapsed time, disk use, title/bilingual status, confirmation originals were not
modified, and whether retained models permit immediate offline reuse.
