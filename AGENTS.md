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
- Public project documentation lives at the repository root: `README.md` is the
  user entry point, `AI_OPERATIONS.md` is the automation runbook,
  `CONTRIBUTING.md` defines change hygiene, `SECURITY.md` defines private
  reporting, and `CHANGELOG.md` records user-visible changes.
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
python -m subtitle_pipeline --json effective-config
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
  Chinese, then advance. Do not translate or rename video titles.
- On Windows, every locked sample or processing run must hold a system-sleep
  inhibition request for its full lifetime and release it on every exit path.
  The display may still turn off; do not change persistent OS power-plan settings.
- Decode long media sequentially into temporary five-minute mono PCM chunks.
  Include a short source-audio overlap on both sides and assign each result to
  the non-overlapping nominal chunk by segment midpoint. Validate and checkpoint
  each chunk, then delete temporary audio.
- Use temperature fallback and `condition_on_previous_text=False` for MLX
  Whisper. On Linux/WSL and Windows, faster-whisper uses VAD only when no
  explicit clip is supplied; explicit overlapping windows use PCM foreground
  gating because faster-whisper bypasses VAD for clip timestamps.
- Overlap adjacent short decode windows and assign a non-overlapping ownership
  region to each result so speech crossing a window boundary is decoded twice
  but emitted once.
- Build a loud-audio coverage map after the primary decode. Retry uncovered
  high-energy intervals and low-confidence cues without Whisper's no-speech
  rejection. Accept recovery only after confidence/text checks and agreement
  with the CPU/float32 Japanese specialist, except for independently strong
  output. A narrowly sub-threshold agreement may pass only when both models are
  strongly confident and the source interval is clearly loud.
  Run all specialist intervals for a chunk in one isolated worker process; the
  Windows CTranslate2 runtimes for the GPU primary and CPU specialist must not
  coexist in one process. Do not request specialist word timestamps: the pinned
  official conversion access-violates on Windows when that feature is enabled.
- Apply FFmpeg de-clipping only to temporary rescue PCM when clipped samples are
  detected. Record source decode warnings and available audio streams
  under runtime reports; never filter or rewrite the source video.
- Reject empty, malformed, stale, misaligned, untranslated, internally
  repetitive, cross-cue repetitive, or implausibly long outputs.
- Permit an individual atomic chunk checkpoint to be empty only when all primary
  and recovery candidates were rejected by the foreground, confidence,
  agreement, or hallucination gates. The assembled full-video transcript must
  still contain valid cues.
- Ignore empty Whisper decode windows when detecting consecutive repetition
  bursts so blank segments cannot split and conceal a hallucination loop.
- Repeat burst detection must also run after final cue whitespace normalization,
  before SRT rendering and checkpoint validation.
- Run faster-whisper over each five-minute PCM chunk as independent 30-second
  decode windows. Use a PCM dBFS foreground gate to omit low-level/background
  speech, but retain moderately quieter segments when log probability and
  no-speech probability both show strong speech confidence. Then combine and
  validate the retained cues.
- Reuse Japanese outputs only when their provenance fingerprint matches the
  current video, output/clip, backend, model, and transcription revision. Reuse
  Chinese outputs only when their fingerprint matches the exact Japanese
  source, backend, model, and translation/prompt revisions.
- Translate local cue batches as continuous dialogue with configurable
  surrounding context and accepted earlier Japanese-to-Chinese pairs so names,
  ellipsis, sentence fragments, and recurring terms remain consistent across
  batch boundaries. Translation retries must change both their corrective
  instruction and sampling attempt instead of repeating an identical request.
  Run the configured second semantic pass only on cues flagged by script,
  length, fallback, unchanged-source, or context-sensitive-fragment signals;
  preserve one-to-one cue alignment and reject malformed review responses.
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
- Record user-visible behavior or workflow changes under `Unreleased` in
  `CHANGELOG.md`. Do not publish a release, add a license, or change the granted
  reuse rights without explicit owner approval.
- On a new backend, model, prompt, or transcription revision, require a reviewed
  five-minute sample and then one complete-video gate before advancing.
- Only when the user explicitly authorizes skipping that sample, process one
  explicit inventory video with `process --video PATH --skip-sample-gate` and
  stop at the complete-video human review gate. Never use the flag implicitly.
- Automated checks establish structural integrity, not human-level linguistic
  quality. Never claim subtitles are perfect or human-verified without review.
- Do not automatically remove models as recovery. Report disk pressure and stop
  safely between checkpoints.

## Completion reporting

Only report completion after validation succeeds. Include video count/duration,
Japanese and Chinese success/failure counts, unresolved warnings, report paths,
elapsed time, disk use, bilingual status, confirmation originals were not
modified, and whether retained models permit immediate offline reuse.
