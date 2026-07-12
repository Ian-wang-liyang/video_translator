# AGENTS.md — Offline subtitle pipeline

## Mission and safety

Maintain a reusable local Japanese-to-Simplified-Chinese subtitle pipeline.
Original videos are immutable. Never delete or rename videos, models, quarantine
artifacts, or completed subtitles without explicit user approval.

Treat this file as maintained repository documentation. Update it in the same
change whenever a durable workflow, command, layout, or constraint changes.

## Repository and private data

- Tracked source lives in `src/subtitle_pipeline`; setup utilities live in
  `scripts`; tests live in `tests`.
- `videos/` contains private inputs and generated sidecars. Only `.gitkeep` may
  be tracked.
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

Bootstrap a fresh clone with `python scripts/bootstrap.py --backend auto
--non-interactive`. Model downloads are the only required model-network phase.

## Processing invariants

- Process one video at a time: transcribe, validate Japanese, translate, validate
  Chinese, checkpoint title, then advance.
- Decode long media sequentially into temporary five-minute mono PCM chunks.
  Validate and checkpoint each chunk, then delete temporary audio.
- Use temperature fallback and `condition_on_previous_text=False` for MLX
  Whisper. Use faster-whisper with VAD on Windows.
- Reject empty, malformed, stale, misaligned, untranslated, internally
  repetitive, cross-cue repetitive, or implausibly long outputs.
- Reuse Japanese outputs only when their provenance fingerprint matches the
  current video, output/clip, backend, model, and transcription revision. Reuse
  Chinese outputs only when their fingerprint matches the exact Japanese
  source, backend, model, and translation/prompt revisions.
- Write outputs atomically via `.partial`. Preserve all existing artifacts under
  `.subtitle-tools/quarantine/`; quarantine suspect outputs before manual
  replacement.
- A bilingual cue places Simplified Chinese above Japanese and is generated only
  from exactly aligned monolingual SRTs.

## Change and execution discipline

- Read `README.md`, `AI_OPERATIONS.md`, configuration, entrypoints, and relevant
  tests before editing or launching work.
- Confirm no runner is active. The atomic runner lock must reject duplicates;
  status must distinguish active and stale PID files.
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
