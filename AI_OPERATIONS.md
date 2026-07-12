# Generic AI-terminal operations

This runbook is vendor-neutral. An AI assistant should inspect structured state
before changing files or starting inference.

## Start every session

Run these commands through the active virtual-environment Python:

```text
python -m subtitle_pipeline --json doctor
python -m subtitle_pipeline --json inventory
python -m subtitle_pipeline --json status
```

`--json` is a global option and must precede the subcommand. Follow
`next_action`; do not infer activity from a PID file alone.

Exit codes: `0` success, `10` validation failure, `20` configuration error, `21`
missing dependency/model, `30` active-runner conflict, `40` inference failure,
and `130` interruption.

## Safe decisions

- `bootstrap`: run the non-interactive bootstrap with the appropriate backend.
- `add_videos`: place top-level supported video files in ignored `videos/`, then
  rerun inventory and status.
- `run_sample`: run a five-minute sample only; do not start the collection.
- `review_sample`: surface sample paths and wait for human linguistic review.
- `process_first_video`: run `process`; it stops after the first completed video.
- `review_first_video`: surface the first video outputs for human review, then use
  `approve-video-gate --note "..."` only with explicit human approval.
- `resume`: start `process`; atomic outputs and validated chunks are reused.
- `monitor`: use `python -m subtitle_pipeline --json status` and logs; never
  launch a duplicate runner.
- `resolve_failure`: inspect JSONL/human logs and the earliest invalid stage.
- `validate`: run full validation, then generate bilingual outputs if requested.

Never approve a sample on behalf of a human, delete private media, remove models,
rename videos, clear quarantine, or publish a repository without explicit user
authorization. Use `clean --dry-run` before proposing any cleanup.

## Failure handling

Preserve completed atomic outputs. Before manually replacing a suspect generated
SRT, move it into `.subtitle-tools/quarantine/`. For decode failures, inspect
with FFprobe and test a short sequential FFmpeg decode. For repetition or
hallucination, stop translation, identify the earliest bad chunk, adjust the
quality gate, and rerun the sample.

The human-readable log is `.subtitle-tools/logs/pipeline.log`; structured status
records, approvals, provenance, locks, and transcription checkpoints live under
`.subtitle-tools/state`. Runtime paths are private and must never be committed.
