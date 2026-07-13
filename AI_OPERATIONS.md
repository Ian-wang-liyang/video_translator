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

Runner locks publish their ownership atomically. A lock with missing or
unreadable ownership is not automatically removed, because doing so could admit
two inference processes; inspect it and obtain explicit approval before manual
recovery. Structured status reports `runner_lock_state` as `absent`, `active`,
`stale`, or `unreadable`.

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

Approval records are artifact-bound. Any change to a reviewed sample, the first
video, or its monolingual subtitles invalidates the corresponding approval.
Full validation rejects missing, malformed, or stale output provenance, and
`bilingual` refuses to run unless current full validation succeeds. Configured
video paths may be repository-relative or absolute; runtime paths cannot escape
the repository, and configured model paths cannot escape the runtime model
directory. Treat source videos at every location as immutable.

Never approve a sample on behalf of a human, delete private media, remove models,
rename videos, clear quarantine, or publish a repository without explicit user
authorization. Use `clean --dry-run` before proposing any cleanup.

## Failure handling

Preserve completed atomic outputs. Before manually replacing a suspect generated
SRT, move it into `.subtitle-tools/quarantine/`. For decode failures, inspect
with FFprobe and test a short sequential FFmpeg decode. For repetition or
hallucination, stop translation, identify the earliest bad chunk, adjust the
quality gate, and rerun the sample.
Faster-whisper always decodes the unchanged five-minute PCM chunks in independent
30-second windows, then removes segments below the configured foreground dBFS
threshold before normal quality validation.
An explicitly user-authorized sample bypass must name one inventory video and
use `process --video PATH --skip-sample-gate`; it must stop for full-video review.
On Windows, the runner lock also holds a system-sleep inhibition request for the
entire sample or processing operation and releases it when the command exits.
The display power policy is intentionally left unchanged.

The human-readable log is `.subtitle-tools/logs/pipeline.log`; structured status
records, approvals, provenance, locks, and transcription checkpoints live under
`.subtitle-tools/state`. Runtime paths are private and must never be committed.
The pipeline never translates or renames source video titles; processing and
validation cover subtitle artifacts only.
