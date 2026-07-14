# Security policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the latest revision of
the main development branch; older revisions are not supported.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose private video,
subtitle content, local paths, model credentials, or arbitrary file access.
Contact the repository owner privately through the hosting platform's private
security-reporting channel. If no private channel is configured, request one
without including exploit details or private data.

Include:

- the affected revision and platform;
- impact and realistic attack scenario;
- minimal reproduction using synthetic data;
- relevant logs with usernames, paths, media names, tokens, and content removed;
- any suggested mitigation.

The owner should acknowledge a complete report, assess severity, prepare a fix
and regression test, and coordinate disclosure. No response-time guarantee is
made while this remains a personal pre-1.0 project.

## Security boundaries

- Video content, subtitles, logs, reports, state, and models are local private
  data and must not be uploaded as diagnostics.
- `config.toml` is untracked and may reveal external paths or local settings.
- Bootstrap is the networked trust boundary: it installs dependencies and
  downloads pinned model revisions. Normal inference is intended to run offline.
- Configured runtime and model paths are confined to their private repository
  roots. A configured video directory may be external and is therefore treated
  as untrusted private input.
- Source videos are immutable. Any report of source modification, path escape,
  unsafe lock recovery, provenance bypass, or unintended network access during
  processing should be treated as high priority.

## Operational hygiene

Keep Python, FFmpeg, GPU drivers, and the operating system patched. Review
dependency and pinned-model changes before bootstrap. Run
`python scripts/audit_repo.py`, `git status --ignored`, and `git ls-files` before
sharing a branch or archive.
