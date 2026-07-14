# Contributing

Thanks for helping improve the offline subtitle pipeline. Changes should protect
private media, preserve resumability, and keep human review gates explicit.

## Before you start

1. Read [AGENTS.md](AGENTS.md), the authoritative safety and workflow contract.
2. Read [README.md](README.md) and [AI_OPERATIONS.md](AI_OPERATIONS.md).
3. Confirm no pipeline runner is active with
   `python -m subtitle_pipeline --json status`.
4. Open an issue or agree on scope before making a large behavioral, model, or
   output-format change.

Never add real videos, generated subtitles, model files, runtime state, samples,
reports, logs, caches, approval records, or quarantine contents to a change.
Use synthetic fixtures in tests.

## Local development

Use Python 3.12 and install the lightweight development dependencies without
bootstrapping or downloading inference models:

```text
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.

Before submitting a change, run:

```text
python -m py_compile <every tracked Python file>
python -m ruff check src tests scripts
python -m pytest
python -m subtitle_pipeline --help
python -m subtitle_pipeline --json effective-config
python scripts/audit_repo.py
git status --short --ignored
git ls-files
```

The smoke checks must not download models or process private media. Review the
ignored and tracked-file listings before staging anything.
`effective-config` contains resolved local paths; use it for local verification
and redact its output before sharing it.
The `py_compile` placeholder means passing the tracked `.py` paths using the
file-listing syntax appropriate to your shell.

## Change requirements

- Keep `AGENTS.md`, `README.md`, `AI_OPERATIONS.md`, `config.example.toml`, CLI
  help, and relevant tests synchronized.
- Add or update tests for behavior, safety gates, validation, path handling, and
  failure cases.
- Preserve atomic writes, provenance checks, quarantine, source immutability,
  and one-video-at-a-time processing.
- Do not weaken sample or first-video review gates for convenience.
- Update [CHANGELOG.md](CHANGELOG.md) under `Unreleased` for user-visible changes.
- Keep commits focused and explain the user-visible reason for the change.

## Pull request checklist

- [ ] The working tree contains no private or generated data.
- [ ] Documentation and CLI help match the implementation.
- [ ] Ruff, tests, compilation, smoke checks, and repository audit pass.
- [ ] New backend/model/prompt behavior requires renewed sample and full-video
      review.
- [ ] The change does not claim human-level subtitle quality without human
      review.

## Licensing

The repository does not currently grant a license for reuse or redistribution.
Contributions should not be solicited or accepted publicly until the owner has
selected a license and an appropriate contribution policy.
