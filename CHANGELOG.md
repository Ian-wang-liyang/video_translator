# Changelog

Notable user-visible changes are recorded here. This project follows the spirit
of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) for published
versions.

## [Unreleased]

### Added

- Confidence-aware retention for moderately quiet wanted speech, conditional
  recovery of loud high-confidence near-agreement, and overlap-owned source
  chunks that protect dialogue at media seek boundaries.
- A structured project overview, command reference, limitations, privacy model,
  and development guidance.
- Contribution and security policies.

### Changed

- Translation now uses six surrounding cues by default, carries accepted prior
  translations across batch boundaries, varies failed-format retries, and
  selectively reviews suspicious or context-sensitive cues in a second pass.
- Translation provenance and quality approvals are invalidated for the new
  contextual-review prompt and pipeline revision.
- Operator documentation now uses the actual `review_full_video` status action,
  warns that effective configuration contains private local paths, and
  distinguishes seek-time decode warnings from confirmed source corruption.

### Baseline capabilities

- Version 0.1.0 provides offline Japanese transcription, Simplified Chinese
  translation, and aligned bilingual SRT generation.
- Resumable five-minute checkpoints, atomic output writes, quarantine, and
  provenance-bound reuse.
- Human-reviewed sample and first-complete-video gates.
- MLX support on Apple Silicon and faster-whisper/llama.cpp support on Windows
  and Linux/WSL, including optional CUDA acceleration.
- Foreground gating, overlapping decode windows, repetition detection, rescue
  decoding, specialist agreement checks, and validation reports.
- Machine-readable doctor, inventory, status, and effective-configuration
  commands.

[Unreleased]: https://github.com/Ian-wang-liyang/video_translator
