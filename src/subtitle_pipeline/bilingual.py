#!/usr/bin/env python3
"""Create stacked Japanese/Chinese SRTs from validated aligned sidecars."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .config import load_settings

SETTINGS = load_settings()
ROOT = SETTINGS.root
VIDEO_DIR = SETTINGS.video_dir
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
TIMING_RE = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")


def parse_srt(path: Path) -> list[tuple[int, str, str]]:
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    if not content:
        raise ValueError(f"empty subtitle: {path.name}")
    cues: list[tuple[int, str, str]] = []
    for block in re.split(r"\n{2,}", content):
        lines = block.splitlines()
        if len(lines) < 3 or not TIMING_RE.match(lines[1].strip()):
            raise ValueError(f"malformed subtitle block: {path.name}")
        cues.append((int(lines[0]), lines[1].strip(), "\n".join(lines[2:]).strip()))
    return cues


def atomic_write(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def generate_bilingual() -> dict:
    failures: list[str] = []
    completed = 0
    videos = sorted(
        path for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    for video in videos:
        japanese_path = video.with_suffix(".ja.srt")
        chinese_path = video.with_suffix(".zh-Hans.srt")
        output_path = video.with_suffix(".ja-zh-Hans.srt")
        try:
            japanese = parse_srt(japanese_path)
            chinese = parse_srt(chinese_path)
            if len(japanese) != len(chinese):
                raise ValueError("Japanese/Chinese cue counts differ")
            blocks: list[str] = []
            for source, translated in zip(japanese, chinese, strict=True):
                if source[:2] != translated[:2] or not translated[2]:
                    raise ValueError(f"cue alignment failure at index {source[0]}")
                blocks.append(f"{source[0]}\n{source[1]}\n{translated[2]}\n{source[2]}")
            atomic_write(output_path, "\n\n".join(blocks) + "\n")
            completed += 1
        except Exception as exc:
            failures.append(f"{video.name}: {exc}")
    return {"completed": completed, "total": len(videos), "failures": failures}


def main() -> int:
    result = generate_bilingual()
    for failure in result["failures"]:
        print(f"SKIP bilingual: {failure}", file=sys.stderr, flush=True)
    print(f"Bilingual outputs: {result['completed']}/{result['total']}", flush=True)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
