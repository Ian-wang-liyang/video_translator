from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_TRACKED_BYTES = 10_000_000
PRIVATE_PREFIXES = (".subtitle-tools/", ".venv/")
GENERATED_SUFFIXES = (".ja.srt", ".zh-Hans.srt", ".ja-zh-Hans.srt", ".partial")


def main() -> int:
    output = subprocess.check_output(["git", "ls-files", "-z"])
    names = [name for name in output.decode().split("\0") if name]
    failures: list[str] = []
    for name in names:
        path = Path(name)
        if name.startswith(PRIVATE_PREFIXES):
            failures.append(f"private runtime path tracked: {name}")
        if name.startswith("videos/") and name != "videos/.gitkeep":
            failures.append(f"private video/output tracked: {name}")
        if name.endswith(GENERATED_SUFFIXES):
            failures.append(f"generated output tracked: {name}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"oversized file tracked: {name} ({path.stat().st_size} bytes)")
    if failures:
        print(*failures, sep="\n", file=sys.stderr)
        return 1
    print(f"Repository audit passed: {len(names)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
