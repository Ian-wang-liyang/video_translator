from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
COMPLETE_MARKER = ".subtitle-model-complete"


def model_is_complete(destination: Path, spec: dict) -> bool:
    if "filename" in spec:
        return destination.is_file() and destination.stat().st_size > 0
    return destination.is_dir() and (destination / COMPLETE_MARKER).is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("mac", "linux", "windows"), required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))[args.platform]
    model_root = ROOT / ".subtitle-tools" / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for spec in manifest.values():
        destination = model_root / spec["directory"]
        if model_is_complete(destination, spec):
            print(f"Model already present: {destination}")
            continue
        if args.offline:
            raise RuntimeError(f"Model missing in offline mode: {destination}")
        if "filename" in spec:
            downloaded = Path(
                hf_hub_download(
                    repo_id=spec["repo"],
                    filename=spec["filename"],
                    revision=spec["revision"],
                )
            )
            shutil.copy2(downloaded, destination)
        else:
            snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=destination)
            (destination / COMPLETE_MARKER).write_text(f'{spec["repo"]}@{spec["revision"]}\n', encoding="utf-8")
        print(f"Downloaded: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
