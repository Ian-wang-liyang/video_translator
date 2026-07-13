from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


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
        if destination.exists() and any(destination.iterdir() if destination.is_dir() else [destination]):
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
        print(f"Downloaded: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
