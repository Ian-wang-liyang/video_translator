from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def cuda_wheel_tag() -> str | None:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return None
    result = subprocess.run([nvidia], capture_output=True, text=True, check=False)
    match = re.search(r"CUDA Version:\s*(12\.[1-5])", result.stdout)
    return "cu" + match.group(1).replace(".", "") if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the local runtime and download selected models")
    parser.add_argument("--backend", choices=("auto", "mac", "windows-cuda", "windows-cpu"), default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Bootstrap requires Python 3.12")
    backend = args.backend
    if backend == "auto":
        backend = "mac" if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} else (
            "windows-cuda" if cuda_wheel_tag() else "windows-cpu"
        )
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH")
    if not venv_python().exists():
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    local_config = ROOT / "config.toml"
    if not local_config.exists():
        if backend == "mac":
            transcription, translation, device = "mlx", "mlx", "metal"
        else:
            transcription, translation = "faster-whisper", "llama-cpp"
            device = "cuda" if backend == "windows-cuda" else "cpu"
        local_config.write_text(
            "[backend]\n"
            f'transcription = "{transcription}"\n'
            f'translation = "{translation}"\n'
            f'device = "{device}"\n',
            encoding="utf-8",
        )
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    extra = "mac" if backend == "mac" else "windows"
    install = [python, "-m", "pip", "install", "-e", f"{ROOT}[{extra}]"]
    if args.offline:
        install.append("--no-index")
    run(install)
    if backend == "windows-cuda":
        tag = cuda_wheel_tag()
        if not tag:
            raise RuntimeError("No supported CUDA 12.1-12.5 runtime detected; use --backend windows-cpu")
        run([python, "-m", "pip", "install", "--force-reinstall", "llama-cpp-python==0.3.23",
             "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{tag}"])
    model_platform = "mac" if backend == "mac" else "windows"
    command = [python, str(ROOT / "scripts" / "download_models.py"), "--platform", model_platform]
    if args.offline:
        command.append("--offline")
    run(command)
    run([python, "-m", "subtitle_pipeline", "--json", "doctor"])
    print(f"Bootstrap complete: backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
