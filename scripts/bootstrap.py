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
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    major, minor = map(int, match.groups())
    # New NVIDIA drivers can run applications built against older CUDA
    # toolkits. Cap at the newest wheel published by this project.
    if (major, minor) >= (12, 5):
        return "cu125"
    if major == 12 and minor >= 1:
        return f"cu12{minor}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the local runtime and download selected models")
    parser.add_argument(
        "--backend",
        choices=("auto", "mac", "linux-cuda", "linux-cpu", "windows-cuda", "windows-cpu"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-wheel",
        choices=("auto", "cu121", "cu122", "cu123", "cu124", "cu125"),
        default="auto",
        help="CUDA llama.cpp wheel; override detection in hardware-isolated WSL/container sessions",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Bootstrap requires Python 3.12")
    backend = args.backend
    detected_cuda_tag = cuda_wheel_tag()
    if backend == "auto":
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            backend = "mac"
        else:
            system = "windows" if platform.system() == "Windows" else "linux"
            cuda_available = detected_cuda_tag or (args.cuda_wheel if args.cuda_wheel != "auto" else None)
            backend = f"{system}-cuda" if cuda_available else f"{system}-cpu"
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
            device = "cuda" if backend.endswith("-cuda") else "cpu"
        local_config.write_text(
            "[backend]\n"
            f'transcription = "{transcription}"\n'
            f'translation = "{translation}"\n'
            f'device = "{device}"\n',
            encoding="utf-8",
        )
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    extra = "cuda" if backend.endswith("-cuda") else (
        "mac" if backend == "mac" else ("linux" if backend.startswith("linux-") else "windows")
    )
    install = [python, "-m", "pip", "install", "-e", f"{ROOT}[{extra}]"]
    if args.offline:
        install.append("--no-index")
    run(install)
    if backend.endswith("-cuda"):
        tag = detected_cuda_tag if args.cuda_wheel == "auto" else args.cuda_wheel
        if not tag:
            raise RuntimeError("No CUDA 12.1+ compatible NVIDIA driver detected; use the platform CPU backend")
        run([
            python, "-m", "pip", "install", "--force-reinstall", "--only-binary=:all:",
            "llama-cpp-python==0.3.23", "--extra-index-url",
            f"https://abetlen.github.io/llama-cpp-python/whl/{tag}",
        ])
    model_platform = "mac" if backend == "mac" else ("linux" if backend.startswith("linux-") else "windows")
    command = [python, str(ROOT / "scripts" / "download_models.py"), "--platform", model_platform]
    if args.offline:
        command.append("--offline")
    run(command)
    run([python, "-m", "subtitle_pipeline", "--json", "doctor"])
    print(f"Bootstrap complete: backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
