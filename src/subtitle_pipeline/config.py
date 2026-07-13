from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    runtime_dir: Path
    video_dir: Path
    transcription_backend: str
    translation_backend: str
    device: str
    chunk_seconds: int
    whisper_model: Path
    translation_model: Path


def project_root() -> Path:
    override = os.environ.get("SUBTITLE_PIPELINE_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]


def has_nvidia_gpu() -> bool:
    command = shutil.which("nvidia-smi")
    if not command:
        return False
    return subprocess.run([command, "-L"], capture_output=True, check=False).returncode == 0


def load_settings(root: Path | None = None) -> Settings:
    root = (root or project_root()).resolve()
    data: dict = {}
    local = root / "config.toml"
    if local.exists():
        data = tomllib.loads(local.read_text(encoding="utf-8"))
    runtime = _confined_path(root, data.get("paths", {}).get("runtime", ".subtitle-tools"), "runtime")
    videos = _configured_path(root, data.get("paths", {}).get("videos", "videos"))
    backend = data.get("backend", {})
    system = platform.system()
    machine = platform.machine().lower()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    transcription = backend.get("transcription", "auto")
    translation = backend.get("translation", "auto")
    device = backend.get("device", "auto")
    if transcription == "auto":
        transcription = "mlx" if apple_silicon else "faster-whisper"
    if translation == "auto":
        translation = "mlx" if apple_silicon else "llama-cpp"
    if device == "auto":
        device = "metal" if apple_silicon else ("cuda" if has_nvidia_gpu() else "cpu")
    models = data.get("models", {})
    whisper_name = (
        "whisper-large-v3-turbo" if transcription == "mlx"
        else "faster-whisper-large-v3-turbo"
    )
    translation_name = (
        "qwen3-4b-instruct-4bit" if translation == "mlx"
        else "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    )
    model_root = runtime / "models"
    whisper_model = _confined_path(model_root, models.get("whisper", whisper_name), "whisper model")
    translation_model = _confined_path(model_root, models.get("translation", translation_name), "translation model")
    chunk_seconds = int(data.get("processing", {}).get("chunk_seconds", 300))
    if chunk_seconds <= 0:
        raise ValueError("processing.chunk_seconds must be positive")
    return Settings(
        root=root,
        runtime_dir=runtime,
        video_dir=videos,
        transcription_backend=transcription,
        translation_backend=translation,
        device=device,
        chunk_seconds=chunk_seconds,
        whisper_model=whisper_model,
        translation_model=translation_model,
    )


def _confined_path(parent: Path, configured: str, label: str) -> Path:
    candidate = (parent / configured).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"configured {label} path must remain inside {parent}") from exc
    return candidate


def _configured_path(root: Path, configured: str) -> Path:
    """Resolve a repository-relative or absolute user-configured path."""
    return (root / Path(configured).expanduser()).resolve()
