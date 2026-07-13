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
    decode_window_seconds: int
    vad_threshold: float
    foreground_min_dbfs: float
    translation_context_cues: int
    translation_n_ctx: int
    translation_gpu_layers: int
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
        else "faster-whisper-large-v3"
    )
    translation_name = (
        "qwen3-4b-instruct-4bit" if translation == "mlx"
        else "Qwen3-8B-Q4_K_M.gguf"
    )
    model_root = runtime / "models"
    whisper_model = _confined_path(model_root, models.get("whisper", whisper_name), "whisper model")
    translation_model = _confined_path(model_root, models.get("translation", translation_name), "translation model")
    processing = data.get("processing", {})
    chunk_seconds = int(processing.get("chunk_seconds", 300))
    decode_window_seconds = int(processing.get("decode_window_seconds", 30))
    vad_threshold = float(processing.get("vad_threshold", 0.65))
    foreground_min_dbfs = float(processing.get("foreground_min_dbfs", -36.0))
    translation_context_cues = int(processing.get("translation_context_cues", 2))
    translation_n_ctx = int(processing.get("translation_n_ctx", 4096))
    translation_gpu_layers = int(processing.get("translation_gpu_layers", -1))
    if chunk_seconds <= 0:
        raise ValueError("processing.chunk_seconds must be positive")
    if decode_window_seconds <= 0 or decode_window_seconds > chunk_seconds:
        raise ValueError("processing.decode_window_seconds must be positive and no larger than chunk_seconds")
    if not 0 < vad_threshold < 1:
        raise ValueError("processing.vad_threshold must be between zero and one")
    if foreground_min_dbfs > 0:
        raise ValueError("processing.foreground_min_dbfs must be zero or negative")
    if translation_context_cues < 0:
        raise ValueError("processing.translation_context_cues cannot be negative")
    if translation_n_ctx <= 0:
        raise ValueError("processing.translation_n_ctx must be positive")
    if translation_gpu_layers < -1:
        raise ValueError("processing.translation_gpu_layers must be -1 or non-negative")
    return Settings(
        root=root,
        runtime_dir=runtime,
        video_dir=videos,
        transcription_backend=transcription,
        translation_backend=translation,
        device=device,
        chunk_seconds=chunk_seconds,
        decode_window_seconds=decode_window_seconds,
        vad_threshold=vad_threshold,
        foreground_min_dbfs=foreground_min_dbfs,
        translation_context_cues=translation_context_cues,
        translation_n_ctx=translation_n_ctx,
        translation_gpu_layers=translation_gpu_layers,
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
