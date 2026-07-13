from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .config import Settings

_DLL_DIRECTORY_HANDLES: list[object] = []


def configure_windows_cuda_dlls() -> None:
    """Expose pip-installed NVIDIA runtime DLLs to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    cublas_bin = nvidia_root / "cublas" / "bin"
    runtime_bin = nvidia_root / "cuda_runtime" / "bin"
    if not (cublas_bin / "cublas64_12.dll").is_file():
        raise RuntimeError(
            "CUDA transcription requires nvidia-cublas-cu12; rerun bootstrap to install the Windows CUDA runtime"
        )
    if not (runtime_bin / "cudart64_12.dll").is_file():
        raise RuntimeError(
            "CUDA inference requires nvidia-cuda-runtime-cu12; rerun bootstrap to install the Windows CUDA runtime"
        )
    dll_directories = sorted(path for path in nvidia_root.glob("*/bin") if path.is_dir())
    for directory in dll_directories:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in dll_directories), os.environ.get("PATH", "")])


class Transcriber(Protocol):
    def transcribe(self, audio: str, *, clip_timestamps: str = "0") -> dict: ...


class Translator(Protocol):
    def generate(self, instruction: str, max_tokens: int) -> str: ...


class MLXTranscriber:
    def __init__(self, settings: Settings):
        import mlx_whisper
        self.module = mlx_whisper
        self.model = str(settings.whisper_model)

    def transcribe(self, audio: str, *, clip_timestamps: str = "0") -> dict:
        return self.module.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language="ja",
            task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=False,
            word_timestamps=False,
            clip_timestamps=clip_timestamps,
            verbose=None,
        )


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        if settings.device == "cuda":
            configure_windows_cuda_dlls()
        from faster_whisper import WhisperModel
        device = "cuda" if settings.device == "cuda" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(str(settings.whisper_model), device=device, compute_type=compute_type)

    def transcribe(self, audio: str, *, clip_timestamps: str = "0") -> dict:
        options = {}
        if clip_timestamps != "0":
            options["clip_timestamps"] = clip_timestamps
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            temperature=0.0,
            **options,
        )
        return {"segments": [asdict(segment) for segment in segments]}


class MLXTranslator:
    def __init__(self, settings: Settings):
        from mlx_lm import load
        self.model, self.tokenizer = load(str(settings.translation_model))

    def generate(self, instruction: str, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        messages = _messages(instruction)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0), verbose=False,
        ).strip()


class LlamaCppTranslator:
    def __init__(self, settings: Settings):
        if settings.device == "cuda":
            configure_windows_cuda_dlls()
        from llama_cpp import Llama
        layers = -1 if settings.device == "cuda" else 0
        self.model = Llama(
            model_path=str(settings.translation_model), n_ctx=8192,
            n_gpu_layers=layers, verbose=False,
        )

    def generate(self, instruction: str, max_tokens: int) -> str:
        result = self.model.create_chat_completion(
            messages=_messages(instruction), temperature=0.0,
            max_tokens=max_tokens, seed=0,
        )
        return str(result["choices"][0]["message"]["content"]).strip()


def _messages(instruction: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a professional Japanese-to-Simplified-Chinese subtitle translator. "
                "Follow the output format exactly. Never add commentary."
            ),
        },
        {"role": "user", "content": instruction},
    ]


def load_transcriber(settings: Settings) -> Transcriber:
    if settings.transcription_backend == "mlx":
        return MLXTranscriber(settings)
    if settings.transcription_backend == "faster-whisper":
        return FasterWhisperTranscriber(settings)
    raise ValueError(f"Unsupported transcription backend: {settings.transcription_backend}")


def load_translation_backend(settings: Settings) -> Translator:
    if settings.translation_backend == "mlx":
        return MLXTranslator(settings)
    if settings.translation_backend == "llama-cpp":
        return LlamaCppTranslator(settings)
    raise ValueError(f"Unsupported translation backend: {settings.translation_backend}")
