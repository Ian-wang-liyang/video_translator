from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from . import __version__, bilingual, core
from .config import Settings, load_settings

EXIT_OK = 0
EXIT_VALIDATION = 10
EXIT_CONFIG = 20
EXIT_DEPENDENCY = 21
EXIT_ACTIVE_RUNNER = 30
EXIT_INFERENCE = 40
EXIT_INTERRUPTED = 130


def emit(payload: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def settings_fingerprint(settings: Settings) -> str:
    material = "|".join(
        [settings.transcription_backend, settings.translation_backend, settings.device,
         str(settings.chunk_seconds), str(settings.whisper_model), str(settings.translation_model),
         core.TRANSCRIPTION_REVISION, core.TRANSLATION_REVISION, core.TRANSLATION_PROMPT_REVISION]
    )
    return hashlib.sha256(material.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_fingerprint(settings: Settings, review: dict) -> str:
    material = json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((settings_fingerprint(settings) + "|" + material).encode()).hexdigest()


def confined_sample_artifact(settings: Settings, configured: str) -> Path:
    sample_dir = (settings.runtime_dir / "sample").resolve()
    candidate = (sample_dir / configured).resolve()
    candidate.relative_to(sample_dir)
    return candidate


def sample_review_is_current(settings: Settings, review: dict) -> bool:
    try:
        japanese = confined_sample_artifact(settings, review["japanese_file"])
        chinese = confined_sample_artifact(settings, review["chinese_file"])
        return (
            review["fingerprint"] == settings_fingerprint(settings)
            and review["status"] == "awaiting_review"
            and file_digest(japanese) == review["japanese_sha256"]
            and file_digest(chinese) == review["chinese_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def video_gate_fingerprint(settings: Settings, video: Path) -> str:
    japanese = video.with_suffix(".ja.srt")
    chinese = video.with_suffix(".zh-Hans.srt")
    source_cues = core.parse_srt(japanese)
    translated_cues = core.parse_srt(chinese)
    errors = core.transcript_quality_errors(source_cues)
    aligned = len(source_cues) == len(translated_cues) and all(
        source.index == translated.index
        and source.timing == translated.timing
        and bool(translated.text.strip())
        and not re.search(r"[\u3040-\u30ff]", translated.text)
        for source, translated in zip(source_cues, translated_cues, strict=True)
    )
    transcription_metadata = core.provenance_path("transcription", japanese)
    translation_metadata = core.provenance_path("translation", chinese)
    try:
        recorded_transcription = json.loads(transcription_metadata.read_text())["fingerprint"]
        transcription_current = recorded_transcription == core.transcription_fingerprint(video, japanese, "0")
        recorded_translation = json.loads(translation_metadata.read_text())["fingerprint"]
        translation_current = recorded_translation == core.translation_fingerprint(japanese)
    except Exception:
        transcription_current = translation_current = False
    if errors or not aligned or not transcription_current or not translation_current:
        raise ConfigurationError(
            "first-video outputs are not aligned, quality-valid, and current; rerun process before approval"
        )
    if not core.existing_title_mapping().get(video.name):
        raise ConfigurationError("the first video has no completed title checkpoint; rerun process before approval")
    stat = video.stat()
    material = "|".join(
        [settings_fingerprint(settings), video.name, str(stat.st_size), str(stat.st_mtime_ns),
         file_digest(japanese), file_digest(chinese)]
    )
    return hashlib.sha256(material.encode()).hexdigest()


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        return windows_process_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def windows_process_alive(pid: int) -> bool:
    """Check a Windows PID without sending CTRL_C_EVENT via os.kill(pid, 0)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    close_handle(handle)
    return True


@contextmanager
def run_lock(settings: Settings):
    state = settings.runtime_dir / "state"
    lock = state / "runner.lock"
    pid_file = state / "runner.pid"
    state.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            pid = -1
        if pid > 0 and process_alive(pid):
            raise ActiveRunnerError(f"runner PID {pid} is active")
        shutil.rmtree(lock, ignore_errors=True)
        lock.mkdir()
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        yield
    finally:
        pid_file.unlink(missing_ok=True)
        shutil.rmtree(lock, ignore_errors=True)


class ActiveRunnerError(RuntimeError):
    pass


def doctor(settings: Settings) -> dict:
    checks: dict[str, dict] = {}
    for command in ("ffmpeg", "ffprobe"):
        path = shutil.which(command)
        checks[command] = {"ok": bool(path), "path": path}
    checks["videos_directory"] = {"ok": settings.video_dir.is_dir(), "path": str(settings.video_dir)}
    checks["whisper_model"] = {"ok": settings.whisper_model.exists(), "path": str(settings.whisper_model)}
    checks["translation_model"] = {"ok": settings.translation_model.exists(), "path": str(settings.translation_model)}
    transcription_module = "mlx_whisper" if settings.transcription_backend == "mlx" else "faster_whisper"
    translation_module = "mlx_lm" if settings.translation_backend == "mlx" else "llama_cpp"
    checks["transcription_module"] = {
        "ok": importlib.util.find_spec(transcription_module) is not None,
        "module": transcription_module,
    }
    checks["translation_module"] = {
        "ok": importlib.util.find_spec(translation_module) is not None,
        "module": translation_module,
    }
    return {
        "ok": all(item["ok"] for item in checks.values()),
        "version": __version__, "backend": settings.transcription_backend,
        "translation_backend": settings.translation_backend, "device": settings.device,
        "checks": checks,
    }


def inventory(settings: Settings) -> dict:
    items = core.videos()
    total_size = sum(path.stat().st_size for path in items)
    japanese = sum(
        path.with_suffix(".ja.srt").stat().st_size > 0
        for path in items
        if path.with_suffix(".ja.srt").exists()
    )
    chinese = sum(
        path.with_suffix(".zh-Hans.srt").stat().st_size > 0
        for path in items
        if path.with_suffix(".zh-Hans.srt").exists()
    )
    return {
        "video_count": len(items),
        "total_bytes": total_size,
        "japanese_complete": japanese,
        "chinese_complete": chinese,
        "free_bytes": shutil.disk_usage(settings.root).free,
    }


def status(settings: Settings) -> dict:
    data = inventory(settings)
    pid_file = settings.runtime_dir / "state" / "runner.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        pid = None
    active = bool(pid and process_alive(pid))
    approval = settings.runtime_dir / "state" / "sample-approved.json"
    approved = False
    if approval.exists():
        try:
            approval_data = json.loads(approval.read_text())
            review_data = json.loads((settings.runtime_dir / "state" / "sample-review.json").read_text())
            approved = (
                sample_review_is_current(settings, review_data)
                and approval_data["fingerprint"] == settings_fingerprint(settings)
                and approval_data["review_fingerprint"] == review_fingerprint(settings, review_data)
            )
        except Exception:
            pass
    gate = settings.runtime_dir / "state" / "video-gate-approved.json"
    gate_approved = False
    if gate.exists():
        try:
            gate_data = json.loads(gate.read_text())
            first = core.videos()[0] if core.videos() else None
            gate_approved = bool(
                first
                and gate_data["settings_fingerprint"] == settings_fingerprint(settings)
                and gate_data["artifact_fingerprint"] == video_gate_fingerprint(settings, first)
            )
        except Exception:
            pass
    if not doctor(settings)["ok"]:
        action = "bootstrap"
    elif active:
        action = "monitor"
    elif data["video_count"] == 0:
        action = "add_videos"
    elif not approved:
        review = settings.runtime_dir / "state" / "sample-review.json"
        review_current = False
        if review.exists():
            try:
                review_current = json.loads(review.read_text())["fingerprint"] == settings_fingerprint(settings)
            except Exception:
                pass
        action = "review_sample" if review_current else "run_sample"
    elif not gate_approved:
        first = core.videos()[0] if core.videos() else None
        action = (
            "review_first_video"
            if first and first.with_suffix(".zh-Hans.srt").exists()
            else "process_first_video"
        )
    elif data["chinese_complete"] < data["video_count"]:
        action = "resume"
    else:
        action = "validate"
    data.update(
        {
            "runner_active": active,
            "runner_pid": pid,
            "sample_approved": approved,
            "video_gate_approved": gate_approved,
            "next_action": action,
        }
    )
    return data


def write_sample_review(settings: Settings, video: Path, japanese: Path, chinese: Path) -> None:
    path = settings.runtime_dir / "state" / "sample-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": settings_fingerprint(settings),
        "status": "awaiting_review",
        "video": video.name,
        "japanese_file": japanese.name,
        "chinese_file": chinese.name,
        "japanese_sha256": file_digest(japanese),
        "chinese_sha256": file_digest(chinese),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    core.atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def approve_sample(settings: Settings, note: str) -> None:
    review_path = settings.runtime_dir / "state" / "sample-review.json"
    try:
        review = json.loads(review_path.read_text())
    except Exception as exc:
        raise ConfigurationError("no completed sample is awaiting review") from exc
    if not sample_review_is_current(settings, review):
        raise ConfigurationError("the available sample artifacts do not match the current review")
    path = settings.runtime_dir / "state" / "sample-approved.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    core.atomic_write(path, json.dumps({"fingerprint": settings_fingerprint(settings),
                                        "review_fingerprint": review_fingerprint(settings, review), "note": note,
                                        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")


def require_approval(settings: Settings) -> None:
    if not status(settings)["sample_approved"]:
        raise ConfigurationError("current backend/configuration has no approved sample; run sample then approve-sample")


class ConfigurationError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


def clean(settings: Settings, dry_run: bool) -> dict:
    candidates = list(settings.runtime_dir.rglob("*.partial"))
    if not dry_run:
        for path in candidates:
            path.unlink(missing_ok=True)
    return {"dry_run": dry_run, "candidate_count": len(candidates), "paths": [str(p) for p in candidates]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subtitle-pipeline")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON/JSONL")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "inventory", "status", "validate", "bilingual"):
        sub.add_parser(command)
    sub.add_parser("process")
    sample = sub.add_parser("sample")
    sample.add_argument("--video", type=Path)
    sample.add_argument("--minutes", type=int, default=5)
    approve = sub.add_parser("approve-sample")
    approve.add_argument("--note", required=True)
    gate = sub.add_parser("approve-video-gate")
    gate.add_argument("--note", required=True)
    clean_parser = sub.add_parser("clean")
    clean_mode = clean_parser.add_mutually_exclusive_group(required=True)
    clean_mode.add_argument("--dry-run", action="store_true")
    clean_mode.add_argument("--execute", action="store_true")
    sub.add_parser("effective-config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    core.set_json_output(args.json)
    try:
        if args.command == "doctor":
            result = doctor(settings)
            emit(result, args.json)
            return EXIT_OK if result["ok"] else EXIT_DEPENDENCY
        elif args.command == "inventory":
            emit(inventory(settings), args.json)
        elif args.command == "status":
            emit(status(settings), args.json)
        elif args.command == "effective-config":
            effective = {
                **settings.__dict__,
                "root": str(settings.root),
                "runtime_dir": str(settings.runtime_dir),
                "video_dir": str(settings.video_dir),
                "whisper_model": str(settings.whisper_model),
                "translation_model": str(settings.translation_model),
            }
            emit(effective, args.json)
        elif args.command == "sample":
            with run_lock(settings):
                chosen = args.video or (core.videos()[0] if core.videos() else None)
                if chosen is None:
                    raise ConfigurationError("no videos found")
                chosen = chosen.resolve()
                try:
                    chosen.relative_to(settings.video_dir.resolve())
                except ValueError as exc:
                    raise ConfigurationError("sample video must be inside the configured videos directory") from exc
                japanese, chinese = core.sample(chosen, args.minutes * 60)
                write_sample_review(settings, chosen, japanese, chinese)
            emit({"ok": True, "next_action": "review_sample"}, args.json)
        elif args.command == "approve-sample":
            approve_sample(settings, args.note)
            emit({"ok": True, "next_action": "process_first_video"}, args.json)
        elif args.command == "approve-video-gate":
            first = core.videos()[0] if core.videos() else None
            if first is None or not first.with_suffix(".zh-Hans.srt").exists():
                raise ConfigurationError("the first video has no completed Chinese subtitle to approve")
            artifact_fingerprint = video_gate_fingerprint(settings, first)
            path = settings.runtime_dir / "state" / "video-gate-approved.json"
            core.atomic_write(
                path,
                json.dumps(
                    {
                        "settings_fingerprint": settings_fingerprint(settings),
                        "artifact_fingerprint": artifact_fingerprint,
                        "video": first.name,
                        "note": args.note,
                        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
            )
            emit({"ok": True, "next_action": "resume"}, args.json)
        elif args.command == "process":
            require_approval(settings)
            gate_path = settings.runtime_dir / "state" / "video-gate-approved.json"
            gate_approved = False
            if gate_path.exists():
                try:
                    gate_data = json.loads(gate_path.read_text())
                    first = core.videos()[0] if core.videos() else None
                    gate_approved = bool(
                        first
                        and gate_data["settings_fingerprint"] == settings_fingerprint(settings)
                        and gate_data["artifact_fingerprint"] == video_gate_fingerprint(settings, first)
                    )
                except Exception:
                    pass
            with run_lock(settings):
                core.process_collection(max_videos=None if gate_approved else 1)
            next_action = "validate" if gate_approved else "review_first_video"
            emit({"ok": True, "next_action": next_action}, args.json)
        elif args.command == "validate":
            try:
                core.validate()
            except RuntimeError as exc:
                raise ValidationError(str(exc)) from exc
            emit({"ok": True, "next_action": "bilingual"}, args.json)
        elif args.command == "bilingual":
            try:
                core.validate()
            except RuntimeError as exc:
                message = "bilingual generation requires successful current validation: " + str(exc)
                raise ValidationError(message) from exc
            report = bilingual.generate_bilingual()
            code = 1 if report["failures"] else 0
            next_action = "complete" if code == 0 else "resolve_failure"
            emit({"ok": code == 0, "next_action": next_action, **report}, args.json)
            return code
        elif args.command == "clean":
            emit(clean(settings, not args.execute), args.json)
        return EXIT_OK
    except ActiveRunnerError as exc:
        emit({"ok": False, "error": "active_runner", "message": str(exc), "next_action": "monitor"}, args.json)
        return EXIT_ACTIVE_RUNNER
    except ConfigurationError as exc:
        emit({"ok": False, "error": "configuration", "message": str(exc), "next_action": "resolve_failure"}, args.json)
        return EXIT_CONFIG
    except ValidationError as exc:
        emit(
            {"ok": False, "error": "validation", "message": str(exc), "next_action": "resolve_failure"},
            args.json,
        )
        return EXIT_VALIDATION
    except KeyboardInterrupt:
        emit({"ok": False, "error": "interrupted", "next_action": "resume"}, args.json)
        return EXIT_INTERRUPTED
    except Exception as exc:
        payload = {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "next_action": "resolve_failure",
        }
        emit(payload, args.json)
        return EXIT_INFERENCE
