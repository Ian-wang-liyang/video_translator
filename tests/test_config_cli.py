import importlib.util
import json
from pathlib import Path

import pytest

from subtitle_pipeline import backends, cli
from subtitle_pipeline.cli import (
    ActiveRunnerError,
    approve_sample,
    build_parser,
    run_lock,
    runner_lock_state,
    sample_review_is_current,
    settings_fingerprint,
    status,
)
from subtitle_pipeline.config import load_settings


def load_bootstrap_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_download_models_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("download_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_cpu_configuration(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        '[backend]\ntranscription="faster-whisper"\ntranslation="llama-cpp"\ndevice="cpu"\n',
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.device == "cpu"
    assert settings.transcription_backend == "faster-whisper"
    assert len(settings_fingerprint(settings)) == 64


def test_pipeline_revisions_invalidate_approval_fingerprint(tmp_path: Path, monkeypatch):
    settings = load_settings(tmp_path)
    original = settings_fingerprint(settings)
    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSCRIPTION_REVISION", "chunked-v5")
    assert settings_fingerprint(settings) != original

    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSCRIPTION_REVISION", "chunked-v4")
    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSLATION_REVISION", "batched-v2")
    assert settings_fingerprint(settings) != original

    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSLATION_REVISION", "batched-v1")
    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSLATION_PROMPT_REVISION", "ja-zh-hans-v2")
    assert settings_fingerprint(settings) != original


def test_example_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    import tomllib
    assert tomllib.loads((root / "config.example.toml").read_text(encoding="utf-8"))
    assert json.loads((root / "models.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("label", "reported_version", "expected"),
    [
        ("CUDA Version", "12.1", "cu121"),
        ("CUDA Version", "12.5", "cu125"),
        ("CUDA UMD Version", "13.3", "cu125"),
        ("CUDA Version", "11.8", None),
    ],
)
def test_cuda_wheel_tag_uses_newest_compatible_wheel(
    monkeypatch, label: str, reported_version: str, expected: str | None
):
    bootstrap = load_bootstrap_module()
    monkeypatch.setattr(bootstrap.shutil, "which", lambda command: "/usr/bin/nvidia-smi")
    result = type("Result", (), {"stdout": f"{label}: {reported_version}"})()
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: result)
    assert bootstrap.cuda_wheel_tag() == expected


def test_partial_model_snapshot_is_not_complete(tmp_path: Path):
    download_models = load_download_models_module()
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "config.json").write_text("{}", encoding="utf-8")
    spec = {"repo": "owner/model", "revision": "abc", "directory": "model"}

    assert not download_models.model_is_complete(destination, spec)
    (destination / download_models.COMPLETE_MARKER).write_text("owner/model@abc\n", encoding="utf-8")
    assert download_models.model_is_complete(destination, spec)


def test_file_model_must_be_nonempty(tmp_path: Path):
    download_models = load_download_models_module()
    destination = tmp_path / "model.gguf"
    spec = {"filename": "model.gguf"}

    destination.touch()
    assert not download_models.model_is_complete(destination, spec)
    destination.write_bytes(b"model")
    assert download_models.model_is_complete(destination, spec)


def test_windows_cuda_dll_directory_is_added(tmp_path: Path, monkeypatch):
    cublas_bin = tmp_path / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
    runtime_bin = tmp_path / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin"
    cublas_bin.mkdir(parents=True)
    runtime_bin.mkdir(parents=True)
    (cublas_bin / "cublas64_12.dll").touch()
    (runtime_bin / "cudart64_12.dll").touch()
    added: list[str] = []
    monkeypatch.setattr(backends.os, "name", "nt")
    monkeypatch.setattr(backends.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        backends.os,
        "add_dll_directory",
        lambda path: added.append(path) or object(),
        raising=False,
    )

    backends.configure_windows_cuda_dlls()

    assert added == [str(cublas_bin), str(runtime_bin)]
    assert backends.os.environ["PATH"].split(backends.os.pathsep, 1)[0] == str(cublas_bin)


def test_run_lock_is_released(tmp_path: Path):
    settings = load_settings(tmp_path)
    with run_lock(settings):
        assert (settings.runtime_dir / "state" / "runner.lock").is_file()
        assert runner_lock_state(settings) == ("active", cli.os.getpid())
    assert not (settings.runtime_dir / "state" / "runner.lock").exists()
    assert runner_lock_state(settings) == ("absent", None)


def test_run_lock_rejects_duplicate(tmp_path: Path):
    settings = load_settings(tmp_path)
    with run_lock(settings):
        with pytest.raises(ActiveRunnerError):
            with run_lock(settings):
                pass
        assert not list((settings.runtime_dir / "state").glob(".runner.lock-*.json"))


def test_run_lock_reclaims_definitively_stale_owner(tmp_path: Path, monkeypatch):
    settings = load_settings(tmp_path)
    lock = settings.runtime_dir / "state" / "runner.lock"
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text('{"pid": 1234, "token": "stale"}\n', encoding="utf-8")
    monkeypatch.setattr(cli, "process_alive", lambda pid: False)

    with run_lock(settings):
        owner = json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] != 1234
        assert owner["token"] != "stale"


def test_run_lock_fails_closed_when_owner_is_not_readable(tmp_path: Path):
    settings = load_settings(tmp_path)
    lock = settings.runtime_dir / "state" / "runner.lock"
    lock.mkdir(parents=True)
    assert runner_lock_state(settings) == ("unreadable", None)

    with pytest.raises(ActiveRunnerError, match="ownership is missing or unreadable"):
        with run_lock(settings):
            pass
    assert not list((settings.runtime_dir / "state").glob(".runner.lock-*.json"))


def test_sample_minutes_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample", "--minutes", "0"])


def test_process_alive_uses_non_signaling_windows_check(monkeypatch):
    checked: list[int] = []
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli, "windows_process_alive", lambda pid: checked.append(pid) or True)
    monkeypatch.setattr(cli.os, "kill", lambda *_: pytest.fail("os.kill must not be used on Windows"))

    assert cli.process_alive(1234)
    assert checked == [1234]


def test_empty_project_requests_videos(tmp_path: Path, monkeypatch):
    (tmp_path / "videos").mkdir()
    settings = load_settings(tmp_path)
    monkeypatch.setattr("subtitle_pipeline.cli.doctor", lambda _: {"ok": True})
    monkeypatch.setattr("subtitle_pipeline.cli.core.videos", lambda: [])
    assert status(settings)["next_action"] == "add_videos"


def test_status_reads_non_ascii_sample_review_as_utf8(tmp_path: Path, monkeypatch):
    video_dir = tmp_path / "videos"
    sample_dir = tmp_path / ".subtitle-tools" / "sample"
    video_dir.mkdir()
    sample_dir.mkdir(parents=True)
    video = video_dir / "日本語.mp4"
    video.write_bytes(b"video")
    japanese = sample_dir / "日本語.sample.ja.srt"
    chinese = sample_dir / "日本語.sample.zh-Hans.srt"
    japanese.write_text("1\n00:00:00,000 --> 00:00:01,000\n日本語\n", encoding="utf-8")
    chinese.write_text("1\n00:00:00,000 --> 00:00:01,000\n简体中文\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    cli.write_sample_review(settings, video, japanese, chinese)
    monkeypatch.setattr("subtitle_pipeline.cli.doctor", lambda _: {"ok": True})
    monkeypatch.setattr("subtitle_pipeline.cli.core.videos", lambda: [video])

    assert status(settings)["next_action"] == "review_sample"


@pytest.mark.parametrize(
    "config",
    [
        '[paths]\nruntime="../elsewhere"\n',
        '[models]\nwhisper="../../outside-model"\n',
    ],
)
def test_configured_paths_cannot_escape_their_private_roots(tmp_path: Path, config: str):
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match="must remain inside"):
        load_settings(tmp_path)


def test_video_path_may_be_absolute_and_outside_repository(tmp_path: Path):
    external = tmp_path.parent / "external-videos"
    (tmp_path / "config.toml").write_text(f'[paths]\nvideos="{external.as_posix()}"\n', encoding="utf-8")
    assert load_settings(tmp_path).video_dir == external.resolve()


def test_sample_approval_requires_a_completed_current_review(tmp_path: Path):
    settings = load_settings(tmp_path)
    with pytest.raises(Exception, match="no completed sample"):
        approve_sample(settings, "reviewed")


def test_sample_review_artifacts_cannot_escape_sample_directory(tmp_path: Path):
    settings = load_settings(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    review = {
        "fingerprint": settings_fingerprint(settings),
        "status": "awaiting_review",
        "japanese_file": "../../secret.txt",
        "chinese_file": "../../secret.txt",
        "japanese_sha256": "unused",
        "chinese_sha256": "unused",
    }

    assert not sample_review_is_current(settings, review)
