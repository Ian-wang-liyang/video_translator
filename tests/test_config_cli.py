import json
from pathlib import Path

import pytest

from subtitle_pipeline.cli import ActiveRunnerError, run_lock, settings_fingerprint, status
from subtitle_pipeline.config import load_settings


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
    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSCRIPTION_REVISION", "chunked-v2")
    assert settings_fingerprint(settings) != original

    monkeypatch.setattr("subtitle_pipeline.cli.core.TRANSCRIPTION_REVISION", "chunked-v1")
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


def test_run_lock_is_released(tmp_path: Path):
    settings = load_settings(tmp_path)
    with run_lock(settings):
        assert (settings.runtime_dir / "state" / "runner.lock").is_dir()
    assert not (settings.runtime_dir / "state" / "runner.lock").exists()


def test_run_lock_rejects_duplicate(tmp_path: Path):
    settings = load_settings(tmp_path)
    with run_lock(settings):
        with pytest.raises(ActiveRunnerError):
            with run_lock(settings):
                pass


def test_empty_project_requests_videos(tmp_path: Path, monkeypatch):
    (tmp_path / "videos").mkdir()
    settings = load_settings(tmp_path)
    monkeypatch.setattr("subtitle_pipeline.cli.doctor", lambda _: {"ok": True})
    monkeypatch.setattr("subtitle_pipeline.cli.core.videos", lambda: [])
    assert status(settings)["next_action"] == "add_videos"
