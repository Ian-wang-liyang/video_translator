import json
from pathlib import Path

import pytest

from subtitle_pipeline import cli
from subtitle_pipeline.cli import (
    ActiveRunnerError,
    approve_sample,
    run_lock,
    sample_review_is_current,
    settings_fingerprint,
    status,
)
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


@pytest.mark.parametrize(
    "config",
    [
        '[paths]\nruntime="../elsewhere"\n',
        '[paths]\nvideos="/tmp/videos"\n',
        '[models]\nwhisper="../../outside-model"\n',
    ],
)
def test_configured_paths_cannot_escape_their_private_roots(tmp_path: Path, config: str):
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match="must remain inside"):
        load_settings(tmp_path)


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
