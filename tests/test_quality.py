import json
from pathlib import Path

from subtitle_pipeline import core
from subtitle_pipeline.core import Cue, cue_text_quality_error, filter_repetition_bursts, transcript_quality_errors


def test_filters_repetition_burst():
    segments = [
        {"start": i, "end": i + 1, "text": text}
        for i, text in enumerate(["時間時間時間", "時間時間", "時間", "時間", "普通の会話"])
    ]
    assert [item["text"] for item in filter_repetition_bursts(segments)] == ["普通の会話"]


def test_rejects_latin_garbage_in_japanese():
    assert cue_text_quality_error("さあげー Soci truth")
    assert cue_text_quality_error("普通の日本語です") is None


def test_empty_transcript_fails():
    assert transcript_quality_errors([]) == ["transcript contains zero cues"]


def test_adjacent_duplicate_threshold_applies_to_short_chunks():
    cues = [Cue(i + 1, f"00:00:{i:02d},000 --> 00:00:{i:02d},500", "同じ") for i in range(30)]
    assert any("adjacent" in error or "repetition" in error for error in transcript_quality_errors(cues))


class FakeTranscriber:
    def transcribe(self, audio: str, *, clip_timestamps: str = "0") -> dict:
        return {"segments": [{"start": 0, "end": 1, "text": "新しい字幕"}]}


class FakeTranslator:
    def __init__(self):
        self.response = "[1] 新字幕"

    def generate(self, instruction: str, max_tokens: int) -> str:
        return self.response


def write_srt(path: Path, text: str) -> None:
    path.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n", encoding="utf-8")


def test_sample_transcription_does_not_replace_full_output_provenance(tmp_path: Path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    full_output = tmp_path / "video.ja.srt"
    sample_output = tmp_path / "sample" / "video.ja.srt"
    sample_output.parent.mkdir()
    state = tmp_path / "state"
    monkeypatch.setattr(core, "STATE_DIR", state)
    monkeypatch.setattr(core, "_TRANSCRIBER", FakeTranscriber())

    full_metadata = core.provenance_path("transcription", full_output)
    core.atomic_write(full_metadata, json.dumps({"fingerprint": "old-full"}))
    core.transcribe_one(video, sample_output, clip="0,300")

    assert json.loads(full_metadata.read_text())["fingerprint"] == "old-full"
    assert core.provenance_path("transcription", sample_output) != full_metadata


def test_translation_rebuilds_when_source_text_changes(tmp_path: Path, monkeypatch):
    source = tmp_path / "video.ja.srt"
    destination = tmp_path / "video.zh-Hans.srt"
    state = tmp_path / "state"
    monkeypatch.setattr(core, "STATE_DIR", state)
    monkeypatch.setattr(core, "TOOLS", tmp_path)
    translator = FakeTranslator()

    write_srt(source, "古い字幕")
    write_srt(destination, "旧字幕")
    metadata = core.provenance_path("translation", destination)
    core.atomic_write(metadata, json.dumps({"fingerprint": core.translation_fingerprint(source)}))

    write_srt(source, "新しい字幕")
    core.translate_srt(translator, None, source, destination)

    assert core.parse_srt(destination)[0].text == "新字幕"
    assert json.loads(metadata.read_text())["fingerprint"] == core.translation_fingerprint(source)
    quarantined = list((tmp_path / "quarantine").rglob("video.zh-Hans.srt"))
    assert len(quarantined) == 1
    assert core.parse_srt(quarantined[0])[0].text == "旧字幕"


def test_quarantine_preserves_rejected_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(core, "TOOLS", tmp_path)
    output = tmp_path / "video.ja.srt"
    write_srt(output, "古い字幕")

    destination = core.quarantine_output(output, "invalid transcription")

    assert destination is not None
    assert not output.exists()
    assert destination.read_text(encoding="utf-8").endswith("古い字幕\n")
