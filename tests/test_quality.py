import json
import wave
from array import array
from pathlib import Path

import pytest

from subtitle_pipeline import core
from subtitle_pipeline.core import (
    Cue,
    cue_text_quality_error,
    filter_repetition_bursts,
    segments_to_srt,
    transcript_quality_errors,
)


def test_filters_repetition_burst():
    segments = [
        {"start": i, "end": i + 1, "text": text}
        for i, text in enumerate(["時間時間時間", "時間時間", "時間", "時間", "普通の会話"])
    ]
    assert [item["text"] for item in filter_repetition_bursts(segments)] == ["普通の会話"]


def test_blank_segments_cannot_split_repetition_burst():
    segments = [
        {"start": 0, "end": 1, "text": "繰り返し"},
        {"start": 1, "end": 2, "text": ""},
        {"start": 2, "end": 3, "text": "繰り返し"},
        {"start": 3, "end": 4, "text": "   "},
        {"start": 4, "end": 5, "text": "繰り返し"},
        {"start": 5, "end": 6, "text": "繰り返し"},
        {"start": 6, "end": 7, "text": "通常の会話"},
    ]

    assert [item["text"] for item in filter_repetition_bursts(segments)] == ["通常の会話"]


def test_final_whitespace_normalization_cannot_conceal_repetition_burst():
    segments = [
        {"start": index, "end": index + 1, "text": "同じ  台詞" if index % 2 else "同じ 台詞"}
        for index in range(5)
    ]
    segments.append({"start": 5, "end": 6, "text": "次の台詞"})

    rendered = segments_to_srt(segments)

    assert "同じ 台詞" not in rendered
    assert "次の台詞" in rendered


def test_rejects_latin_garbage_in_japanese():
    assert cue_text_quality_error("さあげー Soci truth")
    assert cue_text_quality_error("普通の日本語です") is None


def test_empty_transcript_fails():
    assert transcript_quality_errors([]) == ["transcript contains zero cues"]


def test_intentionally_empty_chunk_checkpoint_is_valid(tmp_path: Path):
    checkpoint = tmp_path / "chunk.srt"
    checkpoint.write_text("", encoding="utf-8")

    assert core.chunk_checkpoint_quality_errors(checkpoint, []) == []


def test_nonempty_chunk_checkpoint_still_requires_parsed_cues(tmp_path: Path):
    checkpoint = tmp_path / "chunk.srt"
    checkpoint.write_text("not an srt\n", encoding="utf-8")

    assert core.chunk_checkpoint_quality_errors(checkpoint, []) == [
        "transcript contains zero cues"
    ]


def test_chunk_transcription_uses_independent_short_windows(tmp_path: Path, monkeypatch):
    class WindowedTranscriber:
        def __init__(self):
            self.clips = []

        def transcribe(
            self,
            audio: str,
            *,
            clip_timestamps: str = "0",
            rescue: bool = False,
            specialist: bool = False,
        ) -> dict:
            self.clips.append(clip_timestamps)
            start = float(clip_timestamps.split(",")[0])
            return {
                "segments": [
                    {"start": start, "end": start + 1, "text": f"鍥炲窞{int(start)}"}
                ]
            }

    transcriber = WindowedTranscriber()
    monkeypatch.setattr(core, "_TRANSCRIBER", transcriber)
    monkeypatch.setattr(core, "filter_foreground_segments", lambda path, segments: segments)
    monkeypatch.setattr(core, "pcm_activity_intervals", lambda path, threshold: ([], 0.0))
    monkeypatch.setattr(core, "rescue_intervals", lambda activity, segments, duration: [])

    segments = core.transcribe_chunk_segments(tmp_path / "chunk.wav", 65)

    assert transcriber.clips == ["0.0,30.0", "28.0,58.0", "56.0,65"]
    assert len(segments) == 1


def test_decode_window_ownership_has_overlap_without_gaps():
    assert core.decode_windows(65) == [
        (0.0, 30.0, 0.0, 29.0),
        (28.0, 58.0, 29.0, 57.0),
        (56.0, 65, 57.0, 65),
    ]


def test_source_chunks_overlap_but_ownership_has_no_gaps():
    assert core.source_chunk_window(0, 650) == (0.0, 301.0, 0.0, 300.0)
    assert core.source_chunk_window(1, 650) == (299.0, 302.0, 1.0, 301.0)
    assert core.source_chunk_window(2, 650) == (599.0, 51.0, 1.0, 51.0)


def test_source_chunk_midpoint_ownership_emits_boundary_cue_once():
    first_view = {"start": 299.5, "end": 300.5, "text": "boundary"}
    second_view = {"start": 0.5, "end": 1.5, "text": "boundary"}

    first = core.assign_source_chunk_ownership([first_view], 0, 0, 300)
    second = core.assign_source_chunk_ownership([second_view], 299, 1, 301)

    assert first == []
    assert second == [{"start": 299.5, "end": 300.5, "text": "boundary"}]


def test_pcm_activity_finds_loud_interval_and_clipping(tmp_path: Path):
    audio = tmp_path / "activity.wav"
    samples = array("h", [100] * 8_000 + [32767] * 8_000)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())

    intervals, clipping_ratio = core.pcm_activity_intervals(audio, -28)

    assert intervals == [(0.5, 1.0)]
    assert clipping_ratio == 0.5


def test_rescue_requires_agreement_or_strong_single_model():
    primary = [{"start": 1, "end": 2, "text": "早く来て", "avg_logprob": -0.7}]
    specialist = [{"start": 1, "end": 2, "text": "早く来て", "avg_logprob": -0.8}]
    assert core.choose_rescue(primary, specialist) == primary

    disagreement = [{"start": 1, "end": 2, "text": "全然違う", "avg_logprob": -0.2}]
    assert core.choose_rescue(primary, disagreement) == []

    strong = [{"start": 1, "end": 2, "text": "止まって", "avg_logprob": -0.4}]
    assert core.choose_rescue(strong, []) == strong

    stock = [{"start": 1, "end": 2, "text": "ありがとうございました", "avg_logprob": -0.1}]
    assert core.choose_rescue(stock, stock) == []


def test_rescue_accepts_near_agreement_only_for_loud_confident_audio():
    primary = [{"start": 1, "end": 2, "text": "あいうえおかきく", "avg_logprob": -0.4}]
    specialist = [{"start": 1, "end": 2, "text": "あいさしすせそた", "avg_logprob": -0.4}]

    assert core.choose_rescue(primary, specialist, audio_dbfs=-20) == primary
    assert core.choose_rescue(primary, specialist, audio_dbfs=-35) == []


def test_rescue_rejects_near_agreement_when_either_model_is_weak():
    primary = [{"start": 1, "end": 2, "text": "あいうえおかきく", "avg_logprob": -0.4}]
    specialist = [{"start": 1, "end": 2, "text": "あいさしすせそた", "avg_logprob": -0.7}]

    assert core.choose_rescue(primary, specialist, audio_dbfs=-20) == []


def test_merge_rescue_adds_uncovered_segment_and_replaces_weak_text():
    primary = [{"start": 1, "end": 2, "text": "違う", "avg_logprob": -1.2}]
    rescue = [
        {"start": 1, "end": 2, "text": "待って", "avg_logprob": -0.5},
        {"start": 4, "end": 5, "text": "早く", "avg_logprob": -0.6},
    ]

    assert [segment["text"] for segment in core.merge_rescue_segments(primary, rescue)] == [
        "待って",
        "早く",
    ]


def test_foreground_filter_omits_low_level_speech(tmp_path: Path):
    audio = tmp_path / "chunk.wav"
    samples = array("h", [300] * 16_000 + [10_000] * 16_000)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())
    segments = [
        {"start": 0, "end": 1, "text": "quiet"},
        {"start": 1, "end": 2, "text": "foreground"},
    ]

    assert [item["text"] for item in core.filter_foreground_segments(audio, segments, -36)] == [
        "foreground"
    ]


def test_foreground_filter_keeps_quiet_high_confidence_speech(tmp_path: Path):
    audio = tmp_path / "chunk.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(array("h", [300] * 16_000).tobytes())
    confident = {
        "start": 0,
        "end": 1,
        "text": "wanted",
        "avg_logprob": -0.4,
        "no_speech_prob": 0.1,
    }
    uncertain = {**confident, "text": "uncertain", "avg_logprob": -0.8}

    assert core.filter_foreground_segments(audio, [confident], -36) == [confident]
    assert core.filter_foreground_segments(audio, [uncertain], -36) == []


def test_adjacent_duplicate_threshold_applies_to_short_chunks():
    cues = [Cue(i + 1, f"00:00:{i:02d},000 --> 00:00:{i:02d},500", "同じ") for i in range(30)]
    assert any("adjacent" in error or "repetition" in error for error in transcript_quality_errors(cues))


class FakeTranscriber:
    def transcribe(
        self,
        audio: str,
        *,
        clip_timestamps: str = "0",
        rescue: bool = False,
        specialist: bool = False,
    ) -> dict:
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
    state = tmp_path / "state"
    monkeypatch.setattr(core, "STATE_DIR", state)
    monkeypatch.setattr(core, "_TRANSCRIBER", FakeTranscriber())
    monkeypatch.setattr(core, "decode_pcm_clip", lambda source, destination, start, duration: (300, ""))
    monkeypatch.setattr(
        core,
        "transcribe_chunk_segments",
        lambda audio, duration: [{"start": 0, "end": 1, "text": "新しい字幕"}],
    )

    full_metadata = core.provenance_path("transcription", full_output)
    core.atomic_write(full_metadata, json.dumps({"fingerprint": "old-full"}))
    core.transcribe_one(video, sample_output, clip="0,300")

    assert sample_output.is_file()
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


def test_full_validation_rejects_stale_provenance(tmp_path: Path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    japanese = tmp_path / "video.ja.srt"
    chinese = tmp_path / "video.zh-Hans.srt"
    write_srt(japanese, "日本語です")
    write_srt(chinese, "这是日语")
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(core, "STATE_DIR", state)
    monkeypatch.setattr(core, "REPORT_DIR", reports)
    monkeypatch.setattr(core, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(core, "video_duration_ms", lambda _: 1_000)
    core.atomic_write(
        core.provenance_path("transcription", japanese),
        json.dumps({"fingerprint": core.transcription_fingerprint(video, japanese, "0")}),
    )
    core.atomic_write(
        core.provenance_path("translation", chinese),
        json.dumps({"fingerprint": "stale"}),
    )

    with pytest.raises(RuntimeError, match="Validation found 1 failure"):
        core.validate()

    assert "stale translation provenance" in (reports / "validation.tsv").read_text(encoding="utf-8")
