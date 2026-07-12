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
