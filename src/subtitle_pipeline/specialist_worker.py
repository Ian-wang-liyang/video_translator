"""Isolated CPU worker for bounded Japanese specialist ASR retries."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 3:
        raise SystemExit("usage: specialist_worker MODEL AUDIO CLIPS_JSON")
    model_path, audio, clips_json = args
    clips = json.loads(clips_json)
    if not isinstance(clips, list) or not all(isinstance(clip, str) for clip in clips):
        raise ValueError("CLIPS_JSON must be a list of timestamp strings")

    from faster_whisper import WhisperModel

    # This official CTranslate2 model is a float32 conversion. Runtime int8
    # conversion access-violates on Windows, so preserve its native format.
    model = WhisperModel(model_path, device="cpu", compute_type="float32")
    results: list[list[dict]] = []
    for clip in clips:
        segments, _ = model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            no_speech_threshold=None,
            log_prob_threshold=None,
            # This older official CTranslate2 conversion access-violates on
            # Windows when cross-attention word timestamps are requested.
            word_timestamps=False,
            clip_timestamps=clip,
        )
        results.append([asdict(segment) for segment in segments])
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
