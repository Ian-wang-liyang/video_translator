#!/usr/bin/env python3
"""Resumable, fully local Japanese -> Simplified Chinese subtitle pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import unicodedata
import wave
import zlib
from array import array
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .backends import load_transcriber, load_translation_backend
from .config import load_settings

SETTINGS = load_settings()
ROOT = SETTINGS.root
TOOLS = SETTINGS.runtime_dir
VIDEO_DIR = SETTINGS.video_dir
WHISPER_MODEL = SETTINGS.whisper_model
TRANSLATION_MODEL = SETTINGS.translation_model
LOG_DIR = TOOLS / "logs"
STATE_DIR = TOOLS / "state"
REPORT_DIR = TOOLS / "reports"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
TRANSCRIPTION_REVISION = "adaptive-confidence-source-overlap-v7"
TRANSLATION_REVISION = "contextual-batched-v2"
TRANSLATION_PROMPT_REVISION = "ja-zh-hans-v2"
CHUNK_SECONDS = SETTINGS.chunk_seconds
_TRANSCRIBER = None
JSON_OUTPUT = False
TIMING_RE = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> "
    r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})$"
)
RESCUE_SILENCE_HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ありがとうございました",
    "お疲れ様でした",
}


@dataclass
class Cue:
    index: int
    timing: str
    text: str


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    if JSON_OUTPUT:
        print(json.dumps({"event": "log", "timestamp": stamp, "message": message}, ensure_ascii=False), flush=True)
    else:
        print(line, flush=True)
    with (LOG_DIR / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with (LOG_DIR / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "log", "timestamp": stamp, "message": message}, ensure_ascii=False) + "\n")


def set_json_output(enabled: bool) -> None:
    global JSON_OUTPUT
    JSON_OUTPUT = enabled


def videos() -> list[Path]:
    if not VIDEO_DIR.is_dir():
        raise RuntimeError(f"Video directory does not exist: {VIDEO_DIR}")
    return sorted(
        (p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: unicodedata.normalize("NFKC", p.name).casefold(),
    )


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def quarantine_output(path: Path, reason: str) -> Path | None:
    """Preserve a rejected generated output before replacing it."""
    if not path.exists():
        return None
    safe_reason = re.sub(r"[^a-z0-9-]+", "-", reason.casefold()).strip("-") or "rejected"
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    destination_dir = TOOLS / "quarantine" / f"{stamp}-{safe_reason}"
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination = destination_dir / path.name
    path.replace(destination)
    log(f"QUARANTINE {path.name}: {destination}")
    return destination


def provenance_path(stage: str, output: Path) -> Path:
    """Return an output-specific provenance record path."""
    output_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    return STATE_DIR / stage / "outputs" / f"{output_key}.json"


def transcription_fingerprint(video: Path, output: Path, clip: str) -> str:
    material = (
        f"{video.resolve()}|{video.stat().st_size}|{video.stat().st_mtime_ns}|"
        f"{output.resolve()}|{clip}|{TRANSCRIPTION_REVISION}|{SETTINGS.transcription_backend}|"
        f"{SETTINGS.device}|{WHISPER_MODEL}|{SETTINGS.specialist_model}|{SETTINGS.chunk_seconds}|"
        f"{SETTINGS.source_chunk_overlap_seconds}|{SETTINGS.decode_window_seconds}|"
        f"{SETTINGS.window_overlap_seconds}|{SETTINGS.vad_threshold}|"
        f"{SETTINGS.foreground_min_dbfs}|{SETTINGS.foreground_confident_min_dbfs}|"
        f"{SETTINGS.foreground_confident_logprob}|"
        f"{SETTINGS.foreground_confident_no_speech_prob}|{SETTINGS.rescue_activity_dbfs}|"
        f"{SETTINGS.rescue_flag_logprob}|{SETTINGS.rescue_accept_logprob}|"
        f"{SETTINGS.rescue_agreement_threshold}|"
        f"{SETTINGS.rescue_conditional_agreement_threshold}"
    )
    return hashlib.sha256(material.encode()).hexdigest()


def translation_fingerprint(source: Path) -> str:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    material = (
        f"{source.resolve()}|{source_digest}|{TRANSLATION_REVISION}|{TRANSLATION_PROMPT_REVISION}|"
        f"{SETTINGS.translation_backend}|{SETTINGS.device}|{TRANSLATION_MODEL}"
    )
    return hashlib.sha256(material.encode()).hexdigest()


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    candidates: list[tuple[float, float, str]] = []
    common_silence_hallucinations = {
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "ありがとうございました",
        "字幕をご覧いただきありがとうございます",
    }
    for segment in segments:
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        # Whisper commonly emits these phrases across a full 30-second window
        # containing silence, music, or non-speech audio. Only filter unusually
        # long instances so genuine short spoken thanks remain intact.
        normalized = re.sub(r"[。.!！?？\s]", "", text)
        if normalized in common_silence_hallucinations and end - start >= 10:
            log(f"FILTER likely silence hallucination at {srt_timestamp(start)}: {text}")
            continue
        text_error = cue_text_quality_error(text)
        if text_error:
            log(f"FILTER repetitive hallucination at {srt_timestamp(start)}: {text_error}")
            continue
        if not text or end <= start:
            continue
        candidates.append((start, end, text))
    kept: list[tuple[float, float, str]] = []
    position = 0
    while position < len(candidates):
        end = position + 1
        while end < len(candidates) and candidates[end][2] == candidates[position][2]:
            end += 1
        if end - position >= 4:
            log(
                f"FILTER normalized repetition burst ({end - position} cues) at "
                f"{srt_timestamp(candidates[position][0])}: {candidates[position][2]!r}"
            )
        else:
            kept.extend(candidates[position:end])
        position = end
    blocks = [
        f"{number}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}"
        for number, (start, end, text) in enumerate(kept, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    if not content:
        return []
    cues: list[Cue] = []
    for block in re.split(r"\n{2,}", content):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Malformed SRT block in {path.name}: {block[:80]!r}")
        index = int(lines[0].strip())
        timing = lines[1].strip()
        if not TIMING_RE.match(timing):
            raise ValueError(f"Malformed timing in {path.name}: {timing!r}")
        cues.append(Cue(index=index, timing=timing, text="\n".join(lines[2:]).strip()))
    return cues


def cues_to_srt(cues: list[Cue]) -> str:
    return "\n\n".join(
        f"{cue.index}\n{cue.timing}\n{cue.text.strip()}" for cue in cues
    ) + ("\n" if cues else "")


def cue_text_quality_error(text: str) -> str | None:
    """Detect repetition trapped inside one Whisper decoding window."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return "empty cue"
    longest_character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(.)\1+", compact)),
        default=1,
    )
    if longest_character_run >= 12:
        return f"single character repeated {longest_character_run} times"
    latin_words = re.findall(r"[A-Za-z]{4,}", compact)
    if latin_words:
        return f"unexpected Latin text in Japanese transcript: {latin_words[0]!r}"
    encoded = compact.encode("utf-8")
    compression_ratio = len(encoded) / max(1, len(zlib.compress(encoded)))
    if len(compact) >= 40 and compression_ratio > 2.8:
        return f"highly repetitive text (compression ratio {compression_ratio:.1f})"
    if len(compact) > 120:
        return f"implausibly long cue ({len(compact)} characters)"
    return None


def repetition_root(text: str) -> str:
    compact = re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]", "", text).casefold()
    for width in range(1, min(8, len(compact)) + 1):
        if len(compact) % width == 0 and compact == compact[:width] * (len(compact) // width):
            return compact[:width]
    return compact


def filter_repetition_bursts(segments: list[dict]) -> list[dict]:
    """Drop runs of short repeated vocalizations that Whisper renders as words."""
    # Empty decode windows are not cues and must not split an otherwise exact
    # repetition run into smaller groups that evade the burst threshold.
    segments = [segment for segment in segments if repetition_root(str(segment.get("text", "")).strip())]
    kept: list[dict] = []
    position = 0
    while position < len(segments):
        text = str(segments[position].get("text", "")).strip()
        root = repetition_root(text)
        end = position + 1
        while end < len(segments):
            other = repetition_root(str(segments[end].get("text", "")).strip())
            if not root or other != root:
                break
            end += 1
        if end - position >= 4 and len(root) <= 6:
            log(
                f"FILTER repetition burst ({end - position} cues) at "
                f"{srt_timestamp(float(segments[position].get('start', 0)))}: {root!r}"
            )
        else:
            kept.extend(segments[position:end])
        position = end
    return kept


def transcript_quality_errors(cues: list[Cue]) -> list[str]:
    """Return hard failures for characteristic Whisper repetition loops."""
    from collections import Counter

    if not cues:
        return ["transcript contains zero cues"]
    texts = [re.sub(r"\s+", " ", cue.text).strip() for cue in cues]
    errors: list[str] = []
    bad_cues = [
        (cue.index, error)
        for cue in cues
        if (error := cue_text_quality_error(cue.text)) is not None
    ]
    if bad_cues:
        preview = ", ".join(f"{index}: {error}" for index, error in bad_cues[:5])
        errors.append(f"{len(bad_cues)} internally repetitive cue(s): {preview}")
    longest_run = run = 1
    for previous, current in zip(texts, texts[1:]):
        run = run + 1 if current == previous else 1
        longest_run = max(longest_run, run)
    counts = Counter(texts)
    dominant_text, dominant_count = counts.most_common(1)[0]
    adjacent = sum(a == b for a, b in zip(texts, texts[1:]))
    if longest_run >= 25:
        errors.append(f"repetition loop: {longest_run} consecutive identical cues")
    if len(cues) >= 30 and adjacent / (len(cues) - 1) > 0.20:
        errors.append("more than 20% adjacent cues are identical")
    if dominant_count >= 50 and dominant_count / len(cues) > 0.20:
        errors.append(f"one cue dominates transcript ({dominant_count}/{len(cues)}): {dominant_text!r}")
    punctuation_only = sum(not re.search(r"[\w\u3040-\u30ff\u4e00-\u9fff]", text) for text in texts)
    if len(cues) >= 100 and punctuation_only / len(cues) > 0.10:
        errors.append("more than 10% of cues contain punctuation only")
    return errors


def chunk_checkpoint_quality_errors(checkpoint: Path, cues: list[Cue]) -> list[str]:
    """Validate a chunk checkpoint while permitting an intentional empty result.

    A chunk can legitimately contain no accepted foreground speech after the
    primary, recovery, and hallucination filters run.  The assembled full-video
    transcript is still required to contain cues.
    """
    if not checkpoint.read_text(encoding="utf-8").strip():
        return []
    return transcript_quality_errors(cues)


def filter_foreground_segments(
    audio_chunk: Path, segments: list[dict], minimum_dbfs: float | None = None
) -> list[dict]:
    """Drop quiet speech unless both acoustic and ASR confidence are strong."""
    threshold = SETTINGS.foreground_min_dbfs if minimum_dbfs is None else minimum_dbfs
    with wave.open(str(audio_chunk), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"foreground filtering requires mono 16-bit PCM: {audio_chunk}")
        sample_rate = handle.getframerate()
        samples = array("h", handle.readframes(handle.getnframes()))
    kept: list[dict] = []
    for segment in segments:
        start = max(0, round(float(segment.get("start", 0)) * sample_rate))
        end = min(len(samples), round(float(segment.get("end", 0)) * sample_rate))
        if end <= start:
            continue
        window = samples[start:end]
        rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        dbfs = 20 * math.log10(max(rms, 1) / 32768)
        if dbfs < threshold:
            confidently_spoken = (
                dbfs >= SETTINGS.foreground_confident_min_dbfs
                and segment_logprob(segment) >= SETTINGS.foreground_confident_logprob
                and segment_no_speech_prob(segment)
                <= SETTINGS.foreground_confident_no_speech_prob
            )
            if confidently_spoken:
                log(
                    f"KEEP quiet high-confidence speech at "
                    f"{srt_timestamp(float(segment.get('start', 0)))}: "
                    f"{dbfs:.1f} dBFS, logprob {segment_logprob(segment):.2f}"
                )
                kept.append(segment)
                continue
            log(
                f"FILTER low-level speech at {srt_timestamp(float(segment.get('start', 0)))}: "
                f"{dbfs:.1f} dBFS < {threshold:.1f} dBFS"
            )
            continue
        kept.append(segment)
    return kept


def decode_windows(duration: float) -> list[tuple[float, float, float, float]]:
    """Return overlapping decode and non-overlapping ownership intervals."""
    window = SETTINGS.decode_window_seconds
    overlap = SETTINGS.window_overlap_seconds
    step = window - overlap
    starts: list[float] = []
    start = 0.0
    while start < duration:
        starts.append(start)
        start += step
    windows: list[tuple[float, float, float, float]] = []
    for index, start in enumerate(starts):
        end = min(duration, start + window)
        owned_start = start if index == 0 else start + overlap / 2
        owned_end = end if index == len(starts) - 1 else end - overlap / 2
        windows.append((start, end, owned_start, owned_end))
    return windows


def segment_logprob(segment: dict) -> float:
    try:
        return float(segment.get("avg_logprob", -99.0))
    except (TypeError, ValueError):
        return -99.0


def segment_no_speech_prob(segment: dict) -> float:
    try:
        return float(segment.get("no_speech_prob", 1.0))
    except (TypeError, ValueError):
        return 1.0


def interval_dbfs_levels(
    audio_chunk: Path, intervals: list[tuple[float, float]]
) -> list[float]:
    """Return RMS dBFS for several intervals with a single PCM read."""
    with wave.open(str(audio_chunk), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"level analysis requires mono 16-bit PCM: {audio_chunk}")
        sample_rate = handle.getframerate()
        samples = array("h", handle.readframes(handle.getnframes()))
    levels: list[float] = []
    for start, end in intervals:
        left = max(0, round(start * sample_rate))
        right = min(len(samples), round(end * sample_rate))
        window = samples[left:right]
        if not window:
            levels.append(-120.0)
            continue
        rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        levels.append(20 * math.log10(max(rms, 1) / 32768))
    return levels


def pcm_activity_intervals(audio_chunk: Path, threshold: float) -> tuple[list[tuple[float, float]], float]:
    """Find sustained high-energy intervals and return their PCM clipping ratio."""
    with wave.open(str(audio_chunk), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"activity analysis requires mono 16-bit PCM: {audio_chunk}")
        sample_rate = handle.getframerate()
        samples = array("h", handle.readframes(handle.getnframes()))
    if not samples:
        return [], 0.0
    clipping_ratio = sum(abs(sample) >= 32700 for sample in samples) / len(samples)
    frame_size = max(1, round(sample_rate * 0.1))
    active_frames: list[tuple[float, float]] = []
    for position in range(0, len(samples), frame_size):
        frame = samples[position : position + frame_size]
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        dbfs = 20 * math.log10(max(rms, 1) / 32768)
        if dbfs >= threshold:
            active_frames.append((position / sample_rate, min(len(samples), position + len(frame)) / sample_rate))
    intervals: list[tuple[float, float]] = []
    for start, end in active_frames:
        if intervals and start - intervals[-1][1] <= 0.2:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    return [(start, end) for start, end in intervals if end - start >= 0.2], clipping_ratio


def interval_coverage(start: float, end: float, segments: list[dict]) -> float:
    covered = sum(
        max(0.0, min(end, float(segment.get("end", 0))) - max(start, float(segment.get("start", 0))))
        for segment in segments
    )
    return min(1.0, covered / max(0.001, end - start))


def rescue_intervals(
    activity: list[tuple[float, float]], segments: list[dict], duration: float
) -> list[tuple[float, float]]:
    """Select uncovered loud audio and low-confidence recognized speech for retry."""
    selected = [(start, end) for start, end in activity if interval_coverage(start, end, segments) < 0.2]
    selected.extend(
        (float(segment.get("start", 0)), float(segment.get("end", 0)))
        for segment in segments
        if segment_logprob(segment) < SETTINGS.rescue_flag_logprob
    )
    padded = sorted((max(0.0, start - 1.0), min(duration, end + 1.0)) for start, end in selected)
    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= 0.5 and end - merged[-1][0] <= 15.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            cursor = start
            while end - cursor > 15.0:
                merged.append((cursor, cursor + 15.0))
                cursor += 14.0
            merged.append((cursor, end))
    return [(start, end) for start, end in merged if end - start >= 0.2]


def normalized_segment_text(segments: list[dict]) -> str:
    return re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]", "", "".join(str(s.get("text", "")) for s in segments))


def valid_rescue_segments(segments: list[dict]) -> list[dict]:
    return [
        segment
        for segment in segments
        if str(segment.get("text", "")).strip()
        and cue_text_quality_error(str(segment.get("text", ""))) is None
        and re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]", "", str(segment.get("text", "")))
        not in RESCUE_SILENCE_HALLUCINATIONS
        and segment_logprob(segment) >= SETTINGS.rescue_accept_logprob
    ]


def choose_rescue(
    primary: list[dict], specialist: list[dict], audio_dbfs: float | None = None
) -> list[dict]:
    """Require cross-model agreement unless one model is independently strong."""
    primary = valid_rescue_segments(primary)
    specialist = valid_rescue_segments(specialist)
    primary_text = normalized_segment_text(primary)
    specialist_text = normalized_segment_text(specialist)
    if primary_text and specialist_text:
        agreement = SequenceMatcher(None, primary_text, specialist_text).ratio()
        if agreement >= SETTINGS.rescue_agreement_threshold:
            return primary
        strong_models = all(
            segment_logprob(segment) >= SETTINGS.foreground_confident_logprob
            for segment in [*primary, *specialist]
        )
        if (
            agreement >= SETTINGS.rescue_conditional_agreement_threshold
            and audio_dbfs is not None
            and audio_dbfs >= SETTINGS.rescue_activity_dbfs
            and strong_models
        ):
            log(
                f"RESCUE accepting near-agreement {agreement:.2f} on loud "
                f"high-confidence audio ({audio_dbfs:.1f} dBFS)"
            )
            return primary
        log(
            f"REJECT rescue candidate: specialist agreement {agreement:.2f} < "
            f"{SETTINGS.rescue_agreement_threshold:.2f}"
        )
        return []
    candidate = primary or specialist
    if candidate and min(segment_logprob(segment) for segment in candidate) >= -0.55:
        return candidate
    return []


def merge_rescue_segments(primary: list[dict], rescue: list[dict]) -> list[dict]:
    merged = list(primary)
    for candidate in rescue:
        start = float(candidate.get("start", 0))
        end = float(candidate.get("end", 0))
        matches = [
            index
            for index, existing in enumerate(merged)
            if max(0.0, min(end, float(existing.get("end", 0))) - max(start, float(existing.get("start", 0))))
            >= 0.3 * max(0.1, min(end - start, float(existing.get("end", 0)) - float(existing.get("start", 0))))
        ]
        if not matches:
            merged.append(candidate)
        elif len(matches) == 1:
            existing = merged[matches[0]]
            if (
                segment_logprob(existing) < SETTINGS.rescue_flag_logprob
                and segment_logprob(candidate) > segment_logprob(existing) + 0.1
            ):
                merged[matches[0]] = candidate
    return sorted(merged, key=lambda segment: (float(segment.get("start", 0)), float(segment.get("end", 0))))


def prepare_rescue_audio(audio_chunk: Path, clipping_ratio: float) -> Path:
    if clipping_ratio < 0.0005:
        return audio_chunk
    repaired = audio_chunk.with_name(audio_chunk.stem + "-rescue.wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio_chunk),
            "-af", "adeclip", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(repaired),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        repaired.unlink(missing_ok=True)
        log("WARN clipping repair failed; rescue will use original PCM")
        return audio_chunk
    log(f"RESCUE using de-clipped PCM ({clipping_ratio:.3%} clipped samples)")
    return repaired


def decode_pcm_clip(source: Path, destination: Path, start: float, duration: float) -> tuple[float, str]:
    """Decode one bounded mono PCM clip and return actual duration plus warnings."""
    decode = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-ss", str(start), "-i", str(source), "-t", str(duration),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if decode.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"FFmpeg failed decoding audio at {start:.3f}s with exit code {decode.returncode}")
    duration_result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(duration_result.stdout.strip()), decode.stderr.strip()


def source_chunk_window(number: int, video_duration: float) -> tuple[float, float, float, float]:
    """Return overlapped source decode and relative non-overlapping ownership bounds."""
    nominal_start = number * CHUNK_SECONDS
    nominal_end = min(video_duration, nominal_start + CHUNK_SECONDS)
    overlap = SETTINGS.source_chunk_overlap_seconds
    decode_start = max(0.0, nominal_start - overlap)
    decode_end = min(video_duration, nominal_end + overlap)
    return (
        decode_start,
        decode_end - decode_start,
        nominal_start - decode_start,
        nominal_end - decode_start,
    )


def assign_source_chunk_ownership(
    segments: list[dict], decode_start: float, owned_start: float, owned_end: float
) -> list[dict]:
    """Keep only midpoint-owned cues and convert their times to video-relative values."""
    adjusted: list[dict] = []
    for source in segments:
        midpoint = (float(source.get("start", 0)) + float(source.get("end", 0))) / 2
        if not owned_start <= midpoint < owned_end:
            continue
        segment = dict(source)
        segment["start"] = float(segment.get("start", 0)) + decode_start
        segment["end"] = float(segment.get("end", 0)) + decode_start
        adjusted.append(segment)
    return adjusted


def transcribe_chunk_segments(audio_chunk: Path, duration: float) -> list[dict]:
    """Decode overlap-owned windows, then rescue uncovered or uncertain loud speech."""
    primary: list[dict] = []
    for start, end, owned_start, owned_end in decode_windows(duration):
        result = _TRANSCRIBER.transcribe(
            str(audio_chunk), clip_timestamps=f"{start},{end}"
        )
        primary.extend(
            segment
            for segment in result.get("segments", [])
            if owned_start <= (float(segment.get("start", 0)) + float(segment.get("end", 0))) / 2 < owned_end
        )
    primary = filter_foreground_segments(audio_chunk, primary)
    activity, clipping_ratio = pcm_activity_intervals(audio_chunk, SETTINGS.rescue_activity_dbfs)
    intervals = rescue_intervals(activity, primary, duration)
    if not intervals:
        return filter_repetition_bursts(primary)
    log(f"RESCUE reviewing {len(intervals)} loud-gap/low-confidence interval(s)")
    rescue_audio = prepare_rescue_audio(audio_chunk, clipping_ratio)
    recovered: list[dict] = []
    try:
        clips = [f"{start},{end}" for start, end in intervals]
        primary_retries = [
            _TRANSCRIBER.transcribe(
                str(rescue_audio), clip_timestamps=clip, rescue=True
            ).get("segments", [])
            for clip in clips
        ]
        specialist_retries = (
            _TRANSCRIBER.transcribe_specialist_batch(str(rescue_audio), clips)
            if SETTINGS.specialist_model is not None
            else [[] for _ in clips]
        )
        interval_levels = interval_dbfs_levels(audio_chunk, intervals)
        for primary_retry, specialist_retry, audio_dbfs in zip(
            primary_retries, specialist_retries, interval_levels, strict=True
        ):
            chosen = choose_rescue(primary_retry, specialist_retry, audio_dbfs)
            recovered.extend(filter_foreground_segments(audio_chunk, chosen))
    finally:
        if rescue_audio != audio_chunk:
            rescue_audio.unlink(missing_ok=True)
    if recovered:
        log(f"RESCUE accepted {len(recovered)} segment(s)")
    return filter_repetition_bursts(merge_rescue_segments(primary, recovered))


def transcribe_one(video: Path, output: Path, clip: str = "0") -> None:
    fingerprint = transcription_fingerprint(video, output, clip)
    metadata = provenance_path("transcription", output)
    if output.exists():
        try:
            existing = parse_srt(output)
            errors = transcript_quality_errors(existing)
        except Exception as exc:
            errors = [str(exc)]
        try:
            current = json.loads(metadata.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
        except Exception:
            current = False
        if not errors and current:
            log(f"SKIP transcription (quality-validated output exists): {output.name}")
            return
        if not current:
            errors.append("missing or stale transcription fingerprint")
        log(f"REBUILD invalid transcription {output.name}: {'; '.join(errors)}")
        quarantine_output(output, "invalid-transcription")
    global _TRANSCRIBER
    if _TRANSCRIBER is None:
        _TRANSCRIBER = load_transcriber(SETTINGS)

    log(f"START transcription: {video.name}")
    started = time.monotonic()
    if clip != "0":
        clip_start, clip_end = (float(value) for value in clip.split(",", 1))
        if clip_end <= clip_start:
            raise ValueError("transcription clip end must be after its start")
        audio_dir = TOOLS / "cache" / "audio" / fingerprint
        audio_chunk = audio_dir / "sample.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            clip_duration, warnings = decode_pcm_clip(video, audio_chunk, clip_start, clip_end - clip_start)
            if warnings:
                report = REPORT_DIR / "decode-warnings" / f"{fingerprint}.log"
                atomic_write(report, warnings + "\n")
                log(f"WARN source decode issues recorded: {report}")
            adjusted = []
            for segment in transcribe_chunk_segments(audio_chunk, clip_duration):
                segment = dict(segment)
                segment["start"] = float(segment.get("start", 0)) + clip_start
                segment["end"] = float(segment.get("end", 0)) + clip_start
                adjusted.append(segment)
            rendered = segments_to_srt(adjusted)
        finally:
            shutil.rmtree(audio_dir, ignore_errors=True)
    else:
        checkpoint_dir = STATE_DIR / "transcription" / fingerprint
        audio_dir = TOOLS / "cache" / "audio" / fingerprint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        decode_warnings: list[str] = []
        warning_report = REPORT_DIR / "decode-warnings" / f"{fingerprint}.log"
        try:
            blocks: list[str] = []
            cue_number = 1
            video_duration = video_duration_ms(video) / 1000
            chunk_count = max(1, math.ceil(video_duration / CHUNK_SECONDS))
            for number in range(chunk_count):
                offset, expected_duration, owned_start, owned_end = source_chunk_window(
                    number, video_duration
                )
                audio_chunk = audio_dir / f"chunk-{number:04d}.wav"
                checkpoint = checkpoint_dir / f"chunk-{number:04d}.srt"
                if checkpoint.exists():
                    chunk_cues = parse_srt(checkpoint)
                    chunk_errors = chunk_checkpoint_quality_errors(checkpoint, chunk_cues)
                    if chunk_errors:
                        checkpoint.unlink()
                        chunk_cues = []
                        checkpoint_valid = False
                    else:
                        checkpoint_valid = True
                else:
                    chunk_cues = []
                    checkpoint_valid = False
                if not checkpoint_valid:
                    try:
                        chunk_duration, warnings = decode_pcm_clip(
                            video, audio_chunk, offset, expected_duration
                        )
                        if warnings:
                            decode_warnings.append(
                                f"chunk {number + 1}/{chunk_count} at {offset:.3f}s\n{warnings}"
                            )
                        adjusted = assign_source_chunk_ownership(
                            transcribe_chunk_segments(audio_chunk, chunk_duration),
                            offset,
                            owned_start,
                            owned_end,
                        )
                        chunk_text = segments_to_srt(adjusted)
                        atomic_write(checkpoint, chunk_text)
                        chunk_cues = parse_srt(checkpoint)
                        chunk_errors = chunk_checkpoint_quality_errors(checkpoint, chunk_cues)
                        if chunk_errors:
                            raise RuntimeError(
                                f"chunk {number + 1}/{chunk_count} failed quality validation: "
                                + "; ".join(chunk_errors)
                            )
                        if not chunk_cues:
                            log(
                                f"CHECKPOINT no accepted foreground speech "
                                f"for chunk {number + 1}/{chunk_count}"
                            )
                    finally:
                        audio_chunk.unlink(missing_ok=True)
                for cue in chunk_cues:
                    blocks.append(f"{cue_number}\n{cue.timing}\n{cue.text}")
                    cue_number += 1
                log(f"Transcription progress {video.name}: chunk {number + 1}/{chunk_count}")
            rendered = "\n\n".join(blocks) + ("\n" if blocks else "")
        finally:
            shutil.rmtree(audio_dir, ignore_errors=True)
            if decode_warnings:
                stream_probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index,codec_name", "-of", "json", str(video),
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
                try:
                    audio_streams = json.loads(stream_probe.stdout).get("streams", [])
                except json.JSONDecodeError:
                    audio_streams = []
                decode_warnings.insert(
                    0,
                    "audio streams: " + json.dumps(audio_streams, ensure_ascii=False, sort_keys=True),
                )
                atomic_write(warning_report, "\n\n".join(decode_warnings) + "\n")
                log(f"WARN source decode issues recorded: {warning_report}")
    if not rendered.strip():
        raise RuntimeError(f"Whisper produced no subtitle cues for {video.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    errors = transcript_quality_errors(parse_srt(temporary))
    if errors:
        raise RuntimeError("transcript quality validation failed: " + "; ".join(errors))
    temporary.replace(output)
    atomic_write(metadata, json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    elapsed = time.monotonic() - started
    log(f"DONE transcription ({elapsed / 60:.1f} min): {output.name}")


def transcribe_folder() -> None:
    failures: list[str] = []
    for video in videos():
        try:
            transcribe_one(video, video.with_suffix(".ja.srt"))
        except Exception as exc:  # continue the unattended batch
            failures.append(video.name)
            log(f"ERROR transcription {video.name}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
    atomic_write(REPORT_DIR / "transcription-failures.txt", "\n".join(failures) + ("\n" if failures else ""))
    if failures:
        raise RuntimeError(f"Transcription failed for {len(failures)} video(s)")


def load_translator():
    log("Loading local Qwen translation model")
    return load_translation_backend(SETTINGS), None


def generate_response(model, tokenizer, instruction: str, max_tokens: int) -> str:
    return model.generate(instruction, max_tokens)


def parse_numbered_response(response: str, count: int) -> list[str] | None:
    found: dict[int, str] = {}
    pattern = re.compile(r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)")
    for match in pattern.finditer(response):
        number = int(match.group(1))
        value = match.group(2).strip()
        if 1 <= number <= count and value:
            found[number] = value
    if len(found) != count:
        return None
    return [found[number] for number in range(1, count + 1)]


def translate_group(
    model,
    tokenizer,
    texts: list[str],
    *,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
) -> list[str]:
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, 1))
    context_lines = [
        *(f"Previous context: {text}" for text in (context_before or [])),
        *(f"Following context: {text}" for text in (context_after or [])),
    ]
    context = "\n".join(context_lines) or "(no surrounding context)"
    instruction = (
        "Translate these Japanese subtitle cues into natural, concise Simplified Chinese.\n"
        "Use the surrounding dialogue only to resolve meaning, names, pronouns, and sentence fragments. "
        "Do not translate or return the context lines.\n"
        "Preserve names, episode/catalogue codes, numbers, and meaning. Render Japanese names "
        "in suitable Chinese characters or Chinese phonetic transcription. Do not censor or summarize.\n"
        "Return exactly one translated item for every input using the same [number] markers.\n"
        "Use Chinese script only: do not leave any Japanese hiragana or katakana. Do not include "
        "Japanese originals, explanations, markdown, or timestamps.\n"
        "If an utterance is genuinely too unclear to translate, return （日语发音不清） for that item.\n\n"
        f"Surrounding dialogue (context only):\n{context}\n\n"
        f"Numbered cues to translate:\n{numbered}"
    )
    for attempt in range(3):
        response = generate_response(model, tokenizer, instruction, max_tokens=max(256, len(texts) * 100))
        parsed = parse_numbered_response(response, len(texts))
        if parsed is not None and not any(re.search(r"[\u3040-\u30ff]", value) for value in parsed):
            return parsed
        # Small models sometimes omit the marker for a single item even when
        # the translation itself is valid and contains only Chinese script.
        if len(texts) == 1:
            unmarked = re.sub(r"^\s*\[1\]\s*", "", response).strip()
            if unmarked and not re.search(r"[\u3040-\u30ff]", unmarked):
                return [unmarked]
        log(f"WARN malformed or Japanese-script translation response; retry {attempt + 1}/3")
    if len(texts) > 1:
        output: list[str] = []
        for index, text in enumerate(texts):
            output.extend(
                translate_group(
                    model,
                    tokenizer,
                    [text],
                    context_before=[*(context_before or []), *texts[:index]],
                    context_after=[*texts[index + 1 :], *(context_after or [])],
                )
            )
        return output
    log(f"WARN using unclear-speech fallback for cue: {texts[0]!r}")
    return ["（日语发音不清）"]


def translate_srt(model, tokenizer, source: Path, destination: Path) -> None:
    source_cues = parse_srt(source)
    if not source_cues:
        raise ValueError(f"Japanese source contains no cues: {source.name}")
    quality_errors = transcript_quality_errors(source_cues)
    if quality_errors:
        raise ValueError(f"Japanese source failed quality validation: {'; '.join(quality_errors)}")
    fingerprint = translation_fingerprint(source)
    metadata = provenance_path("translation", destination)
    if destination.exists():
        try:
            existing = parse_srt(destination)
            aligned = len(existing) == len(source_cues) and all(
                translated.index == original.index
                and translated.timing == original.timing
                and bool(translated.text.strip())
                and not re.search(r"[\u3040-\u30ff]", translated.text)
                for original, translated in zip(source_cues, existing, strict=True)
            )
        except (ValueError, TypeError):
            aligned = False
        try:
            current = json.loads(metadata.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
        except Exception:
            current = False
        if aligned and current:
            log(f"SKIP translation (complete aligned current output exists): {destination.name}")
            return
        log(f"REBUILD stale, incomplete, misaligned, or unproven translation: {destination.name}")
        quarantine_output(destination, "invalid-translation")
    translated: list[Cue] = []
    log(f"START translation: {source.name}")
    for offset in range(0, len(source_cues), 10):
        group = source_cues[offset : offset + 10]
        context_size = SETTINGS.translation_context_cues
        before = [cue.text for cue in source_cues[max(0, offset - context_size) : offset]]
        after = [
            cue.text
            for cue in source_cues[offset + len(group) : offset + len(group) + context_size]
        ]
        chinese = translate_group(
            model,
            tokenizer,
            [cue.text for cue in group],
            context_before=before,
            context_after=after,
        )
        translated.extend(
            Cue(index=cue.index, timing=cue.timing, text=text)
            for cue, text in zip(group, chinese, strict=True)
        )
        log(f"Translation progress {source.name}: {len(translated)}/{len(source_cues)} cues")
    atomic_write(destination, cues_to_srt(translated))
    atomic_write(metadata, json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    log(f"DONE translation: {destination.name}")


def process_collection(max_videos: int | None = None, selected_video: Path | None = None) -> None:
    """Finish transcription and translation for one video at a time."""
    transcription_failures: list[str] = []
    translation_failures: list[str] = []
    items = [selected_video] if selected_video is not None else videos()
    if max_videos is not None:
        items = items[:max_videos]
    for video in items:
        japanese = video.with_suffix(".ja.srt")
        chinese = video.with_suffix(".zh-Hans.srt")
        log(f"START video: {video.name}")
        try:
            transcribe_one(video, japanese)
        except Exception as exc:
            transcription_failures.append(video.name)
            log(f"ERROR transcription {video.name}: {type(exc).__name__}: {exc}")
            gc.collect()
            continue
        gc.collect()

        model = tokenizer = None
        try:
            model, tokenizer = load_translator()
            translate_srt(model, tokenizer, japanese, chinese)
            log(f"DONE video: {video.name}")
        except Exception as exc:
            translation_failures.append(video.name)
            log(f"ERROR translation {video.name}: {type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()

    atomic_write(
        REPORT_DIR / "transcription-failures.txt",
        "\n".join(transcription_failures) + ("\n" if transcription_failures else ""),
    )
    atomic_write(
        REPORT_DIR / "translation-failures.txt",
        "\n".join(translation_failures) + ("\n" if translation_failures else ""),
    )
    if transcription_failures or translation_failures:
        raise RuntimeError(
            f"Per-video processing failed: {len(transcription_failures)} transcription, "
            f"{len(translation_failures)} translation"
        )


def translate_folder() -> None:
    model, tokenizer = load_translator()
    failures: list[str] = []
    try:
        for video in videos():
            source = video.with_suffix(".ja.srt")
            destination = video.with_suffix(".zh-Hans.srt")
            if not source.exists():
                failures.append(video.name + " (missing Japanese SRT)")
                log(f"ERROR missing Japanese SRT: {source.name}")
                continue
            try:
                translate_srt(model, tokenizer, source, destination)
            except Exception as exc:
                failures.append(video.name)
                log(f"ERROR translation {video.name}: {type(exc).__name__}: {exc}")
    finally:
        del model
        gc.collect()
    atomic_write(REPORT_DIR / "translation-failures.txt", "\n".join(failures) + ("\n" if failures else ""))
    if failures:
        raise RuntimeError(f"Translation failed for {len(failures)} video(s)")


def timestamp_ms(timing_side: str) -> int:
    match = re.match(r"^(\d+):(\d{2}):(\d{2}),(\d{3})$", timing_side.strip())
    if not match:
        raise ValueError(timing_side)
    h, m, s, ms = map(int, match.groups())
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def video_duration_ms(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return round(float(result.stdout.strip()) * 1000)


def validate() -> None:
    report: list[str] = []
    failures = 0
    japanese_re = re.compile(r"[\u3040-\u30ff]")
    for video in videos():
        duration = video_duration_ms(video)
        parsed: dict[str, list[Cue]] = {}
        for language in ("ja", "zh-Hans"):
            path = video.with_suffix(f".{language}.srt")
            if not path.exists():
                report.append(f"FAIL\t{path.name}\tmissing")
                failures += 1
                continue
            try:
                cues = parse_srt(path)
                if not cues:
                    raise ValueError("subtitle contains zero cues")
                if language == "ja":
                    quality_errors = transcript_quality_errors(cues)
                    if quality_errors:
                        raise ValueError("; ".join(quality_errors))
                    metadata = provenance_path("transcription", path)
                    expected_fingerprint = transcription_fingerprint(video, path, "0")
                    provenance_label = "transcription"
                else:
                    metadata = provenance_path("translation", path)
                    expected_fingerprint = translation_fingerprint(video.with_suffix(".ja.srt"))
                    provenance_label = "translation"
                try:
                    recorded_fingerprint = json.loads(metadata.read_text(encoding="utf-8"))["fingerprint"]
                except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"missing or malformed {provenance_label} provenance") from exc
                if recorded_fingerprint != expected_fingerprint:
                    raise ValueError(f"stale {provenance_label} provenance")
                previous_start = -1
                for expected, cue in enumerate(cues, 1):
                    if cue.index != expected:
                        raise ValueError(f"cue index {cue.index}, expected {expected}")
                    left, right = cue.timing.split(" --> ")
                    start, end = timestamp_ms(left), timestamp_ms(right)
                    if start < previous_start or end <= start or end > duration + 2000:
                        raise ValueError(f"invalid cue timing at {cue.index}")
                    if language == "zh-Hans" and japanese_re.search(cue.text):
                        raise ValueError(f"Japanese script remains at cue {cue.index}")
                    previous_start = start
                parsed[language] = cues
                report.append(f"PASS\t{path.name}\t{len(cues)} cues")
            except Exception as exc:
                report.append(f"FAIL\t{path.name}\t{exc}")
                failures += 1
        if "ja" in parsed and "zh-Hans" in parsed:
            japanese = parsed["ja"]
            chinese = parsed["zh-Hans"]
            aligned = len(japanese) == len(chinese) and all(
                source.index == translated.index
                and source.timing == translated.timing
                and bool(translated.text.strip())
                for source, translated in zip(japanese, chinese, strict=True)
            )
            if aligned:
                report.append(f"PASS\t{video.name}\tJapanese/Chinese cues aligned")
            else:
                report.append(f"FAIL\t{video.name}\tJapanese/Chinese cue count, index, or timing mismatch")
                failures += 1
    atomic_write(REPORT_DIR / "validation.tsv", "\n".join(report) + "\n")
    log(f"Validation complete: {failures} failure(s)")
    if failures:
        raise RuntimeError(f"Validation found {failures} failure(s)")


def sample(video: Path, seconds: int) -> tuple[Path, Path]:
    sample_dir = TOOLS / "sample"
    ja = sample_dir / f"{video.stem}.sample.ja.srt"
    zh = sample_dir / f"{video.stem}.sample.zh-Hans.srt"
    transcribe_one(video, ja, clip=f"0,{seconds}")
    model, tokenizer = load_translator()
    try:
        translate_srt(model, tokenizer, ja, zh)
    finally:
        del model
        gc.collect()
    log(f"Sample ready: {zh}")
    return ja, zh
