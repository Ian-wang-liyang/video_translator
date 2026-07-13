#!/usr/bin/env python3
"""Resumable, fully local Japanese -> Simplified Chinese subtitle pipeline."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import shutil
import subprocess
import time
import unicodedata
import zlib
from dataclasses import dataclass
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
TRANSCRIPTION_REVISION = "chunked-v1"
TRANSLATION_REVISION = "batched-v1"
TRANSLATION_PROMPT_REVISION = "ja-zh-hans-v1"
CHUNK_SECONDS = SETTINGS.chunk_seconds
_TRANSCRIBER = None
JSON_OUTPUT = False
TIMING_RE = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> "
    r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})$"
)


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
        f"{SETTINGS.device}|{WHISPER_MODEL}"
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
    blocks: list[str] = []
    cue_number = 1
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
        blocks.append(
            f"{cue_number}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}"
        )
        cue_number += 1
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
        result = _TRANSCRIBER.transcribe(str(video), clip_timestamps=clip)
        rendered = segments_to_srt(filter_repetition_bursts(result.get("segments", [])))
    else:
        checkpoint_dir = STATE_DIR / "transcription" / fingerprint
        audio_dir = TOOLS / "cache" / "audio" / fingerprint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", str(video),
                 "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
                 "-reset_timestamps", "1", str(audio_dir / "chunk-%04d.wav")],
                check=True,
            )
            blocks: list[str] = []
            cue_number = 1
            offset = 0.0
            chunks = sorted(audio_dir.glob("chunk-*.wav"))
            if not chunks:
                raise RuntimeError("FFmpeg produced no audio chunks")
            for number, audio_chunk in enumerate(chunks):
                checkpoint = checkpoint_dir / f"chunk-{number:04d}.srt"
                if checkpoint.exists():
                    chunk_cues = parse_srt(checkpoint)
                    chunk_errors = transcript_quality_errors(chunk_cues)
                    if chunk_errors:
                        checkpoint.unlink()
                        chunk_cues = []
                else:
                    chunk_cues = []
                if not chunk_cues:
                    result = _TRANSCRIBER.transcribe(str(audio_chunk))
                    adjusted = []
                    for segment in filter_repetition_bursts(result.get("segments", [])):
                        segment = dict(segment)
                        segment["start"] = float(segment.get("start", 0)) + offset
                        segment["end"] = float(segment.get("end", 0)) + offset
                        adjusted.append(segment)
                    chunk_text = segments_to_srt(adjusted)
                    atomic_write(checkpoint, chunk_text)
                    chunk_cues = parse_srt(checkpoint)
                    chunk_errors = transcript_quality_errors(chunk_cues)
                    if chunk_errors:
                        raise RuntimeError(
                            f"chunk {number + 1}/{len(chunks)} failed quality validation: "
                            + "; ".join(chunk_errors)
                        )
                for cue in chunk_cues:
                    blocks.append(f"{cue_number}\n{cue.timing}\n{cue.text}")
                    cue_number += 1
                duration = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(audio_chunk)],
                    check=True, capture_output=True, text=True,
                )
                offset += float(duration.stdout.strip())
                audio_chunk.unlink()
                log(f"Transcription progress {video.name}: chunk {number + 1}/{len(chunks)}")
            rendered = "\n\n".join(blocks) + ("\n" if blocks else "")
        finally:
            shutil.rmtree(audio_dir, ignore_errors=True)
    if not rendered.strip():
        raise RuntimeError(f"Whisper produced no subtitle cues for {video.name}")
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


def translate_group(model, tokenizer, texts: list[str], *, titles: bool = False) -> list[str]:
    kind = "video titles" if titles else "subtitle cues"
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, 1))
    instruction = (
        f"Translate these Japanese {kind} into natural, concise Simplified Chinese.\n"
        "Preserve names, episode/catalogue codes, numbers, and meaning. Render Japanese names "
        "in suitable Chinese characters or Chinese phonetic transcription. Do not censor or summarize.\n"
        "Return exactly one translated item for every input using the same [number] markers.\n"
        "Use Chinese script only: do not leave any Japanese hiragana or katakana. Do not include "
        "Japanese originals, explanations, markdown, or timestamps.\n"
        "If an utterance is genuinely too unclear to translate, return （日语发音不清） for that item.\n\n"
        f"{numbered}"
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
        for text in texts:
            output.extend(translate_group(model, tokenizer, [text], titles=titles))
        return output
    if not titles:
        log(f"WARN using unclear-speech fallback for cue: {texts[0]!r}")
        return ["（日语发音不清）"]
    raise RuntimeError(f"Could not parse translation response: {response[:300]!r}")


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
        chinese = translate_group(model, tokenizer, [cue.text for cue in group])
        translated.extend(
            Cue(index=cue.index, timing=cue.timing, text=text)
            for cue, text in zip(group, chinese, strict=True)
        )
        log(f"Translation progress {source.name}: {len(translated)}/{len(source_cues)} cues")
    atomic_write(destination, cues_to_srt(translated))
    atomic_write(metadata, json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    log(f"DONE translation: {destination.name}")


def translate_titles(model, tokenizer) -> None:
    items = videos()
    translated: list[str] = []
    for offset in range(0, len(items), 8):
        translated.extend(
            translate_group(model, tokenizer, [p.stem for p in items[offset : offset + 8]], titles=True)
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_DIR / "title-mapping.csv.partial"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_filename", "suggested_simplified_chinese_title"])
        for path, title in zip(items, translated, strict=True):
            writer.writerow([path.name, title])
    temporary.replace(REPORT_DIR / "title-mapping.csv")
    log("DONE translated title mapping: .subtitle-tools/reports/title-mapping.csv")


def existing_title_mapping() -> dict[str, str]:
    path = REPORT_DIR / "title-mapping.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["original_filename"]: row["suggested_simplified_chinese_title"]
            for row in csv.DictReader(handle)
            if row.get("original_filename") and row.get("suggested_simplified_chinese_title")
        }


def checkpoint_title(model, tokenizer, video: Path, mapping: dict[str, str]) -> None:
    if video.name not in mapping:
        mapping[video.name] = translate_group(model, tokenizer, [video.stem], titles=True)[0]
    temporary = REPORT_DIR / "title-mapping.csv.partial"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_filename", "suggested_simplified_chinese_title"])
        for item in videos():
            if item.name in mapping:
                writer.writerow([item.name, mapping[item.name]])
    temporary.replace(REPORT_DIR / "title-mapping.csv")
    log(f"DONE title checkpoint: {video.name}")


def process_collection(max_videos: int | None = None) -> None:
    """Finish transcription, translation, and title for one video at a time."""
    transcription_failures: list[str] = []
    translation_failures: list[str] = []
    titles = existing_title_mapping()
    items = videos()[:max_videos] if max_videos is not None else videos()
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
            checkpoint_title(model, tokenizer, video, titles)
            log(f"DONE video: {video.name}")
        except Exception as exc:
            translation_failures.append(video.name)
            log(f"ERROR translation/title {video.name}: {type(exc).__name__}: {exc}")
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
            f"{len(translation_failures)} translation/title"
        )


def translate_folder() -> None:
    model, tokenizer = load_translator()
    failures: list[str] = []
    try:
        try:
            translate_titles(model, tokenizer)
        except Exception as exc:
            failures.append("title mapping")
            log(f"ERROR title mapping: {type(exc).__name__}: {exc}")
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
    try:
        title_path = REPORT_DIR / "title-mapping.csv"
        with title_path.open(encoding="utf-8-sig", newline="") as handle:
            raw_title_rows = list(csv.DictReader(handle))
        title_rows = existing_title_mapping()
        expected_names = {video.name for video in videos()}
        if len(raw_title_rows) != len(title_rows):
            raise ValueError("duplicate or incomplete title mapping rows")
        if set(title_rows) != expected_names:
            missing = sorted(expected_names - set(title_rows))
            extra = sorted(set(title_rows) - expected_names)
            raise ValueError(f"missing={missing}; extra={extra}")
        report.append(f"PASS\ttitle-mapping.csv\t{len(title_rows)} unique rows")
    except Exception as exc:
        report.append(f"FAIL\ttitle-mapping.csv\t{exc}")
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
