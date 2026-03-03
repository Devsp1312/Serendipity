"""
Stage 2: Analyses an MP3 to produce a JSON file containing:
  - Audio emotion  (from tone of voice)
  - Text emotion   (from the words spoken — 27 GoEmotions labels)
  - Likes / dislikes (what the speaker likes or dislikes)
  - Person type    (what kind of person the speaker is)

Usage: python analyse.py
Run download_models.py first to cache models locally.
"""

import atexit
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel

from model_pool import ModelPool
from audio_emotion  import analyse_audio_emotion
from text_emotion   import analyse_text_emotion
from likes_dislikes import extract_likes_dislikes
from person_type    import classify_person_type

# Clean up cached models on shutdown
atexit.register(lambda: ModelPool.get().shutdown())

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent          # Serendipity/
AUDIO_DIR     = ROOT / "data" / "audio"
JSON_DIR      = ROOT / "data" / "reports"
SEGMENTS_DIR  = ROOT / "data" / "transcripts"
WHISPER_CACHE = ROOT / "models" / "whisper"

AUDIO_EXTENSIONS = ["*.mp3", "*.m4a", "*.wav", "*.aac", "*.flac"]

# ─── File picker ──────────────────────────────────────────────────────────────

def pick_audio_file() -> Path:
    """
    Scans the Audio files folder for .mp3 files and shows a numbered list.
    Keeps asking until the user picks a valid number, then returns that file's Path.
    Exits if no MP3 files are found.
    """
    all_audio = []
    for ext in AUDIO_EXTENSIONS:
        all_audio.extend(AUDIO_DIR.glob(ext))
    audio_files = sorted(set(all_audio), key=lambda p: p.name.lower())

    if not audio_files:
        print(f"No audio files found in: {AUDIO_DIR}")
        print(f"  Supported formats: {', '.join(e.lstrip('*') for e in AUDIO_EXTENSIONS)}")
        sys.exit(1)

    print("Audio files available:\n")
    for i, f in enumerate(audio_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {f.name}  ({size_mb:.1f} MB)")

    print()
    while True:
        choice = input(f"Select a file [1-{len(audio_files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(audio_files):
            return audio_files[int(choice) - 1]
        print("  Invalid choice, try again.")


# ─── Step 1: Get timestamped segments ─────────────────────────────────────────

def get_segments(audio_path: Path) -> tuple[list[dict], object]:
    """
    Returns a list of timed segments [{start, end, text}, ...] and an info object
    containing language and duration.

    Fast path — Stage 1 already ran:
      Reads the .segments.json file that transcribe.py saved to Text files/.
      No model is loaded, no audio is processed — just reads from disk.

    Fallback — Stage 1 hasn't been run yet:
      Automatically transcribes the audio using faster-whisper (same model as
      Stage 1), then saves both the .txt and .segments.json files so Stage 1's
      output is available for next time.
    """
    seg_file = SEGMENTS_DIR / (audio_path.stem + ".segments.json")

    if seg_file.exists():
        # Stage 1 has already run — load segments from disk, no re-transcription needed
        print(f"  Loading segments from Stage 1 cache ...")
        data = json.loads(seg_file.read_text(encoding="utf-8"))
        info = SimpleNamespace(
            duration=data["total_duration_sec"],
            language=data["language"],
            language_probability=data["language_confidence"],
        )
        return data["segments"], info

    # Stage 1 hasn't been run yet — transcribe now and save both output files
    print("  Stage 1 not run yet — transcribing now ...")
    print("  (This will also save a .txt and .segments.json to Text files/)")
    snapshots = list(WHISPER_CACHE.glob("models--Systran--faster-whisper-small/snapshots/*"))
    model_src = str(snapshots[0]) if snapshots else "small"
    model = WhisperModel(model_src, device="cpu", compute_type="int8")
    raw_segments, info = model.transcribe(str(audio_path))
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in raw_segments]
    del model

    # Save .txt so the human-readable transcript is available too
    text = " ".join(s["text"] for s in segments)
    txt_path = SEGMENTS_DIR / audio_path.with_suffix(".txt").name
    txt_path.write_text(text, encoding="utf-8")
    print(f"  Saved transcript: {txt_path.name}")

    # Save .segments.json for future Stage 2 runs (avoids re-transcribing next time)
    seg_data = {
        "file": audio_path.name,
        "language": info.language,
        "language_confidence": round(info.language_probability, 3),
        "total_duration_sec": round(info.duration, 1),
        "segments": segments,
    }
    seg_file.write_text(json.dumps(seg_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return segments, info


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    Runs the full Stage 2 pipeline in 5 steps:
      [1] Load or auto-generate timestamped segments from Stage 1
      [2] Analyse emotion from the audio signal (tone of voice)
      [3] Analyse emotion from the text (the actual words)
      [4] Extract what the speaker likes and dislikes (ABSA model)
      [5] Classify what type of person the speaker is (zero-shot)

    Each model is loaded, used, then deleted to keep peak memory low.
    Saves a single JSON file to the JSON files folder when done.
    """
    audio_file = pick_audio_file()
    JSON_DIR.mkdir(exist_ok=True)

    total_start = time.time()

    print("\n[1/5] Getting transcript segments ...")
    segments, info = get_segments(audio_file)
    full_text = " ".join(s["text"] for s in segments)
    print(f"      {len(segments)} segments, {info.duration:.0f}s audio")

    print("[2/5] Analysing audio emotion ...")
    audio_emotion = analyse_audio_emotion(audio_file, segments)
    print(f"      dominant: {audio_emotion['dominant']}")

    print("[3/5] Analysing text emotion (27 labels) ...")
    text_emotion = analyse_text_emotion(segments)
    print(f"      dominant: {text_emotion['dominant']}")

    print("[4/5] Extracting likes and dislikes ...")
    likes_dislikes = extract_likes_dislikes(segments)
    print(f"      {len(likes_dislikes['likes'])} likes, {len(likes_dislikes['dislikes'])} dislikes")

    print("[5/5] Classifying person type ...")
    person_type = classify_person_type(full_text)
    print(f"      top traits: {', '.join(person_type['top_traits'])}")

    output = {
        "file": audio_file.name,
        "analysed_at": datetime.now().isoformat(timespec="seconds"),
        "transcript": {
            "segment_count": len(segments),
            "total_duration_sec": round(info.duration, 1),
            "language": info.language,
            "language_confidence": round(info.language_probability, 3),
        },
        "audio_emotion": audio_emotion,
        "text_emotion": text_emotion,
        **likes_dislikes,
        "person_type": person_type,
    }

    out_path = JSON_DIR / audio_file.with_suffix(".json").name
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - total_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
