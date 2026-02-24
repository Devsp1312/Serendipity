"""
Transcribes an MP3 from the Audio files folder using faster-whisper.
Saves two output files to the Text files folder:
  - filename.txt           plain transcript for reading
  - filename.segments.json timestamped segments used by Stage 2 (analyse.py)

The model is downloaded automatically on first run and cached locally.
Run download_model.py first if you want to pre-download it.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

AUDIO_DIR  = Path(__file__).parent.parent / "Audio files"
TEXT_DIR   = Path(__file__).parent.parent / "Text files"
MODEL_PATH = Path(__file__).parent / "model"
MODEL_SIZE = "small"  # tiny | base | small | medium | large-v3


def find_local_model() -> Optional[Path]:
    """
    Looks inside MODEL_PATH for the snapshot directory that faster-whisper
    creates when you run download_model.py.  Returns the full path to that
    snapshot so the model loads from disk instead of the internet.
    Returns None if the model hasn't been downloaded yet (it will download
    automatically on first run in that case).
    """
    snapshots = list(MODEL_PATH.glob(f"models--Systran--faster-whisper-{MODEL_SIZE}/snapshots/*"))
    return snapshots[0] if snapshots else None


def pick_audio_file() -> Path:
    """
    Scans the Audio files folder for .mp3 files and prints a numbered list.
    Keeps asking until the user enters a valid number, then returns that file's Path.
    Exits the program if no MP3 files are found.
    """
    mp3_files = sorted(AUDIO_DIR.glob("*.mp3"))

    if not mp3_files:
        print(f"No MP3 files found in: {AUDIO_DIR}")
        sys.exit(1)

    print("Audio files available:\n")
    for i, f in enumerate(mp3_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {f.name}  ({size_mb:.1f} MB)")

    print()
    while True:
        choice = input(f"Select a file [1-{len(mp3_files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(mp3_files):
            return mp3_files[int(choice) - 1]
        print("  Invalid choice, try again.")


def transcribe(audio_path: Path) -> tuple[str, list[dict], object]:
    """
    Loads the faster-whisper model and transcribes the given audio file.
    Collects every segment (with its start/end timestamps and text) into a list
    so both the plain text and the timed segment data are available to the caller.

    Returns:
      text      — the full transcript as a single string
      segments  — list of dicts: [{start, end, text}, ...]
      info      — faster-whisper TranscriptionInfo (language, duration, etc.)
    """
    local = find_local_model()
    model_source = str(local) if local else MODEL_SIZE
    print(f"\nLoading model from {'local cache' if local else 'internet'} ...")
    model = WhisperModel(model_source, device="cpu", compute_type="int8")

    print(f"Transcribing: {audio_path.name}\n")
    start = time.time()

    # Collect the generator into a list so we keep the timing data
    raw_segments, info = model.transcribe(str(audio_path))
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in raw_segments]
    text = " ".join(s["text"] for s in segments)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    return text, segments, info


def main():
    """
    Runs the full Stage 1 pipeline:
      1. Ask the user to pick an audio file
      2. Transcribe it with faster-whisper
      3. Save the plain text to Text files/filename.txt
      4. Save the timestamped segments to Text files/filename.segments.json
         (Stage 2 reads this file instead of re-transcribing from scratch)
    """
    audio_file = pick_audio_file()
    text, segments, info = transcribe(audio_file)

    # Save plain transcript (.txt) for human reading
    txt_path = TEXT_DIR / audio_file.with_suffix(".txt").name
    txt_path.write_text(text, encoding="utf-8")
    print(f"\nTranscription saved to:\n  {txt_path}")

    # Save timestamped segments (.segments.json) for Stage 2
    seg_path = TEXT_DIR / (audio_file.stem + ".segments.json")
    seg_data = {
        "file": audio_file.name,
        "language": info.language,
        "language_confidence": round(info.language_probability, 3),
        "total_duration_sec": round(info.duration, 1),
        "segments": segments,
    }
    seg_path.write_text(json.dumps(seg_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Segments saved to:\n  {seg_path}")

    # Short preview so you can sanity-check the transcript
    print(f"\n--- Preview (first 500 chars) ---\n{text[:500]}")


if __name__ == "__main__":
    main()
