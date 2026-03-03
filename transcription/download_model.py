"""
Clears the old model folder and downloads the faster-whisper model locally.
Run this once before using transcribe.py offline.
"""

import shutil
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_SIZE = "small"  # must match MODEL_SIZE in transcribe.py
MODEL_PATH = Path(__file__).parent.parent / "models" / "whisper"

# Remove old model folder if it exists
if MODEL_PATH.exists():
    print(f"Removing old model at {MODEL_PATH} ...")
    shutil.rmtree(MODEL_PATH)

MODEL_PATH.mkdir()

print(f"Downloading faster-whisper '{MODEL_SIZE}' model ...")
print("This may take a few minutes. It only needs to run once.\n")

# Instantiating the model triggers the download; download_root controls where it's cached
WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", download_root=str(MODEL_PATH))

print(f"\nModel saved to {MODEL_PATH}/")
print("You can now run transcribe.py without an internet connection.")
