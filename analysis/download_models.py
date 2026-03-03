"""
Downloads all analysis models into models/.
Run once before going offline. Safe to re-run — clears and re-downloads each model.
"""

import shutil
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = [
    {
        "id": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "dir": "wav2vec2-emotion",
        "feature_extractor": Wav2Vec2FeatureExtractor,
        "model_class": Wav2Vec2ForSequenceClassification,
        "size": "~1.2 GB",
    },
    {
        "id": "monologg/bert-base-cased-goemotions-original",
        "dir": "text-emotion",
        "feature_extractor": AutoTokenizer,
        "model_class": AutoModelForSequenceClassification,
        "size": "~400 MB",
    },
    {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "dir": "sentiment",
        "feature_extractor": AutoTokenizer,
        "model_class": AutoModelForSequenceClassification,
        "size": "~67 MB",
    },
    {
        "id": "typeform/distilbert-base-uncased-mnli",
        "dir": "zero-shot",
        "feature_extractor": AutoTokenizer,
        "model_class": AutoModelForSequenceClassification,
        "size": "~255 MB",
    },
]


def download_model(entry: dict):
    """
    Downloads a single transformers model (tokeniser/feature extractor + weights)
    and saves it to a named subfolder inside models/.

    Fix 14: uses an atomic write pattern — downloads to a temporary directory
    first, then replaces the old directory in one rename step.  This means
    that if the download fails (network error, disk full, etc.) the existing
    working model is left intact and the user isn't left with a broken install.
    """
    save_dir = MODELS_DIR / entry["dir"]
    tmp_dir  = MODELS_DIR / (entry["dir"] + "__tmp")

    # Clean up any leftover temp directory from a previous failed download
    if tmp_dir.exists():
        print(f"  Cleaning up leftover temp dir for {entry['dir']} ...")
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"  Downloading {entry['id']} ({entry['size']}) ...")
    try:
        entry["feature_extractor"].from_pretrained(entry["id"]).save_pretrained(str(tmp_dir))
        entry["model_class"].from_pretrained(entry["id"]).save_pretrained(str(tmp_dir))
    except Exception:
        # Download failed — remove partial temp dir and propagate
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Atomically replace old model only after a successful download
    if save_dir.exists():
        print(f"  Replacing old {entry['dir']} ...")
        shutil.rmtree(save_dir)
    tmp_dir.rename(save_dir)
    print(f"  Saved to {save_dir}")


def main():
    print(f"Saving all models to: {MODELS_DIR}\n")
    for entry in MODELS:
        download_model(entry)
        print()
    print("All models downloaded. You can now run analyse.py without an internet connection.")


if __name__ == "__main__":
    main()
