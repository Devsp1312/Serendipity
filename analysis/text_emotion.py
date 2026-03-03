from __future__ import annotations

"""
text_emotion.py — detects emotion from the meaning of the spoken words.

Model:   monologg/bert-base-cased-goemotions-original
Labels:  27 GoEmotions (admiration, amusement, anger, annoyance, approval,
         caring, confusion, curiosity, desire, disappointment, disapproval,
         disgust, embarrassment, excitement, fear, gratitude, grief, joy,
         love, nervousness, optimism, pride, realization, relief, remorse,
         sadness, surprise, neutral)
Size:    ~400 MB

This module is self-contained. To swap the model, only edit this file.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_pool import ModelPool

MODELS_DIR      = Path(__file__).parent.parent / "models"
TEXT_EMOTION_ID = "monologg/bert-base-cased-goemotions-original"
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 16

_pool = ModelPool.get()


def _load_text_emotion():
    """Loader for the model pool — called only on cache miss."""
    local = MODELS_DIR / "text-emotion"
    model_src = str(local) if local.exists() else TEXT_EMOTION_ID
    tok = AutoTokenizer.from_pretrained(model_src)
    m = AutoModelForSequenceClassification.from_pretrained(model_src).to(DEVICE)
    m.eval()
    return (tok, m)


def analyse_text_emotion(segments: list[dict]) -> dict:
    """
    Detects emotion from the meaning of the words (not the audio signal).

    Uses Google's GoEmotions BERT model trained on 58,000 Reddit comments
    labelled with 27 emotions. Runs on each segment's text independently,
    then combines the results using a word-count-weighted average so that
    longer segments count for more in the final result.

    Uses sigmoid activation (not softmax) because multiple emotions can be
    true at the same time — a sentence can be both joyful and admiring.
    Scores are therefore independent 0–1 values per emotion.

    Results are sorted highest-first so the dominant emotion appears first.

    Returns a dict with: dominant emotion, all 27 scores sorted desc, model ID.
    """
    tokenizer, model = _pool.acquire("text_emotion", _load_text_emotion)

    id2label = model.config.id2label

    # Filter to non-empty segments
    texts = []
    word_counts = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            texts.append(text)
            word_counts.append(max(len(text.split()), 1))

    all_probs: list[list[float]] = []

    try:
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                inputs = tokenizer(
                    batch, return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                logits = model(**inputs).logits
                # sigmoid because this is multi-label — emotions aren't mutually exclusive
                probs = torch.sigmoid(logits).cpu().tolist()
                all_probs.extend(probs)
    finally:
        _pool.release("text_emotion")

    # Fix 4: if every segment had empty text, return a neutral default.
    if not all_probs:
        return {"dominant": "neutral", "scores": {}, "model": TEXT_EMOTION_ID}

    # Word-count-weighted average across all segments
    n_labels = len(all_probs[0])
    weights = np.array(word_counts, dtype=float)
    probs_arr = np.array(all_probs)
    avg_arr = (probs_arr * weights[:, None]).sum(axis=0) / weights.sum()

    avg = {
        id2label[i]: round(float(avg_arr[i]), 4)
        for i in range(n_labels)
    }

    # Sort highest-first so the JSON is easy to read
    avg = dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))
    dominant = next(iter(avg))

    return {"dominant": dominant, "scores": avg, "model": TEXT_EMOTION_ID}
