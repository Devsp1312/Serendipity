"""
person_type.py — classifies what kind of person the speaker is.

Model:   typeform/distilbert-base-uncased-mnli  (zero-shot NLI)
Labels:  10 person type categories (see PERSON_LABELS below)
Size:    ~255 MB

Uses zero-shot classification: no fine-tuning on speaker types was ever done.
The NLI model checks whether the transcript "entails" each label description,
so it generalises to any new labels you add to PERSON_LABELS without retraining.

This module is self-contained. To swap the model or add/remove person types,
only edit this file.
"""

from pathlib import Path

import torch
from transformers import pipeline

from model_pool import ModelPool

MODELS_DIR   = Path(__file__).parent.parent / "models"
ZERO_SHOT_ID = "typeform/distilbert-base-uncased-mnli"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

PERSON_LABELS = [
    "food enthusiast",
    "film critic",
    "tech reviewer",
    "sports fan",
    "music lover",
    "fitness and health conscious person",
    "entrepreneur or business-minded person",
    "academic or intellectual",
    "lifestyle content creator",
    "political commentator",
]

_pool = ModelPool.get()


def _load_zero_shot():
    """Loader for the model pool — called only on cache miss."""
    local = MODELS_DIR / "zero-shot"
    model_src = str(local) if (local / "config.json").exists() else ZERO_SHOT_ID
    return pipeline(
        "zero-shot-classification",
        model=model_src,
        device=DEVICE,
    )


def classify_person_type(full_text: str) -> dict:
    """
    Figures out what kind of person the speaker is using zero-shot NLI.

    How it works:
      The model was trained on Natural Language Inference — deciding whether one
      sentence "entails" another. We exploit this by framing each label as a
      hypothesis:
        Premise:    [the transcript]
        Hypothesis: "This speaker is a food enthusiast."
      If the model believes the transcript entails the hypothesis → high score.

      This runs independently for all 10 labels (multi_label=True), so a person
      can score highly on multiple types at the same time.

    The transcript is trimmed to first 1500 + last 500 chars before being passed
    in to stay within the model's token limit (distilbert max = 512 tokens).

    Returns the top 3 traits and all 10 scores.
    """
    # Fix 5: zero-shot classifier raises on empty string — return safe default
    if not full_text.strip():
        return {"top_traits": [], "scores": {}, "model": ZERO_SHOT_ID}

    # Take start + end of transcript to capture intro and conclusion
    if len(full_text) > 2000:
        text_in = full_text[:1500] + " " + full_text[-500:]
    else:
        text_in = full_text

    classifier = _pool.acquire("zero_shot", _load_zero_shot)
    try:
        result = classifier(
            text_in,
            candidate_labels=PERSON_LABELS,
            hypothesis_template="This speaker is a {}.",
            multi_label=True,
        )
    finally:
        _pool.release("zero_shot")

    scores = {
        label: round(score, 4)
        for label, score in zip(result["labels"], result["scores"])
    }
    top_traits = result["labels"][:3]

    return {"top_traits": top_traits, "scores": scores, "model": ZERO_SHOT_ID}
