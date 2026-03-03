"""
likes_dislikes.py — extracts what the speaker likes and dislikes using sentiment analysis.

Model:  distilbert-base-uncased-finetuned-sst-2-english  (~67 MB)
        Fine-tuned on the Stanford Sentiment Treebank.
        Labels each spoken segment POSITIVE or NEGATIVE with a confidence score.

How topics are extracted:
  - Segments the model is confident are POSITIVE → extract key words → likes
  - Segments the model is confident are NEGATIVE → extract key words → dislikes
  - Low-confidence segments (below THRESHOLD) are skipped

Why not PyABSA?
  PyABSA requires Python 3.10+ (its findfile dependency uses the | type union
  syntax). This approach uses only transformers and torch, which run on Python 3.9+.

This module is self-contained. To swap the model, only edit this file.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_pool import ModelPool

SENTIMENT_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODELS_DIR   = Path(__file__).parent.parent / "models"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

# Minimum confidence before a segment counts as a like or dislike.
# 0.80 means the model must be at least 80% sure. Lower = more results, less accurate.
THRESHOLD = 0.80

BATCH_SIZE = 16

# Words that are too common to be useful as topic descriptors
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him",
    "she", "her", "it", "its", "they", "them", "their",
    "a", "an", "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "about", "as", "into", "through",
    "really", "very", "quite", "just", "so", "also", "actually",
    "not", "no", "never", "always", "sometimes", "one", "two",
    "think", "know", "like", "said", "say", "go", "going", "got",
    "get", "make", "see", "come", "want", "let", "look",
}

_pool = ModelPool.get()


def _load_sentiment():
    """Loader for the model pool — called only on cache miss."""
    local = MODELS_DIR / "sentiment"
    model_src = str(local) if local.exists() else SENTIMENT_ID
    tok = AutoTokenizer.from_pretrained(model_src)
    m = AutoModelForSequenceClassification.from_pretrained(model_src).to(DEVICE)
    m.eval()
    return (tok, m)


def _extract_topic(text: str) -> str:
    """
    Pulls the most meaningful content words out of a sentence.
    Strips stopwords, punctuation, and very short words, then returns
    the first 5 content words joined as a short phrase.

    For example: "I really loved the crispy pizza crust" → "loved crispy pizza crust"
    Not perfect, but captures the main subject without needing spaCy or NLTK.
    """
    words = text.split()
    content = []
    for w in words:
        clean = w.strip(".,!?\"'()[]—-")
        if clean.lower() not in STOPWORDS and len(clean) > 2:
            content.append(clean)
        if len(content) == 5:
            break
    return " ".join(content).strip()


def extract_likes_dislikes(segments: list) -> dict:
    """
    Classifies each spoken segment as positive or negative using a fine-tuned
    DistilBERT model, then extracts key topics from the high-confidence segments.

    How it works:
      1. Load the sentiment model (loads locally from models/sentiment/ if downloaded)
      2. Batch-tokenise segments and run them through the classifier
      3. If POSITIVE and confidence >= threshold -> extract topic -> add to likes
         If NEGATIVE and confidence >= threshold -> extract topic -> add to dislikes
      4. Deduplicate both lists while keeping the first occurrence of each

    Returns a dict with: likes (list of strings), dislikes (list of strings).
    """
    tokenizer, model = _pool.acquire("sentiment", _load_sentiment)

    # Collect non-empty segments with their original text for topic extraction
    texts = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            texts.append(text)

    likes: list = []
    dislikes: list = []

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
                probs = torch.softmax(logits, dim=-1).cpu().tolist()

                for j, text in enumerate(batch):
                    # SST-2 label order: index 0 = NEGATIVE, index 1 = POSITIVE
                    neg_score, pos_score = probs[j][0], probs[j][1]

                    topic = _extract_topic(text)
                    if not topic:
                        continue

                    if pos_score >= THRESHOLD:
                        likes.append(topic)
                    elif neg_score >= THRESHOLD:
                        dislikes.append(topic)
    finally:
        _pool.release("sentiment")

    def dedup(lst: list) -> list:
        """Remove duplicate entries while keeping the first occurrence of each."""
        seen: set = set()
        out: list = []
        for item in lst:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    return {"likes": dedup(likes), "dislikes": dedup(dislikes)}
