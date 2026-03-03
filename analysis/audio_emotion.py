"""
audio_emotion.py — detects emotion from the tone of voice in an audio file.

Model:   ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
Labels:  angry, calm, disgust, fearful, happy, neutral, sad, surprised  (8 labels)
Size:    ~1.2 GB

This module is self-contained. To swap the model, only edit this file.
"""

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from model_pool import ModelPool

MODELS_DIR       = Path(__file__).parent.parent / "models"
AUDIO_EMOTION_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
DEVICE           = "mps" if torch.backends.mps.is_available() else "cpu"

_pool = ModelPool.get()


def _load_audio_emotion():
    """Loader for the model pool — called only on cache miss."""
    local = MODELS_DIR / "wav2vec2-emotion"
    model_src = str(local) if local.exists() else AUDIO_EMOTION_ID
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_src)
    m = Wav2Vec2ForSequenceClassification.from_pretrained(model_src).to(DEVICE)
    m.eval()
    return (fe, m)


def decode_chunk(container, stream, start_sec: float, end_sec: float,
                 target_sr: int = 16000) -> np.ndarray:
    """
    Extracts a slice of audio from an already-open container between
    start_sec and end_sec, converts it to mono and resamples to target_sr.

    Returns a 1-D float32 numpy array of audio samples.
    """
    orig_sr = stream.sample_rate
    container.seek(int(start_sec / stream.time_base), stream=stream)
    frames = []
    for frame in container.decode(audio=0):
        if frame.pts is not None and float(frame.pts * stream.time_base) > end_sec:
            break
        frames.append(frame.to_ndarray().astype(np.float32))

    if not frames:
        return np.zeros(int((end_sec - start_sec) * target_sr), dtype=np.float32)

    # Concatenate frames and average channels to get mono
    audio = np.concatenate(frames, axis=1).mean(axis=0)

    # Resample using torch interpolate — no scipy / librosa needed
    t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    new_len = int(len(audio) * target_sr / orig_sr)
    audio = F.interpolate(t, size=new_len, mode="linear", align_corners=False)
    return audio.squeeze().numpy().astype(np.float32)


def analyse_audio_emotion(audio_path: Path, segments: list[dict]) -> dict:
    """
    Detects emotion from the tone of voice (not the words).

    The wav2vec2 model reads the raw audio signal — things like pitch, pace,
    energy, and how words are delivered — and classifies it into one of 8 emotions.

    Long segments (> 8s) are split into 4-second sub-chunks before being sent
    to the model because emotion models are trained on short utterances and
    struggle with long clips. Scores from each sub-chunk are averaged together
    weighted by how long each chunk was.

    All chunk scores are then combined into a final duration-weighted average
    so longer segments count more in the result.

    Returns a dict with: dominant emotion, all 8 scores, model ID.
    """
    fe, model = _pool.acquire("audio_emotion", _load_audio_emotion)

    all_scores: list[list[float]] = []
    all_weights: list[float] = []

    id2label = model.config.id2label  # full English words — no remapping needed

    try:
        container = av.open(str(audio_path))
        try:
            if not container.streams.audio:
                raise ValueError(
                    f"No audio stream found in '{audio_path.name}'. "
                    "The file may be corrupt or contain only video/data streams."
                )
            stream = container.streams.audio[0]

            with torch.no_grad():
                for seg in segments:
                    duration = seg["end"] - seg["start"]
                    chunk_size = 4.0
                    n_chunks = max(1, int(duration / chunk_size))

                    for i in range(n_chunks):
                        sub_start = seg["start"] + i * chunk_size
                        sub_end   = min(sub_start + chunk_size, seg["end"])

                        audio = decode_chunk(container, stream, sub_start, sub_end)
                        if len(audio) < 400:  # skip clips too short to be meaningful
                            continue

                        inputs = fe(
                            audio, sampling_rate=16000, return_tensors="pt", padding=True
                        )
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        logits = model(**inputs).logits
                        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

                        all_scores.append(probs if isinstance(probs, list) else [probs])
                        all_weights.append(sub_end - sub_start)
        finally:
            container.close()
    finally:
        _pool.release("audio_emotion")

    # Fix 2 (ZeroDivisionError): if every chunk was too short to analyse,
    # all_scores and all_weights are both empty → weights.sum() == 0.
    # Raise a clear error instead of a cryptic ZeroDivisionError.
    if not all_scores:
        raise ValueError(
            "No analysable audio chunks found — the audio may be silent, "
            "too short (< 25 ms per segment), or all segments were skipped."
        )

    # Duration-weighted average across all chunks
    weights    = np.array(all_weights)
    scores_arr = np.array(all_scores)
    avg        = (scores_arr * weights[:, None]).sum(axis=0) / weights.sum()

    scores_dict = {
        id2label[i]: round(float(avg[i]), 4)
        for i in range(len(avg))
    }
    dominant = max(scores_dict, key=scores_dict.get)

    return {"dominant": dominant, "scores": scores_dict, "model": AUDIO_EMOTION_ID}
