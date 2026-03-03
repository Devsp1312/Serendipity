"""
profiles.py — Long-term speaker profiles for Serendipity.

Builds persistent profiles for individual speakers across multiple audio sessions.
Each profile scores the speaker on 19 behavioral/character metrics (20-metric
framework minus Visual Capital, which requires video input).

Metrics are scored using zero-shot NLI (same model as person_type.py),
optionally blended with GoEmotions signal where relevant.

Usage:
    from profiles import add_session, load_profile, list_profiles
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import pipeline as hf_pipeline

from model_pool import ModelPool

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent    # Serendipity/
JSON_DIR     = ROOT / "data" / "reports"
SEGMENTS_DIR = ROOT / "data" / "transcripts"
PROFILES_DIR = ROOT / "data" / "profiles"

MODELS_DIR    = ROOT / "models"
ZERO_SHOT_ID  = "typeform/distilbert-base-uncased-mnli"
ZERO_SHOT_DIR = MODELS_DIR / "zero-shot"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_pool = ModelPool.get()

# Batch size for NLI inference (76 pairs total, processed in sub-batches)
_NLI_BATCH_SIZE = 32

# ─── Phase metadata ───────────────────────────────────────────────────────────

PHASES = {
    1: {"name": "Surface",    "subtitle": "What's immediately visible"},
    2: {"name": "Resource",   "subtitle": "What they control"},
    3: {"name": "Behavioral", "subtitle": "How they act"},
    4: {"name": "Deep Core",  "subtitle": "Who they are"},
}

PHASE_COLOURS = {
    1: "#f59e0b",   # amber  — Surface
    2: "#3b82f6",   # blue   — Resource
    3: "#7c3aed",   # purple — Behavioral
    4: "#10b981",   # emerald — Deep Core
}

# ─── Metric definitions ───────────────────────────────────────────────────────
# Each metric has:
#   phase          — which of the 4 phases it belongs to
#   name           — display name
#   description    — one-line description shown in the UI
#   positive_labels — labels that indicate a high score
#   all_labels     — all candidate labels for zero-shot classification
#   emotion_boost  — GoEmotions labels that raise the score (optional)
#   emotion_suppress — GoEmotions labels that lower the score (optional)

METRIC_DEFINITIONS = {

    # ── Phase 1 — Surface ─────────────────────────────────────────────────────
    "economic_utility": {
        "phase": 1,
        "name": "Economic Utility",
        "description": "Ability to generate profit or solve expensive problems",
        "positive_labels": [
            "a high earning professional",
            "a successful business owner",
        ],
        "all_labels": [
            "a high earning professional",
            "a successful business owner",
            "a financially struggling person",
            "an entry level worker",
        ],
    },
    "institutional_accreditation": {
        "phase": 1,
        "name": "Institutional Accreditation",
        "description": "Formal credentials — degrees, titles, elite affiliations",
        "positive_labels": [
            "a university educated person",
            "a certified professional",
        ],
        "all_labels": [
            "a university educated person",
            "a certified professional",
            "a self-taught person",
            "someone without formal credentials",
        ],
    },
    "network_density": {
        "phase": 1,
        "name": "Network Density",
        "description": "Richness of connections — not just who they know, but who knows them",
        "positive_labels": [
            "a highly connected person",
            "a well networked individual",
        ],
        "all_labels": [
            "a highly connected person",
            "a well networked individual",
            "a socially isolated person",
            "someone with very few connections",
        ],
    },
    "attention_capture": {
        "phase": 1,
        "name": "Attention Capture",
        "description": "Gravitational pull — ability to command a room or a feed",
        "positive_labels": [
            "a compelling storyteller",
            "a charismatic communicator",
        ],
        "all_labels": [
            "a compelling storyteller",
            "a charismatic communicator",
            "a dull speaker",
            "someone difficult to engage with",
        ],
        "emotion_boost":       ["excitement", "surprise", "amusement"],
        "audio_emotion_boost": ["happy", "surprised"],
    },

    # ── Phase 2 — Resource ────────────────────────────────────────────────────
    "asset_accumulation": {
        "phase": 2,
        "name": "Asset Accumulation",
        "description": "Ownership of wealth, property, stocks, or IP",
        "positive_labels": [
            "a wealthy person",
            "an owner of significant assets",
        ],
        "all_labels": [
            "a wealthy person",
            "an owner of significant assets",
            "a financially modest person",
            "someone with few material possessions",
        ],
    },
    "time_sovereignty": {
        "phase": 2,
        "name": "Time Sovereignty",
        "description": "Luxury of controlling one's own schedule",
        "positive_labels": [
            "a person with complete schedule flexibility",
            "someone who is their own boss",
        ],
        "all_labels": [
            "a person with complete schedule flexibility",
            "someone who is their own boss",
            "a time-constrained person",
            "someone who trades time for money",
        ],
    },
    "information_asymmetry": {
        "phase": 2,
        "name": "Information Asymmetry",
        "description": "Access to insider knowledge or data the public lacks",
        "positive_labels": [
            "a person with insider knowledge",
            "an information gatekeeper",
        ],
        "all_labels": [
            "a person with insider knowledge",
            "an information gatekeeper",
            "a transparent communicator",
            "someone without special information access",
        ],
    },
    "scarcity": {
        "phase": 2,
        "name": "Scarcity",
        "description": "How rare and difficult to replace this person is",
        "positive_labels": [
            "a rare and hard to replace person",
            "a person with a highly unique skill set",
        ],
        "all_labels": [
            "a rare and hard to replace person",
            "a person with a highly unique skill set",
            "an easily replaceable person",
            "someone with generic, common skills",
        ],
    },
    "generational_legacy": {
        "phase": 2,
        "name": "Generational Legacy",
        "description": "Thinking in terms of long-term impact and inheritance",
        "positive_labels": [
            "a legacy builder",
            "a person who thinks across generations",
        ],
        "all_labels": [
            "a legacy builder",
            "a person who thinks across generations",
            "a present-focused person",
            "someone unconcerned with long-term impact",
        ],
    },

    # ── Phase 3 — Behavioral ──────────────────────────────────────────────────
    "risk_tolerance": {
        "phase": 3,
        "name": "Risk Tolerance",
        "description": "Willingness to bet big on high-stakes decisions",
        "positive_labels": [
            "a high risk taker",
            "a calculated risk taker",
        ],
        "all_labels": [
            "a high risk taker",
            "a calculated risk taker",
            "a risk averse person",
            "an overly cautious person",
        ],
        "emotion_boost":        ["excitement", "optimism"],
        "emotion_suppress":     ["nervousness", "fear"],
        "audio_emotion_boost":  ["happy"],
        "audio_emotion_suppress": ["fearful"],
    },
    "reliability_consistency": {
        "phase": 3,
        "name": "Reliability / Consistency",
        "description": "Doing what they say they will do, consistently",
        "positive_labels": [
            "a highly reliable person",
            "a person who consistently follows through",
        ],
        "all_labels": [
            "a highly reliable person",
            "a person who consistently follows through",
            "an inconsistent person",
            "someone who often breaks commitments",
        ],
    },
    "social_proof": {
        "phase": 3,
        "name": "Social Proof",
        "description": "Being publicly valued by other high-value people",
        "positive_labels": [
            "a person with strong social proof",
            "an influential and widely recognised figure",
        ],
        "all_labels": [
            "a person with strong social proof",
            "an influential and widely recognised figure",
            "a person seeking external validation",
            "someone without public recognition",
        ],
    },
    "moral_signaling": {
        "phase": 3,
        "name": "Moral Signaling",
        "description": "Public performance of virtue and ethical values",
        "positive_labels": [
            "a virtue driven person",
            "a morally motivated individual",
        ],
        "all_labels": [
            "a virtue driven person",
            "a morally motivated individual",
            "a pragmatic person",
            "someone indifferent to ethics",
        ],
        "emotion_boost":       ["caring", "disapproval", "remorse"],
        "audio_emotion_boost": ["calm", "neutral"],
    },
    "cultural_compliance": {
        "phase": 3,
        "name": "Cultural Compliance",
        "description": "Seamlessly fitting in — or strategically, safely breaking norms",
        "positive_labels": [
            "a culturally fluent person",
            "a strategic and calculated norm breaker",
        ],
        "all_labels": [
            "a culturally fluent person",
            "a strategic and calculated norm breaker",
            "a rigid conformist",
            "someone who blindly follows rules",
        ],
    },

    # ── Phase 4 — Deep Core ───────────────────────────────────────────────────
    "emotional_intelligence": {
        "phase": 4,
        "name": "Emotional Intelligence",
        "description": "Reading, navigating, and managing complex social emotions",
        "positive_labels": [
            "an emotionally intelligent person",
            "a person with high emotional awareness",
        ],
        "all_labels": [
            "an emotionally intelligent person",
            "a person with high emotional awareness",
            "an emotionally unaware person",
            "someone with low emotional intelligence",
        ],
        "emotion_boost":          ["caring", "realization", "relief"],
        "emotion_suppress":       ["anger", "annoyance"],
        "audio_emotion_boost":    ["calm", "neutral"],
        "audio_emotion_suppress": ["angry"],
    },
    "creative_agency": {
        "phase": 4,
        "name": "Creative Agency",
        "description": "Capacity to generate something original from nothing",
        "positive_labels": [
            "a creative builder",
            "a person who invents and creates from scratch",
        ],
        "all_labels": [
            "a creative builder",
            "a person who invents and creates from scratch",
            "a consumer of other people's work",
            "someone who only executes instructions",
        ],
        "emotion_boost":       ["excitement", "curiosity", "desire"],
        "audio_emotion_boost": ["happy", "surprised"],
    },
    "resilience": {
        "phase": 4,
        "name": "Resilience / Antifragility",
        "description": "Growing stronger from failure, stress, and adversity",
        "positive_labels": [
            "a resilient and antifragile person",
            "someone who grows stronger from setbacks",
        ],
        "all_labels": [
            "a resilient and antifragile person",
            "someone who grows stronger from setbacks",
            "a fragile person",
            "someone easily broken by failure",
        ],
        "emotion_boost":          ["optimism", "relief", "realization"],
        "emotion_suppress":       ["grief", "sadness"],
        "audio_emotion_suppress": ["sad", "fearful"],
    },
    "tribal_loyalty": {
        "phase": 4,
        "name": "Tribal Loyalty",
        "description": "Willingness to sacrifice personal gain for the in-group",
        "positive_labels": [
            "a deeply loyal team player",
            "a strong community member",
        ],
        "all_labels": [
            "a deeply loyal team player",
            "a strong community member",
            "a lone wolf",
            "someone who always puts themselves first",
        ],
        "emotion_boost":          ["caring", "gratitude"],
        "emotion_suppress":       ["disappointment", "disgust"],
        "audio_emotion_suppress": ["angry", "disgust"],
    },
    "transcendence": {
        "phase": 4,
        "name": "Transcendence",
        "description": "Ability to symbolise a higher ideal and inspire mass movements",
        "positive_labels": [
            "a visionary person",
            "someone motivated by a higher purpose beyond personal gain",
        ],
        "all_labels": [
            "a visionary person",
            "someone motivated by a higher purpose beyond personal gain",
            "a pragmatic materialist",
            "someone focused only on immediate personal gains",
        ],
    },
}


# ─── Filename safety ──────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """
    Strip path separators and dangerous characters from a profile name so it
    can be safely used as a filename.  Replaces /  \  :  *  ?  "  <  >  |
    with underscores.
    """
    return re.sub(r'[/\\:*?"<>|]', '_', name)


# ─── Text sampling ────────────────────────────────────────────────────────────

def _sample_text(texts: list[str], max_chars: int = 2000) -> str:
    """
    Take a balanced sample from each session's text, up to max_chars total.
    Each session contributes an equal slice (beginning + end) of its transcript.
    This ensures the zero-shot model sees content from all sessions, not just
    the first one.
    """
    if not texts:
        return ""
    per = max(40, max_chars // len(texts))
    samples = []
    for text in texts:
        if len(text) <= per:
            samples.append(text)
        else:
            half = per // 2
            samples.append(text[:half] + " " + text[-half:])
    return " ".join(samples)


# ─── Emotion signal helper ────────────────────────────────────────────────────

def _emotion_signal(emotion_scores: dict, boost: list[str], suppress: list[str]) -> float:
    """
    Derive a 0–1 adjustment signal from aggregated GoEmotions scores.
    Boost emotions raise the score; suppress emotions lower it.
    Returns a value in roughly [0, 1] (0.5 = neutral).
    """
    b = sum(emotion_scores.get(e, 0.0) for e in boost)   / max(len(boost), 1)
    s = sum(emotion_scores.get(e, 0.0) for e in suppress) / max(len(suppress), 1)
    return max(0.0, min(1.0, 0.5 + b * 0.5 - s * 0.3))


# ─── Zero-shot model loader ──────────────────────────────────────────────────

def _load_zero_shot():
    """Loader for the model pool — called only on cache miss."""
    model_src = str(ZERO_SHOT_DIR) if (ZERO_SHOT_DIR / "config.json").exists() else ZERO_SHOT_ID
    return hf_pipeline(
        "zero-shot-classification",
        model=model_src,
        device=DEVICE,
    )


# ─── Precomputed pair layout ─────────────────────────────────────────────────
# Built once at import time so compute_metrics() doesn't rebuild it on each call.

_METRIC_BOUNDARIES: list[tuple[int, int, str]] = []   # (start, end, metric_key)
_ALL_HYPOTHESES: list[str] = []

def _build_pair_layout():
    idx = 0
    for key, defn in METRIC_DEFINITIONS.items():
        labels = defn["all_labels"]
        _METRIC_BOUNDARIES.append((idx, idx + len(labels), key))
        for label in labels:
            _ALL_HYPOTHESES.append(f"This speaker is {label}.")
        idx += len(labels)

_build_pair_layout()


# ─── Core metric scoring ──────────────────────────────────────────────────────

def compute_metrics(
    session_texts: list[str],
    agg_text_emotions: dict,
    agg_audio_emotions: dict | None = None,
) -> dict:
    """
    Score the speaker on all 19 metrics using zero-shot NLI.

    Instead of calling the classifier 19 times (once per metric), this
    batches all 76 premise-hypothesis pairs (19 metrics x 4 labels) into
    a single forward pass, giving a ~5-8x speedup.

    session_texts       — list of full transcript strings, one per session
    agg_text_emotions   — duration-weighted average GoEmotions scores (27 labels)
    agg_audio_emotions  — duration-weighted average wav2vec2 scores (8 labels), optional.
                          When provided, metrics that define audio_emotion_boost /
                          audio_emotion_suppress use a 3-signal blend:
                            70% NLI + 15% text emotion + 15% audio emotion
                          Metrics with only text emotion signals keep the original
                            75% NLI + 25% text emotion split.

    Returns {metric_key: {"score": float, "label": str}}
    """
    combined = _sample_text(session_texts, max_chars=2000)

    # Guard against empty input — return safe defaults instead of crashing.
    if not combined.strip():
        return {
            key: {"score": 0.0, "label": "insufficient data"}
            for key in METRIC_DEFINITIONS
        }

    classifier = _pool.acquire("zero_shot", _load_zero_shot)
    try:
        model = classifier.model
        tokenizer = classifier.tokenizer
        device = next(model.parameters()).device
        entailment_id = classifier.entailment_id

        # Figure out contradiction class ID
        n_classes = model.config.num_labels
        if n_classes == 3:
            contradiction_id = 0 if entailment_id == 2 else 2
        else:
            contradiction_id = 0

        # Build premise list (same text repeated for all 76 pairs)
        premises = [combined] * len(_ALL_HYPOTHESES)

        # Batch tokenize and forward pass
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(premises), _NLI_BATCH_SIZE):
                batch_p = premises[i:i + _NLI_BATCH_SIZE]
                batch_h = _ALL_HYPOTHESES[i:i + _NLI_BATCH_SIZE]
                inputs = tokenizer(
                    batch_p, batch_h,
                    return_tensors="pt", padding=True,
                    truncation="only_first", max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)  # (76, 3)

        # Per-pair: softmax on [contradiction, entailment] → entailment probability
        ce_logits = all_logits[:, [contradiction_id, entailment_id]]
        ce_probs = torch.softmax(ce_logits, dim=-1)
        entail_probs = ce_probs[:, 1]  # (76,)

        # Score each metric
        results = {}

        for start, end, key in _METRIC_BOUNDARIES:
            defn = METRIC_DEFINITIONS[key]
            metric_entail = entail_probs[start:end]

            # Softmax across labels within this metric (matches multi_label=False)
            normalized = torch.softmax(metric_entail, dim=0).cpu().tolist()
            labels = defn["all_labels"]
            label_scores = dict(zip(labels, normalized))

            # Zero-shot score = mean of positive label scores
            pos    = defn["positive_labels"]
            zs_scr = sum(label_scores.get(l, 0.0) for l in pos) / len(pos)

            # Blend with emotion signals if defined.
            text_boost     = defn.get("emotion_boost", [])
            text_suppress  = defn.get("emotion_suppress", [])
            audio_boost    = defn.get("audio_emotion_boost", [])
            audio_suppress = defn.get("audio_emotion_suppress", [])

            has_text  = bool((text_boost or text_suppress) and agg_text_emotions)
            has_audio = bool((audio_boost or audio_suppress) and agg_audio_emotions)

            if has_text and has_audio:
                text_sig  = _emotion_signal(agg_text_emotions,  text_boost,  text_suppress)
                audio_sig = _emotion_signal(agg_audio_emotions, audio_boost, audio_suppress)
                final = 0.70 * zs_scr + 0.15 * text_sig + 0.15 * audio_sig
            elif has_text:
                text_sig = _emotion_signal(agg_text_emotions, text_boost, text_suppress)
                final = 0.75 * zs_scr + 0.25 * text_sig
            elif has_audio:
                audio_sig = _emotion_signal(agg_audio_emotions, audio_boost, audio_suppress)
                final = 0.75 * zs_scr + 0.25 * audio_sig
            else:
                final = zs_scr

            best = max(labels, key=lambda l: label_scores.get(l, 0.0))

            results[key] = {
                "score": round(max(0.0, min(1.0, final)), 3),
                "label": best,
            }

        return results
    finally:
        _pool.release("zero_shot")


# ─── Profile CRUD ─────────────────────────────────────────────────────────────

def load_profile(name: str) -> Optional[dict]:
    PROFILES_DIR.mkdir(exist_ok=True)
    # Fix 13: sanitise name so path separators/special chars can't escape the dir
    path = PROFILES_DIR / f"{_safe_filename(name)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile(data: dict) -> None:
    PROFILES_DIR.mkdir(exist_ok=True)
    # Fix 13: sanitise name before using as filename
    path = PROFILES_DIR / f"{_safe_filename(data['name'])}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_profile(name: str) -> Optional[dict]:
    """
    Reset a profile to its initial empty state, keeping only the name and
    created_at timestamp.  Returns the reset profile dict, or None if not found.
    """
    profile = load_profile(name)
    if profile is None:
        return None
    profile["last_updated"]       = datetime.now().isoformat(timespec="seconds")
    profile["session_count"]      = 0
    profile["total_duration_sec"] = 0.0
    profile["sessions"]           = []
    profile["metrics"]            = {}
    profile["emotion_profile"]    = {}
    profile["interests"]          = {"likes": [], "dislikes": []}
    save_profile(profile)
    return profile


def delete_profile(name: str) -> bool:
    # Fix 13: sanitise name before constructing path
    path = PROFILES_DIR / f"{_safe_filename(name)}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_profiles() -> list[dict]:
    PROFILES_DIR.mkdir(exist_ok=True)
    out = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "name":          data["name"],
                "session_count": data.get("session_count", 0),
                "last_updated":  data.get("last_updated", ""),
            })
        except Exception:
            pass
    return out


# ─── Session management ───────────────────────────────────────────────────────

def add_session(
    profile_name: str,
    audio_filename: str,
    progress_cb=None,
) -> dict:
    """
    Add an audio session to a profile (creating it if it doesn't exist),
    then recompute all 19 metrics from the full session history.

    profile_name   — the name of the profile (used as filename)
    audio_filename — basename of the audio file (e.g. 'interview.mp3')
    progress_cb    — optional callable(str) for streaming status messages

    Returns the updated profile dict.
    Raises FileNotFoundError if the analysis JSON or transcript are missing.
    """
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    stem = Path(audio_filename).stem

    # ── 1. Validate prerequisites ──────────────────────────────────────────
    json_path = JSON_DIR / (stem + ".json")
    seg_path  = SEGMENTS_DIR / (stem + ".segments.json")

    if not json_path.exists():
        raise FileNotFoundError(
            f"No analysis found for '{audio_filename}'. Run analysis first."
        )
    if not seg_path.exists():
        raise FileNotFoundError(
            f"No transcript found for '{audio_filename}'. "
            "Run transcription (Stage 1) or analysis first."
        )

    analysis = json.loads(json_path.read_text(encoding="utf-8"))

    # ── 2. Load or create profile ──────────────────────────────────────────
    profile = load_profile(profile_name)
    if profile is None:
        profile = {
            "name":             profile_name,
            "created_at":       datetime.now().isoformat(timespec="seconds"),
            "last_updated":     "",
            "session_count":    0,
            "total_duration_sec": 0.0,
            "sessions":         [],
            "metrics":          {},
            "emotion_profile":  {},
            "interests":        {"likes": [], "dislikes": []},
        }

    # ── 3. Add session (idempotent) ────────────────────────────────────────
    existing = {s["file"] for s in profile.get("sessions", [])}
    if audio_filename not in existing:
        profile["sessions"].append({
            "file":        audio_filename,
            "analysed_at": analysis.get("analysed_at", ""),
            "duration_sec": analysis.get("transcript", {}).get("total_duration_sec", 0.0),
        })

    n = len(profile["sessions"])
    _cb(f"Loaded {n} session{'s' if n != 1 else ''} for '{profile_name}'")

    # ── 4. Aggregate text and emotions across all sessions ─────────────────
    session_texts    = []
    all_durations    = []   # durations of sessions that have transcript text
    all_emo_durations = []  # Fix 12: durations of sessions that have emotion data
    agg_text_emo     = {}
    agg_audio_emo    = {}
    seen_likes       = set()
    seen_dislikes    = set()
    all_likes        = []
    all_dislikes     = []

    for sess in profile["sessions"]:
        s_stem     = Path(sess["file"]).stem
        s_seg_path = SEGMENTS_DIR / (s_stem + ".segments.json")
        s_json_path = JSON_DIR / (s_stem + ".json")
        dur        = sess.get("duration_sec", 1.0) or 1.0

        if s_seg_path.exists():
            sd   = json.loads(s_seg_path.read_text(encoding="utf-8"))
            text = " ".join(seg["text"] for seg in sd.get("segments", []))
            session_texts.append(text)
            all_durations.append(dur)

        if s_json_path.exists():
            # Fix 12: track duration for emotion normalisation separately from
            # text durations — a session may have .json but no .segments.json
            all_emo_durations.append(dur)
            sa = json.loads(s_json_path.read_text(encoding="utf-8"))
            # Accumulate (will be normalised after the loop)
            for emo, sc in sa.get("text_emotion", {}).get("scores", {}).items():
                agg_text_emo[emo] = agg_text_emo.get(emo, 0.0) + sc * dur
            for emo, sc in sa.get("audio_emotion", {}).get("scores", {}).items():
                agg_audio_emo[emo] = agg_audio_emo.get(emo, 0.0) + sc * dur

            for like in sa.get("likes", []):
                k = like.lower().strip()
                if k not in seen_likes:
                    seen_likes.add(k)
                    all_likes.append(like)
            for dislike in sa.get("dislikes", []):
                k = dislike.lower().strip()
                if k not in seen_dislikes:
                    seen_dislikes.add(k)
                    all_dislikes.append(dislike)

    # Normalise emotion scores using the correct total (Fix 12)
    total_dur = sum(all_emo_durations) or 1.0
    agg_text_emo  = {k: round(v / total_dur, 4) for k, v in agg_text_emo.items()}
    agg_audio_emo = {k: round(v / total_dur, 4) for k, v in agg_audio_emo.items()}

    # ── 5. Compute metrics ─────────────────────────────────────────────────
    _cb(f"Scoring 19 metrics across {len(session_texts)} session(s) …")
    metrics = compute_metrics(session_texts, agg_text_emo, agg_audio_emo)

    # ── 6. Build updated profile ───────────────────────────────────────────
    profile["last_updated"]     = datetime.now().isoformat(timespec="seconds")
    profile["session_count"]    = len(profile["sessions"])
    profile["total_duration_sec"] = round(sum(
        s.get("duration_sec", 0) for s in profile["sessions"]
    ), 1)
    profile["metrics"] = metrics

    if agg_text_emo:
        dom = max(agg_text_emo, key=agg_text_emo.get)
        profile["emotion_profile"]["text_dominant"] = dom
        profile["emotion_profile"]["text_scores"]   = dict(
            sorted(agg_text_emo.items(), key=lambda x: -x[1])
        )
    if agg_audio_emo:
        dom = max(agg_audio_emo, key=agg_audio_emo.get)
        profile["emotion_profile"]["audio_dominant"] = dom
        profile["emotion_profile"]["audio_scores"]   = dict(
            sorted(agg_audio_emo.items(), key=lambda x: -x[1])
        )

    profile["interests"] = {"likes": all_likes, "dislikes": all_dislikes}

    save_profile(profile)
    _cb("Profile saved.")
    return profile
