# Serendipity — CLAUDE.md

A fully local audio analysis pipeline. No API keys, no internet after setup, no data leaves the device.

---

## What it does

Takes an audio file and runs it through 5 sequential steps:

1. **Transcription** — Faster-Whisper converts speech → timestamped text segments
2. **Audio emotion** — wav2vec2 reads the raw waveform to detect tone of voice
3. **Text emotion** — BERT (GoEmotions) classifies 27 emotion labels from the words
4. **Likes / dislikes** — DistilBERT sentiment model extracts positive/negative topics
5. **Person type** — Zero-shot NLI classifies the speaker's personality archetype

Results feed into **Profiles** — long-term speaker records with 19 behavioural metrics, updated each time a new session is analysed.

---

## Project structure

```
Serendipity/
├── Audio files/        ← drop audio here (.mp3 .m4a .wav .aac .flac)
├── Text files/         ← transcripts (.txt + .segments.json)
├── JSON files/         ← analysis output (.json)
├── Reports/            ← HTML visual reports
├── Profiles/           ← speaker profiles (.json)
│
├── stage1/
│   ├── transcribe.py       ← CLI transcription
│   ├── download_model.py   ← downloads Whisper model
│   └── requirements.txt
│
└── stage2/
    ├── app.py              ← Flask web UI + SSE pipeline
    ├── analyse.py          ← CLI analysis runner
    ├── audio_emotion.py    ← wav2vec2 tone detection
    ├── text_emotion.py     ← BERT 27-label emotion
    ├── likes_dislikes.py   ← sentiment topic extraction
    ├── person_type.py      ← zero-shot personality classification
    ├── profiles.py         ← 19-metric speaker profiles
    ├── visualise.py        ← HTML report generator
    └── templates/index.html
```

---

## Key commands

```bash
# Install dependencies
pip install -r stage1/requirements.txt
pip install -r stage2/requirements.txt

# Download models (one-time, ~3 GB)
python stage1/download_model.py
python stage2/download_models.py

# Run the web UI (recommended)
python3 stage2/app.py
# → open http://localhost:5000

# CLI workflow
python stage1/transcribe.py   # step 1: transcribe audio
python stage2/analyse.py      # step 2: run analysis
python stage2/visualise.py    # step 3: open HTML report
```

---

## Architecture notes

### Stage 1 caching

Transcription output is cached as `.segments.json` in `Text files/`. If a cache file exists, Stage 2 skips model loading entirely — important because Whisper takes ~20 s to load. Never delete segment files unless you want to force re-transcription.

### Stage 2 concurrency

`app.py` uses two threading locks: `_busy_lock` for the main analysis pipeline and `_profile_busy_lock` for profile metric computation. Both locks exist because the zero-shot NLI model is shared — simultaneous loads would OOM on CPU. Do not remove these locks.

### SSE streaming

The web UI uses Server-Sent Events (`/stream` route in `app.py`) to push live progress to the browser. A heartbeat ping fires every 15 s to keep the connection alive during long model loads. When modifying pipeline steps, yield progress events to keep the UI responsive.

### Model paths

All models are stored under `stage1/model/` and `stage2/models/` (relative to the repo root). The app resolves paths via `Path(__file__).parent.parent` — don't change this anchor or model loading will break.

### GPU acceleration

All models use CPU by default. MPS (Apple Silicon) and CUDA are detected automatically and used if available. Do not hard-code `device="cpu"`.

---

## Data formats

### Segments file (`Text files/<name>.segments.json`)

```json
[{"start": 0.0, "end": 2.4, "text": "Hello world"}, ...]
```

### Analysis output (`JSON files/<name>.json`)

```json
{
  "audio_emotion": {"label": "happy", "score": 0.91},
  "text_emotions": [{"label": "joy", "score": 0.87}, ...],
  "likes": ["coffee", "music"],
  "dislikes": ["traffic"],
  "person_type": {"label": "Creative", "score": 0.78}
}
```

### Profile (`Profiles/<name>.json`)

19 metrics across 4 phases: Surface, Resource, Behavioral, Deep Core. Each metric is a float 0–1, averaged across all sessions for that speaker.

---

## Conventions

- Use `pathlib.Path` for all file paths — no raw string concatenation.
- All analysis modules expose a single public function (e.g. `analyse_audio_emotion(audio_path, segments)`). Keep that interface stable.
- Keep Stage 1 and Stage 2 dependencies separate (`requirements.txt` in each folder). Stage 1 is intentionally lightweight.
- The web UI is a single-page app served by Flask. All frontend state lives in `index.html` — no separate JS build step.
- Reports are self-contained HTML files — no external dependencies.

---

## Models used

| Model                                                       | Task                          | Approx. size |
| ----------------------------------------------------------- | ----------------------------- | ------------ |
| `Systran/faster-whisper-small`                              | Transcription                 | ~244 MB      |
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Audio emotion                 | ~1.2 GB      |
| `monologg/bert-base-cased-goemotions-original`              | Text emotion (27 labels)      | ~400 MB      |
| `distilbert-base-uncased-finetuned-sst-2-english`           | Sentiment / likes & dislikes  | ~67 MB       |
| `typeform/distilbert-base-uncased-mnli`                     | Person type + profile metrics | ~255 MB      |
