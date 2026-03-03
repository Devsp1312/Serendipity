# Serendipity

A fully local audio analysis pipeline that transcribes any audio file and tells you:

- **Audio emotion** — what emotion comes through in the speaker's tone of voice
- **Text emotion** — what emotions are in the actual words (27 labels)
- **Likes & dislikes** — what the speaker is positive or negative about
- **Person type** — what kind of person the speaker likely is
- **Speaker profiles** — long-term behavioural metrics tracked across multiple sessions

Everything runs on your machine. No API keys, no internet connection needed after setup, no data leaves your device.

---

## Requirements

- Python 3.9 or later
- ~3 GB free disk space (for models)
- macOS, Linux, or Windows

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/your-username/serendipity.git
cd serendipity
```

**2. Install transcription dependencies**

```bash
pip3 install -r transcription/requirements.txt
```

**3. Download the Whisper transcription model**

```bash
python3 transcription/download_model.py
```

**4. Install analysis dependencies**

```bash
pip3 install -r analysis/requirements.txt
```

**5. Download the analysis models** (~2 GB, one-time)

```bash
python3 analysis/download_models.py
```

---

## Usage

### Web UI (recommended)

```bash
python3 web/app.py
```

Then open **http://localhost:5000** in your browser.

- Drop audio files into `data/audio/`
- Click a file in the sidebar to select it
- Click **Analyse** to run the full pipeline
- Watch live progress and view the report when done

### Command line

Transcribe first:

```bash
python3 transcription/transcribe.py
```

Then analyse:

```bash
python3 analysis/analyse.py
```

Results are saved to `data/reports/`. To open a visual HTML report:

```bash
python3 analysis/visualise.py
```

---

## Supported audio formats

`.mp3` `.m4a` `.wav` `.aac` `.flac`

---

## Project structure

```
Serendipity/
├── transcription/           ← audio → text (Whisper)
│   ├── transcribe.py
│   ├── download_model.py
│   └── requirements.txt
│
├── analysis/                ← text → insights (5 analysis modules)
│   ├── audio_emotion.py         tone of voice → 8 emotions
│   ├── text_emotion.py          words → 27 GoEmotions labels
│   ├── likes_dislikes.py        sentiment → topic extraction
│   ├── person_type.py           zero-shot → personality archetype
│   ├── profiles.py              19-metric speaker profiles
│   ├── model_pool.py            LRU model caching for performance
│   ├── analyse.py               CLI runner
│   ├── visualise.py             HTML report generator
│   ├── download_models.py       download all analysis models
│   └── requirements.txt
│
├── web/                     ← browser interface (Flask)
│   ├── app.py
│   └── templates/index.html
│
├── data/                    ← all input/output data
│   ├── audio/                   put audio files here
│   ├── transcripts/             .txt + .segments.json
│   ├── reports/                 analysis output (.json)
│   └── profiles/                speaker profiles (.json)
│
└── models/                  ← cached ML models (~3 GB)
    ├── whisper/                 Faster-Whisper (transcription)
    ├── wav2vec2-emotion/        audio emotion
    ├── text-emotion/            text emotion (GoEmotions)
    ├── sentiment/               likes & dislikes
    └── zero-shot/               person type + profile metrics
```

---

## How it works

| Step | What happens                                                                        |
| ---- | ----------------------------------------------------------------------------------- |
| 1    | **Transcription** — Faster-Whisper converts speech to timestamped text segments     |
| 2    | **Audio emotion** — wav2vec2 reads the audio signal to detect tone of voice         |
| 3    | **Text emotion** — BERT (GoEmotions) classifies 27 emotions from the words          |
| 4    | **Likes / dislikes** — DistilBERT sentiment model finds positive/negative topics    |
| 5    | **Person type** — Zero-shot NLI model classifies what kind of person the speaker is |

All models run locally on CPU. A GPU is used automatically if available (MPS on Apple Silicon, CUDA on NVIDIA).

Models are cached in memory across analysis runs via `model_pool.py`, so the second+ analysis is dramatically faster.

---

## Models used

| Model                                                       | Task                         | Size    |
| ----------------------------------------------------------- | ---------------------------- | ------- |
| `Systran/faster-whisper-small`                              | Transcription                | ~244 MB |
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Audio emotion                | ~1.2 GB |
| `monologg/bert-base-cased-goemotions-original`              | Text emotion (27 labels)     | ~400 MB |
| `distilbert-base-uncased-finetuned-sst-2-english`           | Sentiment / likes & dislikes | ~67 MB  |
| `typeform/distilbert-base-uncased-mnli`                     | Person type (zero-shot)      | ~255 MB |
