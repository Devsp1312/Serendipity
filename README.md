# Serendipity

A fully local audio analysis pipeline that transcribes any audio file and tells you:

- 🎙️ **Audio emotion** — what emotion comes through in the speaker's tone of voice
- 💬 **Text emotion** — what emotions are in the actual words (27 labels)
- 👍 **Likes & dislikes** — what the speaker is positive or negative about
- 🧠 **Person type** — what kind of person the speaker likely is

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

**2. Install Stage 1 dependencies** (transcription)
```bash
pip install -r stage1/requirements.txt
```

**3. Download the Whisper transcription model**
```bash
python stage1/download_model.py
```

**4. Install Stage 2 dependencies** (analysis + web UI)
```bash
pip install -r stage2/requirements.txt
```

**5. Download the analysis models** (~2 GB, one-time)
```bash
python stage2/download_models.py
```

---

## Usage

### Web UI (recommended)

```bash
python stage2/app.py
```

Then open **http://localhost:5000** in your browser.

- Drop audio files into the `Audio files/` folder
- Click a file in the sidebar to select it
- Click **Analyse** to run the full pipeline
- Watch live progress and view the report when done

### Command line

Run Stage 1 first to transcribe:
```bash
python stage1/transcribe.py
```

Then run Stage 2 to analyse:
```bash
python stage2/analyse.py
```

Results are saved to `JSON files/`. To open a visual HTML report:
```bash
python stage2/visualise.py
```

---

## Supported audio formats

`.mp3` · `.m4a` · `.wav` · `.aac` · `.flac`

---

## Project structure

```
Serendipity/
├── Audio files/          ← put your audio files here
├── Text files/           ← transcripts saved here (.txt + .segments.json)
├── JSON files/           ← analysis output saved here (.json)
├── Reports/              ← HTML visual reports saved here
│
├── stage1/
│   ├── transcribe.py         ← transcribe audio → text
│   ├── download_model.py     ← download Whisper model
│   ├── requirements.txt
│   └── model/                ← Whisper model cache (created on download)
│
└── stage2/
    ├── app.py                ← web UI (Flask)
    ├── analyse.py            ← command-line analysis
    ├── visualise.py          ← generate HTML report from JSON
    ├── download_models.py    ← download all analysis models
    ├── requirements.txt
    ├── templates/
    │   └── index.html        ← web UI frontend
    └── models/               ← analysis model cache (created on download)
        ├── wav2vec2-emotion/
        ├── text-emotion/
        ├── sentiment/
        └── zero-shot/
```

---

## How it works

| Step | What happens |
|------|-------------|
| 1 | **Transcription** — Faster-Whisper converts speech to timestamped text segments |
| 2 | **Audio emotion** — wav2vec2 reads the audio signal to detect tone of voice |
| 3 | **Text emotion** — BERT (GoEmotions) classifies 27 emotions from the words |
| 4 | **Likes / dislikes** — DistilBERT sentiment model finds positive/negative topics |
| 5 | **Person type** — Zero-shot NLI model classifies what kind of person the speaker is |

All models run locally on CPU. A GPU is used automatically if available (MPS on Apple Silicon, CUDA on NVIDIA).

---

## Models used

| Model | Task | Size |
|-------|------|------|
| `Systran/faster-whisper-small` | Transcription | ~244 MB |
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Audio emotion | ~1.2 GB |
| `monologg/bert-base-cased-goemotions-original` | Text emotion (27 labels) | ~400 MB |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment / likes & dislikes | ~67 MB |
| `typeform/distilbert-base-uncased-mnli` | Person type (zero-shot) | ~255 MB |
