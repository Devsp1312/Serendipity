================================================================================
  SERENDIPITY — STAGE 2: HOW IT WORKS
================================================================================


WHAT STAGE 2 DOES
-----------------
Stage 2 takes an audio file (MP3) and analyses it to answer 4 questions:

  1. What emotions are coming through in the tone of voice?
  2. What emotions are in the actual words being spoken?
  3. What does the speaker like or dislike?
  4. What kind of person is the speaker?

The result is saved as a JSON file in the "JSON files" folder.


--------------------------------------------------------------------------------
THE PIPELINE (step by step)
--------------------------------------------------------------------------------

  [1] TRANSCRIBE WITH TIMESTAMPS
      - Re-runs the same faster-whisper model from Stage 1
      - Produces a list of segments, each with: start time, end time, text
      - Example: { start: 0.0, end: 4.2, text: "This pizza is incredible" }
      - These timestamps are used to align the audio chunks with the words

  [2] AUDIO EMOTION
      - Decodes each audio segment from the MP3 using PyAV
      - Resamples from 44,100 Hz to 16,000 Hz (what the model expects)
      - If a segment is longer than 8 seconds, splits it into 4-second chunks
        (emotion models are trained on short utterances, not long clips)
      - Runs each chunk through the wav2vec2 emotion model
      - Averages all the scores weighted by duration
      - Output: which emotion dominates the speaker's tone of voice

  [3] TEXT EMOTION
      - Takes the transcribed text from each segment
      - Runs it through the GoEmotions text model
      - Uses sigmoid (not softmax) because multiple emotions can be true at once
      - Averages scores across all segments, weighted by word count
      - Output: which of the 27 emotions dominate the words being spoken

  [4] LIKES / DISLIKES
      - No model used — pure keyword matching
      - Splits the full transcript into sentences
      - Scans each sentence for positive signal words (love, amazing, best, etc.)
        and negative signal words (hate, awful, boring, terrible, etc.)
      - Extracts the topic (up to 5 words after the signal word)
      - Sentences with both positive and negative words are skipped (ambiguous)
      - Output: two lists — what the speaker likes and what they dislike

  [5] PERSON TYPE
      - Takes the full transcript text
      - Runs it through a zero-shot classification model
      - The model checks: "Does this text entail that the speaker is a [type]?"
      - Runs this check for all 10 person type labels independently
      - Returns the top 3 labels with their confidence scores
      - Output: what kind of person the speaker likely is


--------------------------------------------------------------------------------
THE MODELS
--------------------------------------------------------------------------------

  MODEL 1 — AUDIO EMOTION
  ~~~~~~~~~~~~~~~~~~~~~~~~
  Name:    superb/wav2vec2-base-superb-er
  Size:    ~361 MB
  Type:    Wav2Vec2 (Facebook speech model, fine-tuned for emotion)
  Trained: IEMOCAP dataset (acted emotional speech recordings)
  Task:    Speech Emotion Recognition — reads the sound, not the words

  Emotion labels (4):
    - neutral
    - happy
    - angry
    - sad

  How it works:
    Wav2Vec2 was pre-trained on 960 hours of audio to learn speech
    representations. It was then fine-tuned specifically to classify
    emotion from the sound of speech — things like pitch, pace, energy,
    and tone. It does NOT read the words at all, only the audio signal.


  MODEL 2 — TEXT EMOTION (27 labels)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Name:    monologg/bert-base-cased-goemotions-original
  Size:    ~400 MB
  Type:    BERT-base-cased, fine-tuned on Google's GoEmotions dataset
  Trained: 58,000 Reddit comments labelled by human raters
  Task:    Multi-label emotion classification from text

  Emotion labels (27):
    admiration    amusement      anger          annoyance
    approval      caring         confusion      curiosity
    desire        disappointment disapproval    disgust
    embarrassment excitement     fear           gratitude
    grief         joy            love           nervousness
    optimism      pride          realization    relief
    remorse       sadness        surprise       neutral

  How it works:
    BERT reads the full sentence and produces a score for each of the 27
    emotions independently (multi-label means multiple emotions can be
    true at the same time — e.g. a sentence can be both joyful and
    admiring). Scores are between 0 and 1 via sigmoid activation.


  MODEL 3 — PERSON TYPE (zero-shot)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Name:    typeform/distilbert-base-uncased-mnli
  Size:    ~255 MB
  Type:    DistilBERT fine-tuned on MNLI (natural language inference)
  Task:    Zero-shot classification — no retraining needed for new labels

  Person type labels (10):
    - food enthusiast
    - film critic
    - tech reviewer
    - sports fan
    - music lover
    - fitness and health conscious person
    - entrepreneur or business-minded person
    - academic or intellectual
    - lifestyle content creator
    - political commentator

  How it works:
    Natural Language Inference (NLI) models are trained to decide if one
    sentence "entails" another. Zero-shot classification exploits this by
    framing the question as:
      Premise:    [the transcript]
      Hypothesis: "This speaker is a food enthusiast."
    If the model says the premise entails the hypothesis → high score.
    This is run for all 10 labels independently (multi_label=True).
    No fine-tuning on speaker types was ever done — it generalises from
    the NLI training alone.


  MODEL 4 — TRANSCRIPTION (reused from Stage 1)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Name:    faster-whisper-small (Systran)
  Cached:  Device/model/
  Task:    Speech to text with timestamps
  Note:    Already downloaded in Stage 1, Stage 2 reuses it as-is


--------------------------------------------------------------------------------
LIKES / DISLIKES — SIGNAL WORDS
--------------------------------------------------------------------------------

  POSITIVE (→ likes):
    love, like, enjoy, great, amazing, fantastic, excellent, wonderful,
    best, perfect, spectacular, incredible, awesome, brilliant, favourite,
    favorite, appreciate, adore, obsessed, recommend, superb, outstanding,
    delicious, beautiful, stunning, impressive, genius, masterpiece

  NEGATIVE (→ dislikes):
    hate, dislike, don't like, boring, awful, terrible, horrible, bad,
    worst, poor, disappoint, frustrated, annoying, can't stand, avoid,
    waste, mediocre, overrated, underwhelming, bland, gross, disgusting,
    pathetic, ridiculous, nonsense


--------------------------------------------------------------------------------
OUTPUT — JSON STRUCTURE
--------------------------------------------------------------------------------

  {
    "file": "name of the audio file.mp3",
    "analysed_at": "2026-02-21T12:30:00",

    "transcript": {
      "segment_count": 38,          ← how many speech segments
      "total_duration_sec": 275.2,  ← length of the audio
      "language": "en",
      "language_confidence": 0.99
    },

    "audio_emotion": {
      "dominant": "happy",          ← strongest emotion in tone of voice
      "scores": {
        "neutral": 0.42,
        "happy":   0.38,
        "angry":   0.12,
        "sad":     0.08
      },
      "model": "superb/wav2vec2-base-superb-er"
    },

    "text_emotion": {
      "dominant": "joy",            ← strongest emotion in the words
      "scores": {
        "joy":            0.41,
        "excitement":     0.18,
        "admiration":     0.12,
        "approval":       0.09,
        ... (all 27 labels)
      },
      "model": "monologg/bert-base-cased-goemotions-original"
    },

    "likes": [
      "brick oven pizza",
      "cheesy crispy crust with zero flop"
    ],

    "dislikes": [
      "pizza places far from the city"
    ],

    "person_type": {
      "top_traits": [               ← top 3 most likely person types
        "food enthusiast",
        "lifestyle content creator",
        "casual entertainer"
      ],
      "scores": {                   ← all 10 labels with confidence
        "food enthusiast": 0.91,
        "lifestyle content creator": 0.74,
        ...
      },
      "model": "typeform/distilbert-base-uncased-mnli"
    }
  }


--------------------------------------------------------------------------------
THE VISUALISER
--------------------------------------------------------------------------------

  visualise.py reads a JSON file from the "JSON files" folder and generates a
  self-contained HTML report that opens automatically in your browser.
  No extra dependencies — uses only Python's built-in libraries.

  HOW IT WORKS
  ~~~~~~~~~~~~
  1. You pick a JSON file from the "JSON files" folder (same file picker
     style as transcribe.py and analyse.py)
  2. It builds a single HTML file with all CSS inline — no internet needed,
     no external files, works completely offline
  3. Saves the HTML to the "Reports" folder at the project root
  4. Opens the report in your default browser automatically

  WHAT THE HTML REPORT SHOWS (5 sections)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Header
    - Audio file name, date analysed, duration, segment count, language

  Audio Emotion  (tone of voice)
    - 4 horizontal bars, one per emotion, sorted highest first
    - Dominant emotion is bold with a "dominant" badge
    - Colour coded:
        happy   → amber  (#f59e0b)
        neutral → grey   (#6b7280)
        angry   → red    (#ef4444)
        sad     → blue   (#3b82f6)

  Text Emotion  (the words)
    - Top 10 of the 27 GoEmotions labels, sorted highest first
    - Top 3 bars are darker purple, rest are lighter purple
    - Shows how many more labels exist below the top 10

  Likes & Dislikes
    - Two columns side by side
    - Likes in green  ✓    Dislikes in red  ✗
    - "None detected" shown if the list is empty

  Person Type
    - All 10 person type labels as bars, sorted highest first
    - Top 3 traits get a ★ star and green bars
    - Remaining 7 are shown in slate grey


--------------------------------------------------------------------------------
FILE STRUCTURE
--------------------------------------------------------------------------------

  Serendipity/
  ├── Audio files/          ← MP3 input files go here
  ├── Text files/           ← Stage 1 transcripts (.txt) saved here
  ├── JSON files/           ← Stage 2 analysis output (.json) saved here
  ├── Reports/              ← HTML visual reports saved here (one per audio file)
  ├── Device/
  │   ├── model/            ← Stage 1 faster-whisper model cache
  │   ├── transcribe.py     ← Stage 1: transcribe audio → text file
  │   ├── download_model.py ← Stage 1: download faster-whisper model
  │   ├── requirements.txt  ← Stage 1 dependencies
  │   └── Stage 2/
  │       ├── analyse.py         ← Stage 2: run this to analyse audio
  │       ├── visualise.py       ← Stage 2: run this to view results in browser
  │       ├── download_models.py ← Stage 2: download all 3 models
  │       ├── requirements.txt   ← Stage 2 dependencies
  │       └── models/            ← Stage 2 model cache (created on download)
  │           ├── wav2vec2-emotion/
  │           ├── text-emotion/
  │           └── zero-shot/
  ├── Server Backend/       ← future
  ├── QR reader/            ← future
  └── Stage 2 - How It Works.txt  ← this file


--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

  First time setup:
    pip install faster-whisper torch transformers av
    python "Device/Stage 2/download_models.py"    ← downloads ~1 GB of models

  Every time:
    python "Device/Stage 2/analyse.py"            ← pick a file, get JSON output
    python "Device/Stage 2/visualise.py"          ← pick a JSON, open HTML report in browser


================================================================================
