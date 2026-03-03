"""
app.py — Serendipity web interface.

Runs a local Flask server so you can select audio files, run analysis,
and view reports in your browser — no terminal needed.

Usage:   python app.py
Browser: http://localhost:5000
"""

import atexit
import json
import queue
import socket
import sys
import threading
import time
import traceback

import psutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from faster_whisper import WhisperModel

# Add the analysis folder to the path so we can import the analysis modules
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from model_pool import ModelPool
from audio_emotion  import analyse_audio_emotion
from text_emotion   import analyse_text_emotion
from likes_dislikes import extract_likes_dislikes
from person_type    import classify_person_type
import profiles as profile_module

# Clean up cached models on shutdown
atexit.register(lambda: ModelPool.get().shutdown())

# ─── App + paths ──────────────────────────────────────────────────────────────

app = Flask(__name__)

ROOT          = Path(__file__).parent.parent          # Serendipity/
AUDIO_DIR     = ROOT / "data" / "audio"
JSON_DIR      = ROOT / "data" / "reports"
SEGMENTS_DIR  = ROOT / "data" / "transcripts"
WHISPER_CACHE = ROOT / "models" / "whisper"

AUDIO_EXTENSIONS = ["*.mp3", "*.m4a", "*.wav", "*.aac", "*.flac"]
PROFILES_DIR      = ROOT / "data" / "profiles"

# Make sure output directories exist before anything tries to write to them
JSON_DIR.mkdir(exist_ok=True)
SEGMENTS_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)

# Lock so only one analysis run and one profile update can run at a time.
# Fix 7: profile metric computation also loads the zero-shot model, so we
# protect it with its own flag to prevent two heavy model loads simultaneously.
_busy             = False
_busy_lock        = threading.Lock()
_profile_busy     = False
_profile_busy_lock = threading.Lock()


# ─── Segment helper (mirrors analyse.py get_segments) ─────────────────────────

def get_segments(audio_path: Path):
    """
    Returns (segments, info, source).

    Fast path — Stage 1 cache found:
      Reads the .segments.json saved by transcribe.py. No model loaded.

    Fallback — Stage 1 hasn't run yet:
      Runs faster-whisper automatically and saves both .txt and .segments.json
      so future calls use the fast path. source = 'transcribed'.
    """
    seg_file = SEGMENTS_DIR / (audio_path.stem + ".segments.json")

    if seg_file.exists():
        data = json.loads(seg_file.read_text(encoding="utf-8"))
        info = SimpleNamespace(
            duration=data["total_duration_sec"],
            language=data["language"],
            language_probability=data["language_confidence"],
        )
        return data["segments"], info, "cached"

    # Stage 1 not run — transcribe now
    snapshots = list(WHISPER_CACHE.glob("models--Systran--faster-whisper-small/snapshots/*"))
    model_src = str(snapshots[0]) if snapshots else "small"
    model = WhisperModel(model_src, device="cpu", compute_type="int8")
    # Fix 6: wrap transcribe() in try/finally so the model is always deleted
    # even if an exception is raised (e.g. corrupt file, unexpected EOF).
    try:
        raw_segments, info = model.transcribe(str(audio_path))
        segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in raw_segments]
    finally:
        del model

    full_text = " ".join(s["text"] for s in segments)
    (SEGMENTS_DIR / audio_path.with_suffix(".txt").name).write_text(full_text, encoding="utf-8")

    seg_data = {
        "file": audio_path.name,
        "language": info.language,
        "language_confidence": round(info.language_probability, 3),
        "total_duration_sec": round(info.duration, 1),
        "segments": segments,
    }
    seg_file.write_text(json.dumps(seg_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return segments, info, "transcribed"


# ─── Pipeline runner (runs in background thread) ──────────────────────────────

def run_pipeline(audio_path: Path, q: queue.Queue):
    """
    Runs all 5 analysis steps in sequence.
    Posts a 'step' event when each step starts, an 'ok' event when it finishes.
    Posts 'done' with the full JSON output at the end, or 'error' on failure.
    The global _busy flag is cleared in the finally block.
    """
    global _busy
    try:
        # ── 1. Segments ──────────────────────────────────────────────────────
        q.put({"type": "step", "n": 1, "msg": "Loading transcript segments …"})
        segments, info, source = get_segments(audio_path)
        # Improvement 1: catch silent / too-short audio before the downstream
        # models receive empty input and produce cryptic errors.
        if not segments:
            raise ValueError(
                "No speech detected — audio may be silent or too short to transcribe."
            )
        full_text = " ".join(s["text"] for s in segments)
        src_label = "loaded from Stage 1 cache" if source == "cached" else "auto-transcribed"
        q.put({"type": "ok", "n": 1,
               "msg": f"{len(segments)} segments · {info.duration:.0f}s · {src_label}"})

        # Pipeline order: smallest models first for faster perceived progress.
        # Likes (67 MB) → Person type (255 MB) → Text emotion (400 MB) → Audio emotion (1.2 GB)

        # ── 2. Likes / dislikes (67 MB model — fastest) ─────────────────────
        q.put({"type": "step", "n": 2, "msg": "Extracting likes and dislikes …"})
        likes_dislikes = extract_likes_dislikes(segments)
        q.put({"type": "ok", "n": 2,
               "msg": f"{len(likes_dislikes['likes'])} likes · {len(likes_dislikes['dislikes'])} dislikes"})

        # ── 3. Person type (255 MB model) ────────────────────────────────────
        q.put({"type": "step", "n": 3, "msg": "Classifying person type …"})
        person_type = classify_person_type(full_text)
        q.put({"type": "ok", "n": 3,
               "msg": f"Top traits: {', '.join(person_type['top_traits'])}"})

        # ── 4. Text emotion (400 MB model) ───────────────────────────────────
        q.put({"type": "step", "n": 4, "msg": "Analysing text emotion (27 labels) …"})
        text_emotion = analyse_text_emotion(segments)
        q.put({"type": "ok", "n": 4, "msg": f"Dominant: {text_emotion['dominant']}"})

        # ── 5. Audio emotion (1.2 GB model — heaviest) ──────────────────────
        q.put({"type": "step", "n": 5, "msg": "Analysing audio emotion (tone of voice) …"})
        audio_emotion = analyse_audio_emotion(audio_path, segments)
        q.put({"type": "ok", "n": 5, "msg": f"Dominant: {audio_emotion['dominant']}"})

        # ── Save JSON ────────────────────────────────────────────────────────
        output = {
            "file": audio_path.name,
            "analysed_at": datetime.now().isoformat(timespec="seconds"),
            "transcript": {
                "segment_count": len(segments),
                "total_duration_sec": round(info.duration, 1),
                "language": info.language,
                "language_confidence": round(info.language_probability, 3),
            },
            "audio_emotion":  audio_emotion,
            "text_emotion":   text_emotion,
            **likes_dislikes,
            "person_type":    person_type,
        }
        out_path = JSON_DIR / audio_path.with_suffix(".json").name
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        q.put({"type": "done", "data": output})

    except Exception as e:
        q.put({"type": "error", "msg": str(e), "detail": traceback.format_exc()})
    finally:
        with _busy_lock:
            _busy = False


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serves the main single-page web UI."""
    return render_template("index.html")


@app.route("/api/files")
def api_files():
    """
    Lists all MP3 files in the Audio files folder.
    Each entry includes name, size, and status flags (transcribed / analysed).
    """
    if not AUDIO_DIR.exists():
        return jsonify([])

    all_files = []
    for ext in AUDIO_EXTENSIONS:
        all_files.extend(AUDIO_DIR.glob(ext))
    files = []
    for f in sorted(set(all_files), key=lambda p: p.name.lower()):
        files.append({
            "name":        f.name,
            "size_mb":     round(f.stat().st_size / (1024 * 1024), 1),
            "transcribed": (SEGMENTS_DIR / (f.stem + ".segments.json")).exists(),
            "analysed":    (JSON_DIR / f.with_suffix(".json").name).exists(),
        })
    return jsonify(files)


@app.route("/api/report/<path:filename>")
def api_report(filename):
    """
    Returns the JSON analysis report for a given audio file.
    Accepts either the mp3 name (e.g. audio.mp3) or just the stem (audio).
    """
    stem = Path(filename).stem
    path = JSON_DIR / (stem + ".json")
    if not path.exists():
        return jsonify({"error": "No report found for this file"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/status")
def api_status():
    """Returns whether an analysis is currently running."""
    return jsonify({"busy": _busy})


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    Starts the full analysis pipeline for the given file.
    Streams progress back to the browser using Server-Sent Events (SSE).

    Each event is a JSON object on a 'data:' line:
      data: {"type": "step", "n": 2, "msg": "Analysing audio emotion …"}
      data: {"type": "ok",   "n": 2, "msg": "Dominant: happy"}
      data: {"type": "done", "data": { ...full output dict... }}
      data: {"type": "error","msg": "...", "detail": "traceback..."}
    """
    global _busy

    filename   = (request.json or {}).get("file", "")
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404

    with _busy_lock:
        if _busy:
            return jsonify({"error": "An analysis is already running — please wait"}), 409
        _busy = True

    q: queue.Queue = queue.Queue()
    threading.Thread(target=run_pipeline, args=(audio_path, q), daemon=True).start()

    def generate():
        """Reads events from the queue and yields them as SSE lines.
        Sends a heartbeat ping every 15 s so the connection stays alive while
        heavy models are loading (wav2vec2 is ~1.2 GB and can take minutes).
        """
        deadline = time.time() + 1800   # 30-minute absolute limit
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except queue.Empty:
                if time.time() > deadline:
                    yield f"data: {json.dumps({'type': 'error', 'msg': 'Timed out'})}\n\n"
                    break
                # Heartbeat — keeps proxy / browser from closing idle SSE connection
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Profile routes ───────────────────────────────────────────────────────────

@app.route("/api/profiles")
def api_profiles_list():
    """Returns a list of all profiles with name and session count."""
    return jsonify(profile_module.list_profiles())


@app.route("/api/profiles", methods=["POST"])
def api_profiles_create():
    """Creates an empty profile. Body: {"name": "..."}"""
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Profile name is required"}), 400
    if profile_module.load_profile(name) is not None:
        return jsonify({"error": f"Profile '{name}' already exists"}), 409
    stub = {
        "name":               name,
        "created_at":         datetime.now().isoformat(timespec="seconds"),
        "last_updated":       "",
        "session_count":      0,
        "total_duration_sec": 0.0,
        "sessions":           [],
        "metrics":            {},
        "emotion_profile":    {},
        "interests":          {"likes": [], "dislikes": []},
    }
    profile_module.save_profile(stub)
    return jsonify(stub), 201


@app.route("/api/profile/<path:name>")
def api_profile_get(name):
    """Returns the full profile JSON for a given name."""
    data = profile_module.load_profile(name)
    if data is None:
        return jsonify({"error": f"Profile '{name}' not found"}), 404
    return jsonify(data)


@app.route("/api/profile/<path:name>/add-session", methods=["POST"])
def api_profile_add_session(name):
    """
    Adds an audio session to a profile (creating the profile if it doesn't
    exist) and recomputes all 19 metrics. Streams progress via SSE.

    Body: {"file": "audio.mp3"}

    Events:
      data: {"type": "step", "msg": "Loaded 2 sessions…"}
      data: {"type": "done", "data": {...full profile...}}
      data: {"type": "error", "msg": "..."}
    """
    global _profile_busy

    filename = (request.json or {}).get("file", "")
    if not filename:
        return jsonify({"error": "No file specified"}), 400

    # Fix 7: prevent concurrent profile metric computation (each loads the
    # zero-shot model; two concurrent loads would fill RAM / VRAM).
    with _profile_busy_lock:
        if _profile_busy:
            return jsonify({"error": "A profile update is already running — please wait"}), 409
        _profile_busy = True

    q: queue.Queue = queue.Queue()

    def _run():
        global _profile_busy
        try:
            def cb(msg):
                q.put({"type": "step", "msg": msg})
            result = profile_module.add_session(name, filename, progress_cb=cb)
            q.put({"type": "done", "data": result})
        except FileNotFoundError as e:
            q.put({"type": "error", "msg": str(e)})
        except Exception as e:
            q.put({"type": "error", "msg": str(e), "detail": traceback.format_exc()})
        finally:
            with _profile_busy_lock:
                _profile_busy = False

    threading.Thread(target=_run, daemon=True).start()

    def generate():
        """Streams profile-update progress with the same heartbeat pattern."""
        deadline = time.time() + 1800
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except queue.Empty:
                if time.time() > deadline:
                    yield f"data: {json.dumps({'type': 'error', 'msg': 'Timed out'})}\n\n"
                    break
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/profile/<path:name>/reset", methods=["POST"])
def api_profile_reset(name):
    """Resets a profile to its initial empty state, keeping name and created_at."""
    result = profile_module.reset_profile(name)
    if result is None:
        return jsonify({"error": f"Profile '{name}' not found"}), 404
    return jsonify(result)


@app.route("/api/profile/<path:name>", methods=["DELETE"])
def api_profile_delete(name):
    """Deletes a profile by name."""
    if profile_module.delete_profile(name):
        return jsonify({"ok": True})
    return jsonify({"error": f"Profile '{name}' not found"}), 404


# ─── Port helpers ─────────────────────────────────────────────────────────────

_PREFERRED_PORT = 5000


def _kill_port(port: int) -> bool:
    """
    Finds whatever process is holding the given port and terminates it.
    Uses psutil — works on macOS, Linux, and Windows (no lsof required).
    Waits 0.8s after killing so the OS can release the port before we retry.
    Returns True if something was killed, False if the port was already free.
    """
    killed = False
    try:
        for proc in psutil.process_iter(["pid", "connections"]):
            try:
                for conn in proc.info["connections"] or []:
                    if conn.laddr.port == port:
                        proc.terminate()
                        killed = True
                        # Improvement 3: no break — let the outer loop continue
                        # so every process bound to this port gets terminated
                        # (rare but possible if multiple processes share a port).
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    if killed:
        time.sleep(0.8)   # give the OS time to release the port
    return killed


def _find_free_port(preferred: int) -> int:
    """
    Tries to bind to `preferred`. If it's in use, scans preferred+1 … preferred+10
    and returns the first port that is free.
    SO_REUSEADDR is set so a port in TIME_WAIT state after a recent close is
    still considered available.
    Falls back to `preferred` if nothing works (Flask will then show its own error).
    """
    for port in range(preferred, preferred + 11):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


# ─── Model health check ───────────────────────────────────────────────────────

def _check_models() -> None:
    """
    Improvement 2: verify that all four analysis model directories exist and
    are non-empty before Flask starts.  Prints a clear warning if any are
    missing so the user knows to run download_models.py before running analysis.
    """
    models_dir = Path(__file__).parent.parent / "models"
    required = {
        "wav2vec2-emotion": "Audio emotion (wav2vec2-lg-xlsr)",
        "text-emotion":     "Text emotion (GoEmotions BERT)",
        "sentiment":        "Sentiment / likes-dislikes (DistilBERT)",
        "zero-shot":        "Person type / profile metrics (zero-shot NLI)",
    }
    missing = []
    for dirname, label in required.items():
        d = models_dir / dirname
        if not d.exists() or not any(d.iterdir()):
            missing.append(f"     ⚠  {dirname}/ — {label}")
    if missing:
        print("  Models not found (analysis will fail until downloaded):")
        for m in missing:
            print(m)
        print("  Fix: python analysis/download_models.py\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Respect PORT env var when set (e.g. by preview_start / CI).
    # Otherwise use the preferred port with the existing kill-and-retry logic.
    env_port = os.environ.get("PORT")
    if env_port:
        port = int(env_port)
    else:
        port = _find_free_port(_PREFERRED_PORT)

        if port != _PREFERRED_PORT:
            # Preferred port is busy — try to kill whatever is holding it,
            # then check again. If the kill frees it, go back to 5000.
            print(f"  Port {_PREFERRED_PORT} is busy — freeing it …")
            if _kill_port(_PREFERRED_PORT):
                port = _find_free_port(_PREFERRED_PORT)

    _check_models()   # Improvement 2: warn early if models are missing

    print(f"\n  ◆  Serendipity is running")
    print(f"     Open: http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
