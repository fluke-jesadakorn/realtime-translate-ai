"""
Realtime Speaker Translator v2 — BlackHole 2ch → STT → translate → web GUI.

Optimizations (P1+P2+P3) — Nov 2026:
  P1 — Whisper large-v3-turbo default + --prompt + --suppress-regex + threads=8
     — Ollama options: temperature 0, seed 42, num_predict 256, repeat_penalty 1.1
     — Language-pair translation prompts (DE→TH / EN→TH / JA→TH / generic)
     — Glossary + custom prompt support (editable from GUI)
     — Default model priority: gemma > llama3 > mistral > qwen
  P2 — VAD chunking (webrtcvad, 30ms frames, aggressiveness=2)
     — Overlap 1s between consecutive chunks for sentence-boundary continuity
     — Noise reduction (noisereduce, stationary, prop_decrease=1.0)
     — 2-stage pipeline: STT thread + LLM thread (concurrent)
     — Translation memory: last 3 segments fed to LLM as context
     — Output validation + 1 retry on validation failure
  P3 — MLX-Whisper backend (auto-detected, falls back to whisper-cli)
     — MLX-LM backend (auto-detected, falls back to Ollama)
     — Backend diagnostics at /api/backends, prompt preview at /api/prompts

Pipeline:
  system audio (BlackHole 2ch)
    → noisereduce (denoise)
    → webrtcvad (drop silent chunks)
    → 5s chunks + 1s overlap
    → whisper STT (--prompt + --suppress-regex + threads=8)
    → context + memory block
    → translate (temperature 0, seed 42, retry on bad output)
    → validate
    → SSE → live web UI

First run auto-creates .venv and installs deps (sounddevice, numpy, scipy,
webrtcvad, noisereduce). Optional: pip install mlx-whisper mlx-lm to unlock
Apple Neural Engine acceleration (auto-detected at runtime).

Run:   python3 realtime_translator.py
Stop:  Ctrl+C
"""

import os
import sys
import time
import json
import queue
import wave
import threading
import subprocess
import urllib.request
import http.server
import webbrowser
from datetime import datetime

# ============================================================
# Bootstrap: create venv + install deps on first run
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
IN_VENV = sys.prefix.startswith(VENV_DIR) if os.path.exists(VENV_DIR) else False

REQUIRED_PKGS = [
    # core
    "sounddevice", "numpy", "scipy",
    # P2: VAD + denoise
    "webrtcvad", "noisereduce",
    # P3 (optional, auto-detected): MLX backends on Apple Silicon
    #   pip install mlx-whisper mlx-lm  (requires Python 3.10+)
]


# Preferred Python interpreter — newest Homebrew Python that exists on the system.
# MLX requires Python >= 3.10, and mlx-lm/mlx-whisper wheels are available up to 3.14.
PYTHON_CANDIDATES = [
    "/opt/homebrew/bin/python3.14",
    "/opt/homebrew/bin/python3.13",
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/bin/python3.11",
    os.path.expanduser("~/.local/bin/python3.11"),
    "/opt/homebrew/bin/python3.10",
    "/usr/bin/python3",
]


def pick_python():
    """Return absolute path to the best available Python interpreter (3.10+ preferred)."""
    for p in PYTHON_CANDIDATES:
        if os.path.exists(p):
            try:
                # quick version check via subprocess (cheap)
                out = subprocess.check_output(
                    [p, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    stderr=subprocess.DEVNULL, timeout=5,
                ).decode().strip()
                major, minor = (int(x) for x in out.split("."))
                if (major, minor) >= (3, 10):
                    return p
            except Exception:
                continue
    return sys.executable  # last resort — whatever launched us


def bootstrap_venv():
    if IN_VENV:
        # Quick importability check — only run pip if a required module is missing.
        missing = []
        for mod in ("sounddevice", "numpy", "scipy", "webrtcvad", "noisereduce"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if missing:
            print(f"⚠ Missing packages: {', '.join(missing)} — installing…")
            py = sys.executable
            try:
                subprocess.check_call(
                    [py, "-m", "pip", "install", "--quiet", *REQUIRED_PKGS],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"⚠ pip install of required packages failed: {e}")
        return
    if not os.path.exists(VENV_DIR):
        # Choose best Python available (3.10+ for MLX)
        py_for_venv = pick_python()
        py_ver = subprocess.check_output(
            [py_for_venv, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        print(f"🔧 First run: creating virtual environment at .venv using Python {py_ver} ({py_for_venv}) …")
        subprocess.check_call(
            [py_for_venv, "-m", "venv", VENV_DIR],
            stdout=subprocess.DEVNULL,
        )
        pip = os.path.join(VENV_DIR, "bin", "pip")
        # Ensure pip is recent enough for PEP 517 builds (MLX wheels need it)
        print("📦 Upgrading pip …")
        subprocess.check_call([pip, "install", "--quiet", "--upgrade", "pip"])
        print(f"📦 Installing: {', '.join(REQUIRED_PKGS)} …")
        subprocess.check_call([pip, "install", "--quiet", *REQUIRED_PKGS])
    py = os.path.join(VENV_DIR, "bin", "python")
    print(f"🚀 Restarting inside venv: {py}\n")
    os.execv(py, [py] + sys.argv)

bootstrap_venv()

# now we can import the heavy deps
import re  # noqa: E402
from collections import deque  # noqa: E402
import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import webrtcvad  # noqa: E402  (P2 — VAD for sentence-boundary chunking)
import noisereduce as nr  # noqa: E402  (P2 — spectral denoise)
# P3 — MLX backends (optional, auto-detected at runtime)
try:
    import mlx_whisper  # noqa: E402
    HAS_MLX_WHISPER = True
except Exception:
    mlx_whisper = None
    HAS_MLX_WHISPER = False
try:
    from mlx_lm import load as mlx_lm_load, generate as mlx_lm_generate  # noqa: E402
    HAS_MLX_LM = True
except Exception:
    mlx_lm_load = mlx_lm_generate = None
    HAS_MLX_LM = False

# ============================================================
# Config
# ============================================================

SAMPLERATE       = 16000
CHANNELS         = 1
CHUNK_DURATION   = 5          # seconds per STT chunk
CHUNK_OVERLAP    = 0.0        # P2 — overlap between chunks (seconds). 0=no overlap, 1.0=1s.
                       # NOTE: overlap >0 currently produces duplicate transcript output
                       # for the overlap region (~1s of text appears twice). Enable after
                       # adding suffix-overlap dedup post-processing.
VAD_AGGR         = 2          # P2 — webrtcvad aggressiveness 0-3 (2 = balanced)
VAD_FRAME_MS     = 30         # P2 — VAD frame size in ms
VAD_MIN_SPEECH   = 0.4        # P2 — seconds; skip chunks with less speech than this
DENOISE_PROPS    = {          # P2 — noisereduce params (per-chunk stationary noise)
    "stationary": True,
    "prop_decrease": 1.0,
}
WHISPER_BIN      = "/opt/homebrew/bin/whisper-cli"
WHISPER_THREADS  = 8          # M4 has 10 cores — leave 2 for UI/audio
SUPPRESS_REGEX   = r"\b([dD]anke fürs zuschauen|[uU]ntertitel|[sS]ubtitles|[mM]usik|[aA]mara|字幕組)\b"
HALLUC_TAGS      = ["[MUSIK]", "[musik]", "[ Music ]", "[音楽]", "[비명]",
                    "[박수]", "[拍手]", "(Lebhafte Musik)", "(Music)"]
# Default whisper prompt primes German → meeting/tech context
DEFAULT_WHISPER_PROMPT = (
    "Ein Gespräch auf Deutsch in einem beruflichen Meeting. "
    "Häufige Begriffe: API, Deployment, Sprint, Backlog, Stakeholder, "
    "Meeting, Roadmap, Feature, Release, Review. "
    "Sprecher: männlich und weiblich, formelle und informelle Redeweise."
)
TRANSLATION_MEMORY_SIZE = 5   # P2 — last N segments fed to LLM as context (increase history size)

# auto-detect whisper model in several common locations
_WHISPER_CANDIDATES = [
    os.path.expanduser("~/whisper-models/ggml-base.bin"),
    os.path.join(SCRIPT_DIR, "ggml-base.bin"),            # legacy
    "/opt/homebrew/share/whisper-cpp/for-tests-ggml-tiny.bin",  # brew fallback (tiny)
    os.path.expanduser("~/models/ggml-base.bin"),
]
WHISPER_MODEL = next((p for p in _WHISPER_CANDIDATES if os.path.exists(p)), _WHISPER_CANDIDATES[0])

DEFAULT_DEVICE   = "BlackHole 2ch"
OLLAMA_URL       = "http://localhost:11434"
HOST             = "127.0.0.1"
PORT             = 8765        # fixed port — change here if you need different

LANGUAGES = [
    "English", "Thai", "Japanese", "Chinese (Simplified)", "Chinese (Traditional)",
    "Korean", "French", "German", "Spanish", "Portuguese", "Russian",
    "Italian", "Vietnamese", "Indonesian", "Arabic", "Hindi", "Burmese",
    "Lao", "Khmer", "Malay", "Turkish", "Dutch", "Polish", "Swedish",
]
LANG_CODE = {
    "English": "en", "Thai": "th", "Japanese": "ja",
    "Chinese (Simplified)": "zh", "Chinese (Traditional)": "zh",
    "Korean": "ko", "French": "fr", "German": "de",
    "Spanish": "es", "Portuguese": "pt", "Russian": "ru",
    "Italian": "it", "Vietnamese": "vi", "Indonesian": "id",
    "Arabic": "ar", "Hindi": "hi", "Burmese": "my",
    "Lao": "lo", "Khmer": "km", "Malay": "ms",
    "Turkish": "tr", "Dutch": "nl", "Polish": "pl", "Swedish": "sv",
}

# ============================================================
# State (thread-safe)
# ============================================================

_state_lock = threading.Lock()
state = {
    "listening":     False,
    "device":        DEFAULT_DEVICE,
    "src_lang":      "German",
    "tgt_lang":      "Thai",
    "model":         "",
    "wmodel":        WHISPER_MODEL,    # path to whisper ggml model
    "history":       [],               # list of {ts, src, tgt, src_lang, tgt_lang}
    "level":         0.0,
    "last_error":    "",
    "whisper_busy":  False,
    "llm_busy":      False,
    # P1 — always-on audio processing (no UI toggle)
    #   denoise    — noisereduce on every chunk
    #   vad_chunk  — webrtcvad drops pure-silence chunks
    #   use_memory — last 3 segments fed to LLM as context
    "glossary":      "",               # user-defined terms: "Krakenversicherung=Krankenversicherung, M2024=Meeting 2024"
    "custom_prompt": "",               # optional full override of translation system prompt
    "concise_translation": False,      # whether to shorten translation to reduce latency
    "use_mlx_whisper": False,          # use in-process mlx_whisper if available
    "use_mlx_lm":      False,          # use in-process mlx_lm if available (set to False to use Ollama by default)
}

# P2 — 2-stage pipeline: audio_q → stt_thread → llm_q → llm_thread → SSE
audio_q: "queue.Queue" = queue.Queue(maxsize=16)   # raw float32 chunks from audio_thread
llm_q:   "queue.Queue" = queue.Queue(maxsize=8)    # dicts with src text + meta for llm_thread
sse_clients: "set[queue.Queue]" = set()
clients_lock = threading.Lock()

# Cached whisper prompt per language pair (refreshed when src_lang changes)
_whisper_prompt_cache = {"src": None, "prompt": DEFAULT_WHISPER_PROMPT}

# Cached translation prompt per language pair (refreshed when src/tgt changes)
_translation_prompt_cache = {"key": None, "prompt": ""}

# Cached MLX-Whisper loaded model (lazy, in-process)
# Tracks: which whisper basename is using MLX (hf_repo), which has failed once.
_mlx_whisper_model = {
    "basename": None,    # the resolved whisper model name (e.g. "large-v3-turbo")
    "hf_repo": None,     # the resolved HF repo id (e.g. "mlx-community/whisper-large-v3-turbo")
    "loaded": False,     # is the model loaded in memory?
    "failed": set(),     # basenames we've already tried and failed — don't retry
}

# Explicit mapping: whisper model basename (no ggml-, no .bin) → mlx-community HF repo.
# mlx-community's naming is inconsistent — some have -mlx suffix, some don't.
_MLX_WHISPER_REPOS = {
    "tiny":          "mlx-community/whisper-tiny-mlx",
    "tiny.en":       "mlx-community/whisper-tiny.en-mlx",
    "base":          "mlx-community/whisper-base-mlx",
    "base.en":       "mlx-community/whisper-base.en-mlx",
    "small":         "mlx-community/whisper-small-mlx",
    "small.en":      "mlx-community/whisper-small.en-mlx",
    "medium":        "mlx-community/whisper-medium-mlx",
    "medium.en":     "mlx-community/whisper-medium.en-mlx",
    "large":         "mlx-community/whisper-large-mlx",
    "large-v1":      "mlx-community/whisper-large-v1-mlx",
    "large-v2":      "mlx-community/whisper-large-v2-mlx",
    "large-v3":      "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",  # ← no -mlx suffix!
}


def _whisper_basename_to_mlx_repo(basename):
    """Resolve whisper model basename → mlx-community HF repo.
    Falls back to pattern inference, then to as-is."""
    if basename in _MLX_WHISPER_REPOS:
        return _MLX_WHISPER_REPOS[basename]
    # Generic fallback: assume <basename>-mlx
    return f"mlx-community/whisper-{basename}-mlx"

# Cached MLX-LM loaded model + tokenizer (lazy, in-process)
_mlx_lm_cache = {"name": None, "loaded": False, "model_obj": None, "tokenizer": None}


def _public_state():
    with _state_lock:
        s = dict(state)
    s.pop("level", None)
    s["backends"] = {
        "vad":       True,                            # webrtcvad (required)
        "denoise":   True,                            # noisereduce (required)
        "mlx_whisper": HAS_MLX_WHISPER and s.get("use_mlx_whisper", True),
        "mlx_lm":    HAS_MLX_LM and s.get("use_mlx_lm", True),
        "whisper_threads": WHISPER_THREADS,
        "memory_size": TRANSLATION_MEMORY_SIZE,
    }
    return s


def broadcast(event, data):
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with clients_lock:
        targets = list(sse_clients)
    for q in targets:
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


# ============================================================
# Audio / STT / Translation
# ============================================================

def save_wav(path, data):
    int_data = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLERATE)
        w.writeframes(int_data.tobytes())


def find_device(name):
    try:
        return int(sd.query_devices(name, "input")["index"])
    except Exception:
        return None


def list_input_devices():
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append({
                    "index": i, "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sr": d.get("default_samplerate", 0),
                })
    except Exception:
        pass
    return out


def fetch_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def list_whisper_models():
    """Find all ggml-*.bin files in script dir + standard locations."""
    out = []
    seen = set()
    candidates = [
        os.path.expanduser("~/whisper-models"),
        SCRIPT_DIR,
        os.path.expanduser("~/models"),
    ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("ggml-") and f.endswith(".bin"):
                p = os.path.join(d, f)
                if p not in seen:
                    out.append(p)
                    seen.add(p)
    return out


def download_whisper_model(name):
    """Download a whisper.cpp ggml model with resume + progress. Prefers
    ~/whisper-models/ if it exists, else script dir. Returns (path, error)."""
    # expected sizes (MB) so we can show % even when server sends no Content-Length
    EXPECTED = {
        "tiny": 75, "base": 141, "small": 466, "medium": 1500,
        "large-v3-turbo": 1600, "large-v3": 3100, "large-v2": 3100,
        "large-v1": 3100, "medium.en": 1500, "small.en": 466,
        "base.en": 141, "tiny.en": 75,
    }
    base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    fname = f"ggml-{name}.bin"
    models_dir = os.path.expanduser("~/whisper-models")
    if os.path.isdir(models_dir):
        out = os.path.join(models_dir, fname)
    else:
        out = os.path.join(SCRIPT_DIR, fname)
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        return out, None

    part = out + ".part"
    expected_mb = EXPECTED.get(name, 1600)
    expected_bytes = expected_mb * 1024 * 1024

    # resume from existing .part
    resume_from = 0
    if os.path.exists(part):
        resume_from = os.path.getsize(part)

    try:
        url = f"{base}/{fname}"
        req = urllib.request.Request(url)
        if resume_from > 0:
            req.add_header("Range", f"bytes={resume_from}-")
            broadcast("info", {"msg": f"Resuming {name} from {resume_from//(1024*1024)} MB…"})

        with urllib.request.urlopen(req, timeout=900) as r:
            # detect range vs full response
            is_partial = r.status == 206
            if is_partial and resume_from == 0:
                # server returned 206 even without Range header → unexpected
                is_partial = False
            mode = "ab" if (is_partial and resume_from > 0) else "wb"
            with open(part, mode) as f:
                t0 = time.time()
                written = resume_from
                last_bcast = 0
                last_bcast2 = 0
                chunk_size = 512 * 1024  # 512 KB
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    pct = min(100, written * 100 // expected_bytes) if expected_bytes else 0
                    mb = written // (1024 * 1024)
                    msg = f"⬇ {name}: {pct}% · {mb} MB / ~{expected_mb} MB"
                    # status bar (throttled ~2s)
                    if now - last_bcast > 1.5:
                        elapsed = now - t0
                        speed = (written - resume_from) / elapsed / (1024 * 1024) if elapsed > 0 else 0
                        eta = (expected_bytes - written) / ((written - resume_from) / elapsed) \
                              if (written - resume_from) > 0 and elapsed > 0 else 0
                        suffix = f" · {speed:.1f} MB/s" if speed > 0.1 else ""
                        suffix += f" · ETA {eta:.0f}s" if eta > 1 else ""
                        broadcast("info", {"msg": msg + suffix})
                        last_bcast = now
                    # modal progress (throttled ~250ms)
                    if now - last_bcast2 > 0.25:
                        broadcast("dl_progress", {
                            "name": name, "pct": pct,
                            "mb": written // (1024 * 1024),
                            "total_mb": expected_mb,
                        })
                        last_bcast2 = now
        # sanity check — file should be at least 80% of expected
        final_size = os.path.getsize(part)
        if final_size < expected_bytes * 0.8:
            return None, (f"file too small: {final_size//(1024*1024)} MB "
                          f"(expected ~{expected_mb} MB). Check connection.")
        os.rename(part, out)
        return out, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Model availability + system resources
# ---------------------------------------------------------------------------
def get_system_resources():
    """Return (free_ram_mb, total_ram_mb, free_swap_mb) or (None, None, None)
    if unavailable. Used to warn before downloading models that won't fit."""
    try:
        import resource  # POSIX-only
        # ru_maxrss is bytes on Linux, but on macOS it's also bytes.
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = ru.ru_maxrss / (1024 * 1024)
    except Exception:
        rss_mb = None
    free_mb = total_mb = None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()
        total_mb = int(out) // (1024 * 1024) if out else None
    except Exception:
        pass
    # free RAM via vm_stat
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
        page_size = 16384  # Apple Silicon default
        free_pages = 0
        for line in out.splitlines():
            if "Pages free" in line:
                free_pages = int(line.split()[-1].rstrip("."))
        if free_pages:
            free_mb = free_pages * page_size // (1024 * 1024)
    except Exception:
        pass
    return free_mb, total_mb


# MLX model catalog — backend for translate(). name → (size_mb, hf_repo, ram_mb).
# ram_mb is approximate memory required to *load* the model in addition to OS overhead.
MLX_MODEL_CATALOG = {
    "qwen2.5:0.5b":   {"hf_repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "size_mb": 400,  "ram_mb": 800},
    "gemma3:1b":      {"hf_repo": "mlx-community/gemma-3-1b-it-4bit",          "size_mb": 900,  "ram_mb": 1400},
    "qwen2.5:1.5b":   {"hf_repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "size_mb": 1100, "ram_mb": 1800},
    "gemma2:2b":      {"hf_repo": "mlx-community/gemma-2-2b-it-4bit",          "size_mb": 1700, "ram_mb": 2400},
    "gemma3:4b":      {"hf_repo": "mlx-community/gemma-3-4b-it-4bit",          "size_mb": 2500, "ram_mb": 3500},
    "gemma2:9b":      {"hf_repo": "mlx-community/gemma-2-9b-it-4bit",          "size_mb": 6500, "ram_mb": 7800},
    "llama3.1:8b":    {"hf_repo": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "size_mb": 4500, "ram_mb": 5800},
}


def _hf_cache_dir():
    """Return the HuggingFace cache root (where downloads land)."""
    return os.path.expanduser("~/.cache/huggingface/hub")


def list_incomplete_downloads():
    """Find any *.incomplete files in HF cache that look like aborted downloads.
    Returns list of {path, size_mb, age_sec}."""
    out = []
    root = _hf_cache_dir()
    if not os.path.isdir(root):
        return out
    now = time.time()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".incomplete"):
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p)
                    out.append({
                        "path": p,
                        "size_mb": st.st_size // (1024 * 1024),
                        "age_sec": int(now - st.st_mtime),
                    })
                except OSError:
                    pass
    return out


def cleanup_incomplete_downloads():
    """Delete *.incomplete files in HF cache. Returns (removed_count, freed_mb)."""
    items = list_incomplete_downloads()
    freed = 0
    for it in items:
        try:
            freed += it["size_mb"]
            os.remove(it["path"])
        except OSError:
            pass
    return len(items), freed


def list_mlx_models_installed():
    """List MLX model repos present in HF cache. Returns list of
    {hf_repo, size_mb}."""
    root = _hf_cache_dir()
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        # Only list MLX repos
        if not repo.startswith("mlx-community/"):
            continue
        full = os.path.join(root, d)
        size = 0
        for dp, _dn, files in os.walk(full):
            for f in files:
                if f.endswith(".incomplete"):
                    continue
                try:
                    size += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        if size > 0:
            out.append({"hf_repo": repo, "size_mb": size // (1024 * 1024)})
    return out


def check_model_availability(model_name):
    """For an Ollama-style model name (e.g. 'gemma2:9b'), check:
    - is it installed in Ollama?
    - if MLX fallback exists, is it downloaded?
    - can it fit in available RAM?
    Returns dict with status fields."""
    result = {
        "name": model_name,
        "ollama_installed": False,
        "ollama_size_mb": 0,
        "mlx_repo": None,
        "mlx_installed": False,
        "mlx_size_mb": 0,
        "ram_required_mb": 0,
        "ram_available_mb": None,
        "ram_ok": True,
        "warning": None,
    }
    # Ollama
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/show", timeout=3) as r:
            r.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            pass
    # Better: use /api/tags and /api/ps
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
            for m in data.get("models", []):
                if m.get("name") == model_name:
                    result["ollama_installed"] = True
                    result["ollama_size_mb"] = m.get("size", 0) // (1024 * 1024)
                    break
    except Exception:
        pass
    # MLX catalog
    info = MLX_MODEL_CATALOG.get(model_name)
    if info:
        result["mlx_repo"] = info["hf_repo"]
        result["ram_required_mb"] = info["ram_mb"]
        # check HF cache
        repo_dir_name = "models--" + info["hf_repo"].replace("/", "--")
        full = os.path.join(_hf_cache_dir(), repo_dir_name)
        if os.path.isdir(full):
            for dp, _dn, files in os.walk(full):
                for f in files:
                    if f.endswith(".incomplete"):
                        continue
                    try:
                        result["mlx_size_mb"] += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            result["mlx_size_mb"] //= (1024 * 1024)
            if result["mlx_size_mb"] > info["size_mb"] * 0.5:
                result["mlx_installed"] = True
    # RAM
    free_mb, _ = get_system_resources()
    result["ram_available_mb"] = free_mb
    if free_mb and info and free_mb < info["ram_mb"]:
        result["ram_ok"] = False
        result["warning"] = (f"⚠ Only {free_mb} MB free RAM — {model_name} needs "
                             f"~{info['ram_mb']} MB. Free memory or close other apps.")
    elif not info and not result["ollama_installed"]:
        result["warning"] = (f"'{model_name}' is not installed (Ollama) and has no "
                             f"MLX fallback. Try `ollama pull {model_name}` first.")
    return result


def models_status():
    """Aggregate status for the model manager modal.
    Returns dict with: ollama_installed, whisper_installed, mlx_installed,
    ram_free_mb, ram_total_mb, incomplete_downloads."""
    free, total = get_system_resources()
    return {
        "ollama": {
            "installed": fetch_models(),  # already a list of names
            "url": OLLAMA_URL,
        },
        "whisper": {
            "installed": list_whisper_models(),
            "catalog_size": {
                "tiny": 75, "base": 141, "small": 466, "medium": 1500,
                "large-v3-turbo": 1600, "large-v3": 3100,
            },
        },
        "mlx": {
            "installed": list_mlx_models_installed(),
            "catalog": [{"name": k, **v} for k, v in MLX_MODEL_CATALOG.items()],
        },
        "system": {
            "ram_free_mb": free,
            "ram_total_mb": total,
        },
        "incomplete_downloads": list_incomplete_downloads(),
    }


def _ollama_error_message(e):
    """Extract Ollama's actual error message from an HTTPError body.
    Returns a (short, hint) tuple — short is for the UI, hint is the actionable next step."""
    raw = ""
    try:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
    except Exception:
        raw = str(e)
    try:
        body = json.loads(raw)
        msg = body.get("error") or raw or str(e)
    except Exception:
        msg = raw or str(e)
    msg = msg.strip().strip('"')
    # Provide actionable hints for common cases
    low = msg.lower()
    if "does not support chat" in low or "does not support generate" in low:
        # Extract model name (everything before "does not support") — done outside f-string
        # to avoid backslash issues in Python < 3.12 f-string expressions.
        model_name = msg.split("does not support")[0].strip().strip('"')
        hint = (
            "'" + model_name + "' is missing the 'completion' capability. "
            "Pick a chat-capable model like gemma2:9b or qwen2.5:1.5b, "
            "or recreate with `ollama create` using a proper chat template."
        )
        return ("Model doesn't support text generation", hint)
    if "model not found" in low or "not found" in low:
        return ("Model not found: " + msg, "Run `ollama pull <model>` or pick another model.")
    if "connection refused" in low or "unreachable" in low:
        return ("Ollama is not running", "Start it with: `ollama serve` (or open the Ollama app).")
    return (msg, None)


def test_ollama():
    """Send a quick test prompt to verify Ollama is responding."""
    with _state_lock:
        model = state["model"]
    if not model:
        return False, 0.0, "No model selected"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: READY"}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    start = time.time()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
            elapsed = time.time() - start
            content = (data.get("message", {}).get("content") or "").strip()
            return True, elapsed, f"✓ {model} responded in {elapsed:.1f}s · '{content[:30]}'"
    except urllib.error.HTTPError as e:
        short, hint = _ollama_error_message(e)
        suffix = f" · {hint}" if hint else ""
        return False, time.time() - start, f"✗ Ollama HTTP {e.code}: {short}{suffix}"
    except urllib.error.URLError as e:
        return False, time.time() - start, f"✗ Ollama unreachable: {e.reason}. Is `ollama serve` running?"
    except Exception as e:
        return False, time.time() - start, f"✗ {type(e).__name__}: {e}"


def test_capture():
    """Record 5s from the current input device to test_capture.wav.
    Returns (path, level, error)."""
    with _state_lock:
        device_name = state["device"]
    dev_index = find_device(device_name)
    if dev_index is None:
        return None, 0.0, f"Device '{device_name}' not found"
    try:
        # record 5s
        rec = sd.rec(int(SAMPLERATE * 5), samplerate=SAMPLERATE,
                     channels=1, dtype="float32", device=dev_index)
        sd.wait()
        vol = float(np.linalg.norm(rec) / np.sqrt(len(rec)))
        out_path = os.path.join(SCRIPT_DIR, "test_capture.wav")
        save_wav(out_path, rec)
        return out_path, vol, None
    except Exception as e:
        return None, 0.0, str(e)


# ============================================================
# Audio preprocessing — P2: denoise + VAD
# ============================================================

_vad_instance = None

def _get_vad():
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = webrtcvad.Vad(VAD_AGGR)
    return _vad_instance


def is_frame_speech(vad, frame_f32, sr=SAMPLERATE):
    """Run webrtcvad on sub-frames of frame_f32.
    Returns True if at least 30% of the sub-frames contain speech."""
    try:
        pcm16 = (np.clip(frame_f32, -1, 1) * 32767).astype(np.int16).tobytes()
        subframe_samples = int(sr * 0.030)  # 30 ms = 480 samples
        subframe_bytes = subframe_samples * 2  # 16-bit PCM has 2 bytes per sample = 960 bytes
        n_subframes = len(pcm16) // subframe_bytes
        if n_subframes == 0:
            return False
        speech_count = 0
        for i in range(n_subframes):
            subframe = pcm16[i * subframe_bytes:(i + 1) * subframe_bytes]
            if vad.is_speech(subframe, sr):
                speech_count += 1
        return (speech_count / n_subframes) >= 0.30
    except Exception:
        return True  # fail open


def denoise_chunk(audio_f32, sr=SAMPLERATE):
    """Apply noisereduce to a float32 audio chunk. Always-on (no toggle).
    Skips silently short or near-silent audio to avoid scipy UserWarning."""
    if audio_f32 is None or len(audio_f32) < sr // 2:
        return audio_f32
    if float(np.abs(audio_f32).max()) < 1e-4:
        return audio_f32
    try:
        # Suppress the harmless scipy spectral warning for short/silent inputs.
        # We already gated those above; this is belt-and-suspenders.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Pass float32 numpy array directly without casting to float64 and back!
            return nr.reduce_noise(y=audio_f32, sr=sr, **DENOISE_PROPS)
    except Exception:
        return audio_f32


def has_speech(audio_f32, sr=SAMPLERATE):
    """Use webrtcvad to detect if chunk contains enough speech.
    Returns (has_speech: bool, speech_ratio: float). Always-on (no toggle)."""
    if audio_f32 is None or len(audio_f32) < sr // 4:
        return False, 0.0
    try:
        vad = _get_vad()
        # webrtcvad needs 16-bit PCM at 8k/16k/32k/48k, frame sizes 10/20/30 ms
        pcm16 = (np.clip(audio_f32, -1, 1) * 32767).astype(np.int16).tobytes()
        frame_bytes = int(sr * VAD_FRAME_MS / 1000) * 2  # 2 bytes per sample
        if len(pcm16) < frame_bytes * 3:
            return False, 0.0
        n_frames = len(pcm16) // frame_bytes
        speech_frames = 0
        for i in range(n_frames):
            frame = pcm16[i * frame_bytes:(i + 1) * frame_bytes]
            if vad.is_speech(frame, sr):
                speech_frames += 1
        ratio = speech_frames / max(1, n_frames)
        return ratio >= 0.10, ratio  # at least 10% speech frames
    except Exception:
        return True, 1.0  # fail open — don't drop audio on VAD errors


# ============================================================
# Prompts — P1: Thai-tuned, language-pair aware, glossary-aware
# ============================================================

# Hotwords appended to the Whisper --prompt (helps with proper nouns / tech terms)
LANG_HOTWORDS = {
    "German":   "API, Deployment, Sprint, Backlog, Stakeholder, Roadmap, Feature, Release, Review, OKR, KPI.",
    "English":  "API, deployment, sprint, backlog, stakeholder, roadmap, feature, release, review, OKR, KPI.",
    "Japanese": "API, デプロイ, スプリント, バックログ, ステークホルダー, ロードマップ.",
    "French":   "API, déploiement, sprint, backlog, partie prenante, feuille de route.",
    "Spanish":  "API, despliegue, sprint, backlog, parte interesada, hoja de ruta.",
    "Chinese (Simplified)": "API, 部署, 冲刺, 待办, 利益相关者, 路线图.",
}


def build_whisper_prompt(src_lang):
    """Return whisper.cpp --prompt for the current source language."""
    with _state_lock:
        custom = state.get("custom_prompt", "").strip()
        glossary = state.get("glossary", "").strip()
    base = DEFAULT_WHISPER_PROMPT
    hot = LANG_HOTWORDS.get(src_lang, "")
    pieces = [base]
    if hot:
        pieces.append("Schlüsselbegriffe: " + hot if src_lang == "German" else "Key terms: " + hot)
    if glossary:
        pieces.append("Wichtige Namen/Begriffe: " + glossary)
    if custom:
        pieces.append(custom)
    return " ".join(pieces)


def get_whisper_prompt(src_lang):
    if _whisper_prompt_cache["src"] == src_lang:
        return _whisper_prompt_cache["prompt"]
    p = build_whisper_prompt(src_lang)
    _whisper_prompt_cache["src"] = src_lang
    _whisper_prompt_cache["prompt"] = p
    return p


# Language-pair translation rules
LANG_PAIR_PROMPTS = {
    ("German", "Thai"): (
        "You translate spoken German to natural Thai in a business/tech meeting context.\n"
        "Rules:\n"
        "- Output ONLY the Thai translation. No quotes, no 'Translation:', no language tags.\n"
        "- Translate the meaning of German words to natural Thai. Do NOT transliterate German words phonetically (except for proper names), and do NOT put the original German words in parentheses.\n"
        "- Transliterate proper nouns phonetically (e.g. 'Herr Müller' → 'คุณมึลเลอร์'). "
        "  Never translate names.\n"
        "- Keep technical terms in English if commonly used in Thai (API, deployment, sprint, "
        "  backlog, server, dashboard, release).\n"
        "- Numbers: use Arabic digits (1,234.56). Write Thai digits only for years if requested.\n"
        "- Preserve sentence breaks. Never merge or split sentences.\n"
        "- Do NOT add greetings, sign-offs, or commentary."
    ),
    ("English", "Thai"): (
        "You translate spoken English to natural Thai in a business/tech meeting context.\n"
        "Rules:\n"
        "- Output ONLY the Thai translation. No quotes, no preamble.\n"
        "- Transliterate proper nouns phonetically. Never translate names.\n"
        "- Keep technical terms in English if commonly used in Thai.\n"
        "- Use Arabic digits. Preserve sentence breaks. No commentary."
    ),
    ("Japanese", "Thai"): (
        "You translate spoken Japanese to natural Thai.\n"
        "Rules:\n"
        "- Output ONLY the Thai translation. No quotes, no preamble.\n"
        "- Keep names and technical terms in original form."
    ),
}

GENERIC_TRANSLATE_PROMPT = (
    "You are a professional translator. Translate the user's text from {src} to {tgt}. "
    "Output ONLY the direct translation — no quotes, no explanations, no preamble, "
    "no language tags. Preserve tone and line breaks."
)


def build_translation_prompt(src, tgt, glossary=""):
    """Return translation system prompt for a language pair."""
    with _state_lock:
        custom = state.get("custom_prompt", "").strip()
        concise = state.get("concise_translation", False)
    pair = LANG_PAIR_PROMPTS.get((src, tgt))
    base = pair or GENERIC_TRANSLATE_PROMPT.format(src=src, tgt=tgt)
    pieces = [base]
    if concise:
        pieces.append(
            "CRITICAL: Keep the translation extremely concise, brief, and direct. "
            "Remove unnecessary filler words, redundant phrases, and extra details. "
            "Translate only the core meaning in as few words as possible to minimize length and latency."
        )
    if glossary:
        pieces.append("Glossary (use these exact terms):\n" + glossary)
    if custom:
        pieces.append("Additional instructions:\n" + custom)
    return "\n\n".join(pieces)


def get_translation_prompt(src, tgt):
    with _state_lock:
        glossary = state.get("glossary", "").strip()
        concise = state.get("concise_translation", False)
    key = (src, tgt, glossary, concise)
    if _translation_prompt_cache["key"] == key:
        return _translation_prompt_cache["prompt"]
    p = build_translation_prompt(src, tgt, glossary)
    _translation_prompt_cache["key"] = key
    _translation_prompt_cache["prompt"] = p
    return p


# ============================================================
# Validation — P2: detect bad translations, trigger retry
# ============================================================

# Characters that "shouldn't" appear in a clean translation (heuristic)
SRC_CHAR_HINTS = {
    "German":  "äöüÄÖÜß",
    "Japanese": "ぁ-ゖァ-ヺー一-鿿",
    "Chinese (Simplified)": "一-鿿",
    "Chinese (Traditional)": "一-鿿",
    "Korean": "가-힯",
    "Russian": "а-яА-ЯёЁ",
    "Arabic": "؀-ۿ",
    "Hindi": "ऀ-ॿ",
    "Thai": "ก-๛",
}


def validate_translation(src_text, tgt_text, src_lang, tgt_lang):
    """Return (ok: bool, reason: str). Catches obvious translation failures."""
    if not tgt_text or not tgt_text.strip():
        return False, "empty"
    if tgt_text.startswith("[Translate error") or tgt_text.startswith("[Error"):
        return False, "api_error"
    # same length as source → likely echoed instead of translated
    if src_text and len(tgt_text) > len(src_text) * 4:
        return False, "too_long"
    if len(tgt_text) < max(2, len(src_text) // 6):
        return False, "too_short"
    # source characters bleeding into target (rough heuristic)
    hints = SRC_CHAR_HINTS.get(src_lang, "")
    if hints and src_lang != tgt_lang:
        # count characters from src script in tgt
        try:
            bleed = sum(1 for c in tgt_text if c in hints)
            if bleed > max(3, len(tgt_text) // 4):
                return False, "src_chars_in_tgt"
        except Exception:
            pass
    # common LLM preamble patterns
    lower = tgt_text.lower().lstrip()
    bad_prefixes = ("translation:", "translated text:", "here is", "sure,",
                    "certainly,", "okay,", "in thai:", "in german:", "in japanese:")
    if any(lower.startswith(p) for p in bad_prefixes):
        return False, "preamble"
    return True, "ok"


# ============================================================
# STT backends — P1 + P3: whisper.cpp (CLI) + mlx-whisper (in-process)
# ============================================================

def _strip_hallucinations(text):
    """Remove common hallucination tags / suppress-regex matches."""
    if not text:
        return text
    for tag in HALLUC_TAGS:
        text = text.replace(tag, "")
    text = re.sub(SUPPRESS_REGEX, "", text)
    text = "\n".join(
        ln for ln in text.splitlines()
        if "Detected language" not in ln and "Whisper" not in ln
    )
    return text.strip()


def transcribe_cli(wav_path, lang, prompt):
    """whisper.cpp CLI backend (always available)."""
    with _state_lock:
        wmodel = state.get("wmodel") or WHISPER_MODEL
    if not os.path.exists(wmodel):
        broadcast("error", {"msg": f"Whisper model not found: {wmodel}"})
        return None
    cmd = [
        WHISPER_BIN,
        "-m", wmodel,
        "-l", lang,
        "-f", wav_path,
        "-nt",               # no timestamps
        "-t", str(WHISPER_THREADS),
        "--prompt", prompt,
        "--suppress-regex", SUPPRESS_REGEX,
        "--no-speech-thold", "0.7",
        "--entropy-thold", "2.0",
        "--max-len", "120",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", timeout=120)
        text = _strip_hallucinations((res.stdout or "").strip())
        if not text or "blank" in text.lower() or len(text) < 2:
            return None
        return text
    except subprocess.TimeoutExpired:
        broadcast("error", {"msg": "Whisper timed out"})
        return None
    except Exception as e:
        broadcast("error", {"msg": f"STT error: {e}"})
        return None


def _hf_repo_to_local_path(hf_repo):
    """Resolve HF repo id → local cache path. Skip network verification if fully cached.

    Why this exists: huggingface_hub.snapshot_download() (called by mlx_lm.load and
    mlx_whisper.transcribe) prints noisy "Fetching N files..." + "Download complete: 0.00B"
    every time, even when files are already in the cache. By resolving to a local path
    via `local_files_only=True` first, we hand a local directory to the loaders, which
    they accept and skip the network check entirely.

    Returns:
        str: Local cache path on success (no network check needed).
        None: If not cached or hf_repo is empty. Caller will use HF repo name as fallback.
    """
    if not hf_repo:
        return None
    # Already a local path on disk? Just use it.
    if os.path.isdir(hf_repo):
        return hf_repo
    try:
        from huggingface_hub import snapshot_download
        # local_files_only=True → pure cache lookup, raises if any file missing
        return snapshot_download(
            repo_id=hf_repo,
            local_files_only=True,
        )
    except Exception:
        # Not in cache or incomplete — caller will fall back to HF repo name (downloads)
        return None


def transcribe_mlx(audio_input, lang, prompt):
    """MLX-Whisper backend (in-process, uses ANE — requires mlx_whisper installed).
    Falls back to None if not available; caller should fall back to CLI.
    `audio_input` can be a path to a wav file or a numpy float32 array.
    """
    with _state_lock:
        use_mlx = state.get("use_mlx_whisper", True)
    if not HAS_MLX_WHISPER or not use_mlx:
        return None
    with _state_lock:
        wmodel = state.get("wmodel") or WHISPER_MODEL
    basename = os.path.basename(wmodel).replace("ggml-", "").replace(".bin", "")

    # Skip MLX if this basename has already failed once (saves 1-2s per chunk)
    if basename in _mlx_whisper_model["failed"]:
        return None

    # Cache hit — same model as last time, no need to resolve again
    if (basename == _mlx_whisper_model["basename"]
            and _mlx_whisper_model["hf_repo"]
            and _mlx_whisper_model["loaded"]):
        mlx_repo = _mlx_whisper_model["hf_repo"]
    else:
        mlx_repo = _whisper_basename_to_mlx_repo(basename)
        # Try local cache first → skip "Fetching N files" network verification
        local_path = _hf_repo_to_local_path(mlx_repo)
        if local_path:
            mlx_repo = local_path
        _mlx_whisper_model["basename"] = basename
        _mlx_whisper_model["hf_repo"] = mlx_repo
        _mlx_whisper_model["loaded"] = False  # force reload on next transcribe

    try:
        result = mlx_whisper.transcribe(
            audio_input,
            path_or_hf_repo=mlx_repo,
            language=lang if lang != "auto" else None,
            initial_prompt=prompt,
            fp16=True,
        )
        _mlx_whisper_model["loaded"] = True
        text = _strip_hallucinations((result.get("text") or "").strip())
        return text or None
    except Exception as e:
        # Remember the failure — don't retry MLX for this basename
        _mlx_whisper_model["failed"].add(basename)
        _mlx_whisper_model["loaded"] = False
        # Be specific about the error so user knows whether it's worth retrying
        err = type(e).__name__
        if "RepositoryNotFound" in err or "RevisionNotFound" in err:
            broadcast("info", {
                "msg": f"MLX-Whisper repo not found ({mlx_repo}) — falling back to CLI. "
                       f"Add it to _MLX_WHISPER_REPOS if needed."
            })
        elif "ConnectionError" in type(e).__module__:
            broadcast("info", {"msg": f"MLX-Whisper offline — falling back to CLI"})
        else:
            broadcast("info", {"msg": f"MLX-Whisper failed ({err}: {e}) — falling back to CLI"})
        return None


def transcribe(audio_f32, wav_path, lang):
    """Top-level STT — tries MLX-Whisper first (faster) with NumPy array directly,
    falls back to whisper.cpp CLI (writing WAV to disk only on fallback)."""
    prompt = get_whisper_prompt(_current_src_lang())
    # Try MLX first if installed (with in-memory numpy array, no disk write)
    text = transcribe_mlx(audio_f32, lang, prompt)
    if text is None:
        # Fallback to CLI: must write to disk
        save_wav(wav_path, audio_f32)
        text = transcribe_cli(wav_path, lang, prompt)
        try:
            os.remove(wav_path)
        except OSError:
            pass
    return text


def _current_src_lang():
    with _state_lock:
        return state.get("src_lang", "German")


# ============================================================
# Translation backends — P1 + P3: Ollama (HTTP) + MLX-LM (in-process)
# ============================================================

def _memory_block(history):
    """Format the last N translations as context for the LLM."""
    if not history:
        return ""
    lines = []
    for e in history[-TRANSLATION_MEMORY_SIZE:]:
        lines.append(f"[{e.get('src_lang', '?')}→{e.get('tgt_lang', '?')}] {e.get('src', '')}\n→ {e.get('tgt', '')}")
    return "\n\n".join(lines)


def _ollama_translate(text, model, system, use_memory, history):
    """Ollama /api/chat backend with optional streaming (P3) and memory."""
    messages = [{"role": "system", "content": system}]
    if use_memory:
        mem = _memory_block(history)
        if mem:
            messages.append({"role": "system", "content":
                "Previous translations in this conversation (preserve terminology and pronouns):\n" + mem})
    messages.append({"role": "user", "content": text})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 256,
            "repeat_penalty": 1.2,
            "top_p": 0.9,
            "stop": ["<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"],
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
            return (data.get("message", {}).get("content") or "").strip()
    except urllib.error.HTTPError as e:
        short, hint = _ollama_error_message(e)
        # Surface as broadcast so the user sees it in the UI status pill / toast
        msg = f"✗ Ollama {model}: {short}"
        if hint:
            msg += f" — {hint}"
        broadcast("error", {"msg": msg})
        raise RuntimeError(short) from e
    except urllib.error.URLError as e:
        broadcast("error", {"msg": f"✗ Ollama unreachable: {e.reason}. Is `ollama serve` running?"})
        raise RuntimeError(f"Ollama unreachable: {e.reason}") from e


def _mlx_lm_translate(text, model, system, use_memory, history):
    """In-process MLX-LM backend. Returns None if mlx_lm not available.
    Uses a separate cache from _mlx_whisper_model to keep concerns clean."""
    with _state_lock:
        use_mlx = state.get("use_mlx_lm", True)
    if not HAS_MLX_LM or not use_mlx:
        return None
    try:
        # Map ollama-style model name → HF repo if needed
        hf_model = _ollama_to_hf_repo(model)

        # Lazy-load & cache model + tokenizer
        cache = _mlx_lm_cache
        if cache["name"] != hf_model or not cache["loaded"]:
            cache["name"] = hf_model
            # Try local cache first → skip "Fetching N files" network verification
            load_target = _hf_repo_to_local_path(hf_model) or hf_model
            cache["model_obj"], cache["tokenizer"] = mlx_lm_load(load_target)
            cache["loaded"] = True

        # Build chat-templated prompt (works for Qwen, Llama, Mistral, Gemma, etc.)
        user_msg = text
        if use_memory:
            mem = _memory_block(history)
            if mem:
                user_msg = f"Previous translations (preserve terminology and tone):\n{mem}\n\nNow translate:\n{text}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = cache["tokenizer"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # fall back to plain prompt if chat template unsupported
            prompt = f"{system}\n\n{user_msg}\n\nTranslation:"

        out = mlx_lm_generate(
            cache["model_obj"],
            cache["tokenizer"],
            prompt=prompt,
            max_tokens=256,
            # Note: mlx_lm 0.31+ removed `temp=` kwarg. Greedy (no sampler)
            # is deterministic and matches Ollama's temperature=0.0 behavior.
        )
        return (out or "").strip()
    except Exception as e:
        broadcast("info", {"msg": f"MLX-LM unavailable ({type(e).__name__}: {str(e)[:80]}), falling back to Ollama"})
        _mlx_lm_cache["loaded"] = False
        return None


# Map Ollama model names → MLX-compatible HF repos for in-process translation
_OLLAMA_TO_HF = {
    "gemma4:latest":     "mlx-community/gemma-3-4b-it-4bit",
    "gemma3:4b":         "mlx-community/gemma-3-4b-it-4bit",
    "gemma3:1b":         "mlx-community/gemma-3-1b-it-4bit",
    "gemma2:9b":         "mlx-community/gemma-2-9b-it-4bit",
    "gemma2:2b":         "mlx-community/gemma-2-2b-it-4bit",
    "llama3.1:8b":       "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "llama3.2:3b":       "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama3.2:1b":       "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "qwen2.5:7b":        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5:3b":        "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "qwen2.5:1.5b":      "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen2.5:0.5b":      "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mistral:7b":        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "phi3:mini":         "mlx-community/Phi-3-mini-4k-instruct-4bit",
}


def _ollama_to_hf_repo(ollama_name):
    """Resolve Ollama model name to HF repo. Falls back to as-is."""
    if not ollama_name:
        return None
    if ollama_name in _OLLAMA_TO_HF:
        return _OLLAMA_TO_HF[ollama_name]
    # guess from prefix
    base = ollama_name.split(":")[0]
    for k, v in _OLLAMA_TO_HF.items():
        if k.startswith(base):
            return v
    return ollama_name  # assume it's already an HF repo id


def translate_text(text, model, src, tgt, history=None):
    """Top-level translate. Tries MLX-LM first if installed, else Ollama.
    Retries once on validation failure with a stricter prompt."""
    if not text.strip():
        return ""
    history = history if history is not None else _public_state().get("history", [])
    system = get_translation_prompt(src, tgt)
    with _state_lock:
        use_memory = True  # always-on (no UI toggle)

    # First attempt
    try:
        result = _mlx_lm_translate(text, model, system, use_memory, history)
        if result is None:
            result = _ollama_translate(text, model, system, use_memory, history)
    except Exception as e:
        result = f"[Translate error: {e}]"

    # Clean up special tokens like <end_of_turn>
    if isinstance(result, str) and not result.startswith("[Translate error"):
        for token in ["<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"]:
            result = result.replace(token, "")
        result = result.strip()
        if tgt == "Thai":
            result = re.sub(r'([ะัาิีึืุู็่้๊๋์ๆ])\1+', r'\1', result)

    # P2 — validate, retry once with stricter prompt
    ok, reason = validate_translation(text, result, src, tgt)
    if not ok and not result.startswith("[Translate error"):
        strict_system = system + (
            "\n\nIMPORTANT: Your previous response was rejected. "
            "Output ONLY the translation with no preamble, no explanation, no quotes."
        )
        try:
            retry = _ollama_translate(text, model, strict_system, use_memory=False, history=[])
            if retry:
                # Clean up retry text as well
                for token in ["<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"]:
                    retry = retry.replace(token, "")
                retry = retry.strip()
                if tgt == "Thai":
                    retry = re.sub(r'([ะัาิีึืุู็่้๊๋์ๆ])\1+', r'\1', retry)
                
                ok2, _ = validate_translation(text, retry, src, tgt)
                if ok2:
                    return retry
        except Exception:
            pass
        # keep original result even if validation failed
    return result


# ============================================================
# Threads
# ============================================================

def audio_thread():
    """Open BlackHole stream, track speech/silence dynamically, push cleaned chunks to audio_q.
    Updates the level meter in real-time (every 100ms) for smoother UI response."""
    vad = _get_vad()
    frame_duration = 0.1  # 100ms blocks
    frame_samples = int(SAMPLERATE * frame_duration)
    
    # Configuration parameters
    max_speech_duration = 5.0  # seconds (max chunk length)
    silence_threshold = 0.8  # seconds (silence after speech to trigger slice)
    
    pre_roll_buffer = deque(maxlen=5)  # store last 0.5s of silence
    accumulated_chunks = []
    
    is_speaking = False
    speech_time = 0.0
    silence_time = 0.0
    continuous_silent_samples = 0
    last_silent_warn_ts = 0.0
    
    while True:
        with _state_lock:
            listening = state["listening"]
            device_name = state["device"]
        if not listening:
            pre_roll_buffer.clear()
            accumulated_chunks.clear()
            is_speaking = False
            speech_time = 0.0
            silence_time = 0.0
            continuous_silent_samples = 0
            time.sleep(0.2)
            continue
            
        dev_index = find_device(device_name)
        if dev_index is None:
            broadcast("error", {"msg": f"Input device '{device_name}' not found"})
            with _state_lock:
                state["listening"] = False
            broadcast("state", _public_state())
            time.sleep(1)
            continue
            
        try:
            with sd.InputStream(device=dev_index, channels=CHANNELS,
                                samplerate=SAMPLERATE, dtype="float32") as stream:
                broadcast("info", {"msg": f"🎧 Capturing from {device_name} (dynamic VAD + denoise)"})
                # clear backlog
                for q in (audio_q, llm_q):
                    while not q.empty():
                        try: q.get_nowait()
                        except: break
                        
                while True:
                    with _state_lock:
                        if not state["listening"]:
                            break
                            
                    block, _ = stream.read(frame_samples)
                    is_speech = is_frame_speech(vad, block)
                    
                    # Level meter calculation (updated smoothly every 100ms)
                    vol = float(np.linalg.norm(block) / np.sqrt(len(block)))
                    vol = min(1.0, vol * 5)
                    with _state_lock:
                        state["level"] = vol
                    broadcast("level", {"v": vol})
                    
                    # Detect silent-input bug: system audio routing issue
                    raw_peak = float(np.abs(block).max())
                    if raw_peak < 0.001:
                        continuous_silent_samples += len(block)
                        if continuous_silent_samples >= SAMPLERATE * 15:  # 15s of silence
                            now = time.time()
                            if now - last_silent_warn_ts > 30:
                                broadcast("error", {
                                    "msg": f"🔇 No audio input on {device_name} — system audio "
                                           f"isn't routed here. Set '{device_name}' as the OUTPUT "
                                           f"in the source app (Teams/Zoom/browser), or create a "
                                           f"Multi-Output Device in Audio MIDI Setup that sends to "
                                           f"both your speakers and {device_name}."
                                })
                                last_silent_warn_ts = now
                            continuous_silent_samples = SAMPLERATE * 15  # cap
                    else:
                        continuous_silent_samples = 0
                        
                    # Dynamic VAD slicing state machine
                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            accumulated_chunks = list(pre_roll_buffer)
                            pre_roll_buffer.clear()
                            speech_time = len(accumulated_chunks) * frame_duration
                            silence_time = 0.0
                        
                        accumulated_chunks.append(block)
                        speech_time += frame_duration
                        silence_time = 0.0
                    else:
                        if is_speaking:
                            accumulated_chunks.append(block)
                            silence_time += frame_duration
                            speech_time += frame_duration
                            
                            # End of sentence (silence timeout) or maximum length reached
                            if silence_time >= silence_threshold or speech_time >= max_speech_duration:
                                chunk = np.concatenate(accumulated_chunks)
                                speech_ok, _ = has_speech(chunk)
                                if speech_ok:
                                    # apply denoise only before sending to STT queue
                                    chunk_clean = denoise_chunk(chunk)
                                    try:
                                        audio_q.put_nowait(chunk_clean)
                                    except queue.Full:
                                        pass
                                accumulated_chunks.clear()
                                is_speaking = False
                                speech_time = 0.0
                                silence_time = 0.0
                        else:
                            pre_roll_buffer.append(block)
                            
        except Exception as e:
            broadcast("error", {"msg": f"Audio error: {e}"})
            with _state_lock:
                state["listening"] = False
            broadcast("state", _public_state())


def stt_thread():
    """P2 — Stage 1: pull audio chunks → whisper STT → push transcribed text to llm_q.
    Runs in its own thread so the LLM stage can run in parallel with the next STT."""
    wav_path = os.path.join(SCRIPT_DIR, "temp_live_chunk.wav")
    while True:
        chunk = audio_q.get()
        if chunk is None:
            break
        with _state_lock:
            listening = state["listening"]
            src = state["src_lang"]
            state["whisper_busy"] = True
        if not listening:
            with _state_lock:
                state["whisper_busy"] = False
            continue
        try:
            whisper_lang = LANG_CODE.get(src, "auto")
            t0 = time.time()
            text = transcribe(chunk, wav_path, whisper_lang)
            dt = time.time() - t0
            if text:
                with _state_lock:
                    tgt = state["tgt_lang"]
                    model = state["model"]
                if not model:
                    broadcast("info", {"msg": "…no translation model selected"})
                    continue
                try:
                    llm_q.put_nowait({
                        "src": text,
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "model": model,
                        "stt_time": dt,
                        "ts": time.time(),
                    })
                except queue.Full:
                    broadcast("info", {"msg": "LLM queue full — dropping segment"})
            else:
                broadcast("info", {"msg": f"…no speech detected (stt {dt:.1f}s)"})
        except Exception as e:
            broadcast("error", {"msg": f"STT error: {e}"})
        finally:
            with _state_lock:
                state["whisper_busy"] = False
            if os.path.exists(wav_path):
                try: os.remove(wav_path)
                except OSError: pass


def llm_thread():
    """P2 — Stage 2: pull transcribed text → translate → broadcast.
    Runs concurrently with stt_thread so back-to-back chunks pipeline properly."""
    while True:
        item = llm_q.get()
        if item is None:
            break
        with _state_lock:
            state["llm_busy"] = True
        try:
            history_snapshot = list(_public_state().get("history", []))
            t0 = time.time()
            translation = translate_text(
                item["src"], item["model"], item["src_lang"], item["tgt_lang"],
                history=history_snapshot,
            )
            dt = time.time() - t0
            entry = {
                "ts": item["ts"],
                "src": item["src"],
                "tgt": translation,
                "src_lang": item["src_lang"],
                "tgt_lang": item["tgt_lang"],
                "latency": {"stt": round(item.get("stt_time", 0), 2),
                            "llm": round(dt, 2)},
            }
            with _state_lock:
                state["history"].append(entry)
                if len(state["history"]) > 200:
                    state["history"] = state["history"][-200:]
            broadcast("transcript", entry)
        except Exception as e:
            broadcast("error", {"msg": f"LLM error: {e}"})
        finally:
            with _state_lock:
                state["llm_busy"] = False


# ============================================================
# HTTP server
# ============================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0E0E13">
<title>Foundry — Realtime Voice Intelligence · The Factory Group</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23E11D48'/%3E%3Cpath d='M10 7v18M10 7h11M10 16h8' stroke='white' stroke-width='3' stroke-linecap='square' fill='none'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg-0:#08080B;--bg-1:#0E0E13;--bg-2:#15151C;--bg-3:#1C1C25;--bg-4:#25252F;
  --bd-1:#1F1F28;--bd-2:#2D2D38;--bd-3:#3A3A48;
  --tx:#F5F5F8;--tx-2:#B8BAC2;--tx-3:#75787F;--tx-4:#4A4D54;
  --red:#E11D48;--red-hot:#FB2D5C;--red-deep:#9F1239;
  --red-glow:rgba(225,29,72,.40);--red-soft:rgba(225,29,72,.10);
  --ok:#34D399;--warn:#F59E0B;--err:#EF4444;
  --f-display:'DM Serif Display',serif;
  --f-ui:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
  --f-mono:'JetBrains Mono',ui-monospace,monospace;
  --ease:cubic-bezier(.16,1,.3,1);
  --t-fast:.15s var(--ease);--t-med:.25s var(--ease);--t-slow:.4s var(--ease);
  --r-sm:6px;--r-md:10px;--r-lg:14px;--r-xl:20px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--f-ui);background:var(--bg-0);color:var(--tx);-webkit-font-smoothing:antialiased;font-size:14px;line-height:1.5;overflow:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(800px 600px at 50% -20%,rgba(225,29,72,.06),transparent 70%),radial-gradient(600px 400px at 100% 100%,rgba(225,29,72,.04),transparent 70%)}
.app{display:grid;grid-template-rows:64px 84px 1fr 56px;height:100vh;min-width:0;position:relative;z-index:1}
header{display:flex;align-items:center;padding:0 24px;background:rgba(8,8,11,.7);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--bd-1);position:relative;z-index:5}
header.recording::after{content:"";position:absolute;bottom:-1px;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--red),transparent);animation:recGlow 2s ease-in-out infinite}
@keyframes recGlow{0%,100%{opacity:.5}50%{opacity:1;box-shadow:0 0 24px var(--red-glow)}}
.brand{display:flex;align-items:center;gap:14px}
.brand-mark{width:40px;height:40px;background:linear-gradient(135deg,var(--red) 0%,var(--red-deep) 100%);border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;position:relative;box-shadow:0 6px 18px rgba(225,29,72,.3),inset 0 1px 0 rgba(255,255,255,.15)}
.brand-mark::before{content:"";position:absolute;inset:-3px;background:linear-gradient(135deg,var(--red-hot),transparent 60%);border-radius:var(--r-md);z-index:-1;opacity:.4;filter:blur(10px)}
.brand-mark svg{color:#fff;filter:drop-shadow(0 1px 1px rgba(0,0,0,.2))}
.brand-text{display:flex;flex-direction:column;line-height:1.05}
.brand-name{font-family:var(--f-display);font-size:24px;color:var(--tx);letter-spacing:-.01em;font-weight:400}
.brand-tag{font-size:9.5px;color:var(--tx-3);letter-spacing:1.6px;text-transform:uppercase;font-weight:500;margin-top:3px}
.brand-tag b{color:var(--red-hot);font-weight:600}
.header-spacer{flex:1}
.status-pill{display:flex;align-items:center;gap:10px;padding:7px 14px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:999px;font-family:var(--f-mono);font-size:11.5px;font-weight:500;letter-spacing:.5px;transition:all var(--t-med);margin-right:10px;max-width:560px;min-width:0}
.status-pill #status-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.status-pill.live{background:rgba(225,29,72,.08);border-color:rgba(225,29,72,.5);box-shadow:0 0 16px var(--red-glow)}
.status-pill .dot{width:8px;height:8px;border-radius:50%;background:var(--tx-4);position:relative;flex-shrink:0}
.status-pill.live .dot{background:var(--red);box-shadow:0 0 8px var(--red)}
.status-pill.live .dot::after{content:"";position:absolute;inset:-5px;border-radius:50%;border:2px solid var(--red);animation:ping 1.6s ease-out infinite}
@keyframes ping{0%{transform:scale(.5);opacity:1}100%{transform:scale(2.2);opacity:0}}
.status-pill .clock{color:var(--tx-3);font-weight:600;letter-spacing:1px}
.status-pill.live .clock{color:var(--tx-2)}
.icon-btn{width:38px;height:38px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-md);color:var(--tx-2);cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all var(--t-fast);margin-left:8px}
.icon-btn:hover{background:var(--bg-3);color:var(--tx);border-color:var(--bd-3)}
.icon-btn svg{width:18px;height:18px}
.icon-btn:hover svg{color:var(--red-hot)}

.toolbar{display:flex;align-items:flex-end;gap:14px;padding:16px 24px;background:var(--bg-1);border-bottom:1px solid var(--bd-1)}
.field{display:flex;flex-direction:column;gap:5px;min-width:0}
.field-label{font-size:9.5px;text-transform:uppercase;letter-spacing:1.4px;color:var(--tx-3);font-weight:700}
.field-control{display:flex;align-items:center;gap:8px;height:38px;padding:0 12px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-md);font-size:13px;font-weight:500;color:var(--tx);cursor:pointer;transition:all var(--t-fast);min-width:180px}
.field-control:hover{background:var(--bg-3);border-color:var(--bd-3)}
.field-control:focus-within{border-color:var(--red);box-shadow:0 0 0 3px var(--red-soft)}
.field-control select{appearance:none;-webkit-appearance:none;background:transparent;border:0;color:inherit;font:inherit;width:100%;cursor:pointer;outline:none}
.field-control select option{background:var(--bg-1);color:var(--tx);padding:6px}
.field-icon{width:15px;height:15px;color:var(--tx-3);flex-shrink:0}
.chevron{color:var(--tx-3);flex-shrink:0;pointer-events:none}
.swap-btn{width:38px;height:38px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-md);color:var(--tx-2);cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all var(--t-med);flex-shrink:0}
.swap-btn:hover{background:var(--bg-3);color:var(--red-hot);border-color:var(--red);transform:rotate(180deg)}
.swap-btn svg{width:16px;height:16px}
 .toolbar-spacer{flex:1}
.rec-btn{display:inline-flex;align-items:center;gap:10px;padding:0 22px;height:46px;background:linear-gradient(135deg,var(--red) 0%,var(--red-deep) 100%);border:0;border-radius:var(--r-md);color:#fff;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;cursor:pointer;box-shadow:0 6px 20px rgba(225,29,72,.35),inset 0 1px 0 rgba(255,255,255,.15);transition:all var(--t-med);position:relative;flex-shrink:0}
.rec-btn:hover{transform:translateY(-1px);box-shadow:0 8px 28px rgba(225,29,72,.45),inset 0 1px 0 rgba(255,255,255,.2)}
.rec-btn:active{transform:translateY(0)}
.rec-btn .d{width:10px;height:10px;background:#fff;border-radius:50%;box-shadow:0 0 8px rgba(255,255,255,.6);transition:all var(--t-fast)}
.rec-btn.live{background:linear-gradient(135deg,var(--red-deep) 0%,var(--red) 100%)}
.rec-btn.live::before{content:"";position:absolute;inset:-3px;border-radius:var(--r-md);background:linear-gradient(135deg,var(--red-hot),var(--red));z-index:-1;opacity:.6;filter:blur(14px);animation:recPulse 1.4s ease-in-out infinite}
@keyframes recPulse{0%,100%{opacity:.3}50%{opacity:.75}}
.rec-btn.live .d{animation:recDotBlink 1s ease-in-out infinite}
@keyframes recDotBlink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}
.rec-btn .stop-ic{display:none}
.rec-btn.live .d{display:none}
.rec-btn.live .stop-ic{display:inline-block;width:10px;height:10px;background:#fff;border-radius:1.5px}
main{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--bd-1);min-height:0;position:relative}
.pane{display:flex;flex-direction:column;background:var(--bg-1);min-height:0;position:relative}
.pane-head{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--bd-1);background:var(--bg-1);position:sticky;top:0;z-index:2}
.pane-title{display:flex;align-items:center;gap:10px}
.pane-label{font-size:9.5px;text-transform:uppercase;letter-spacing:1.6px;color:var(--tx-3);font-weight:700}
.pane-lang{font-family:var(--f-mono);font-size:10.5px;font-weight:600;padding:3px 8px;background:var(--bg-3);border-radius:4px;color:var(--tx-2);letter-spacing:.4px}
.pane.tgt .pane-lang{background:var(--red-soft);color:var(--red-hot)}
.pane-actions{display:flex;gap:2px}
.pane-btn{padding:6px 10px;background:transparent;border:1px solid transparent;border-radius:var(--r-sm);font-size:11.5px;font-weight:600;color:var(--tx-3);cursor:pointer;transition:all var(--t-fast)}
.pane-btn:hover{color:var(--tx);background:var(--bg-3)}
.pane-btn.on{color:var(--red-hot);background:var(--red-soft);border-color:rgba(225,29,72,.3)}
.feed{flex:1;overflow-y:auto;padding:18px 22px 24px;display:flex;flex-direction:column;gap:12px;scrollbar-color:var(--bd-3) transparent}
.feed::-webkit-scrollbar{width:8px}
.feed::-webkit-scrollbar-track{background:transparent}
.feed::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:4px}
.feed::-webkit-scrollbar-thumb:hover{background:var(--bd-3)}
.entry{position:relative;padding:14px 16px;background:var(--bg-2);border:1px solid var(--bd-1);border-left:3px solid var(--bd-3);border-radius:var(--r-md);animation:entryIn .35s var(--ease);transition:all var(--t-fast)}
.entry:hover{border-color:var(--bd-2);border-left-color:var(--red)}
.pane.tgt .entry{border-left-color:var(--red)}
@keyframes entryIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.entry-meta{display:flex;align-items:center;gap:8px;font-family:var(--f-mono);font-size:10px;color:var(--tx-4);letter-spacing:.5px;margin-bottom:6px;text-transform:uppercase}
.entry-meta .ts{color:var(--tx-3);font-weight:600}
.entry-meta .dot{width:3px;height:3px;border-radius:50%;background:var(--tx-4)}
.entry-meta .lat{color:var(--ok);font-weight:600}
.entry-text{font-size:15px;line-height:1.55;color:var(--tx);white-space:pre-wrap;word-wrap:break-word}
.pane.tgt .entry-text{font-size:16px;font-weight:500}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;color:var(--tx-3);text-align:center;padding:40px 20px}
.empty-icon{width:48px;height:48px;color:var(--tx-4);margin-bottom:14px;opacity:.5;stroke-width:1.2}
.empty-title{font-size:14px;color:var(--tx-2);font-weight:600;margin-bottom:4px;letter-spacing:.2px}
.empty-sub{font-size:12px;color:var(--tx-3);max-width:280px;line-height:1.55}
.empty-sub b{color:var(--red-hot);font-weight:600}

.stats-bar{display:flex;align-items:center;gap:14px;padding:0 24px;background:var(--bg-1);border-top:1px solid var(--bd-1);height:56px}
.waveform{display:flex;align-items:center;gap:3px;height:34px;flex:1;padding:0 8px;background:var(--bg-2);border:1px solid var(--bd-1);border-radius:var(--r-md);overflow:hidden;position:relative}
.waveform::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(225,29,72,.04),transparent);opacity:0;transition:opacity var(--t-med);pointer-events:none}
.waveform.active::before{opacity:1}
.wave-bar{width:3px;background:var(--bd-3);border-radius:2px;transition:height .08s linear,background var(--t-med);min-height:2px;flex-shrink:0}
.waveform.active .wave-bar{background:linear-gradient(180deg,var(--red-hot) 0%,var(--red) 100%);box-shadow:0 0 4px rgba(225,29,72,.3)}
.stat-pill{display:flex;align-items:center;gap:6px;padding:5px 10px;background:var(--bg-2);border:1px solid var(--bd-1);border-radius:var(--r-sm);font-family:var(--f-mono);font-size:10.5px;white-space:nowrap}
.stat-pill .l{font-size:9px;text-transform:uppercase;letter-spacing:1.2px;color:var(--tx-3);font-weight:700;font-family:var(--f-ui)}
.stat-pill .v{color:var(--tx-2);font-weight:600}
.stat-pill.ok .v{color:var(--ok)}
.stat-pill.rec .v{color:var(--red-hot)}
footer{display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:var(--bg-0);border-top:1px solid var(--bd-1);font-size:10.5px;color:var(--tx-3);height:100%}
.backend-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.backend-chip{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:4px;background:var(--bg-2);border:1px solid var(--bd-1);font-size:9.5px;font-weight:600;letter-spacing:.5px;color:var(--tx-3);font-family:var(--f-mono);text-transform:uppercase}
.backend-chip.on{color:var(--ok);border-color:rgba(52,211,153,.3);background:rgba(52,211,153,.05)}
.backend-chip.mlx.on{color:var(--red-hot);border-color:rgba(225,29,72,.35);background:rgba(225,29,72,.06)}
.backend-chip .d{width:5px;height:5px;border-radius:50%;background:currentColor}
.backend-chip.mlx.on .d{box-shadow:0 0 6px currentColor}
.footer-meta{display:flex;align-items:center;gap:14px;font-family:var(--f-mono);font-size:10.5px;color:var(--tx-3)}
.footer-meta b{color:var(--tx-2);font-weight:600}
.footer-meta .sep{color:var(--bd-3)}
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(8px);display:none;align-items:center;justify-content:center;z-index:50;animation:fadeIn var(--t-med)}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal-bg.open{display:flex}
.modal{background:var(--bg-1);border:1px solid var(--bd-2);border-radius:var(--r-lg);width:min(680px,92vw);max-height:84vh;display:flex;flex-direction:column;box-shadow:0 24px 70px rgba(0,0,0,.7),0 0 0 1px rgba(225,29,72,.05);animation:modalIn var(--t-slow)}
@keyframes modalIn{from{opacity:0;transform:scale(.96) translateY(10px)}to{opacity:1;transform:scale(1) translateY(0)}}
.modal-head{padding:18px 24px;border-bottom:1px solid var(--bd-1);display:flex;align-items:center;justify-content:space-between}
.modal-head h2{font-family:var(--f-display);font-size:22px;color:var(--tx);margin:0;font-weight:400}
.modal-head h2 small{display:block;font-family:var(--f-ui);font-size:10px;font-weight:500;color:var(--tx-3);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px}
.modal-close{width:32px;height:32px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-sm);color:var(--tx-2);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all var(--t-fast);line-height:1}
.modal-close:hover{background:var(--bg-3);color:var(--tx);border-color:var(--red)}
.modal-tabs{display:flex;padding:0 24px;border-bottom:1px solid var(--bd-1)}
.modal-tabs button{background:transparent;border:0;border-bottom:2px solid transparent;color:var(--tx-3);padding:14px 18px;font-size:12.5px;font-weight:600;cursor:pointer;transition:all var(--t-fast);text-transform:uppercase;letter-spacing:.8px}
.modal-tabs button:hover{color:var(--tx-2)}
.modal-tabs button.active{color:var(--red-hot);border-bottom-color:var(--red)}
.modal-body{padding:18px 24px;overflow-y:auto;flex:1}
.model-row{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;padding:14px 16px;background:var(--bg-2);border:1px solid var(--bd-1);border-radius:var(--r-md);margin-bottom:10px;transition:all var(--t-fast)}
.model-row:hover{border-color:var(--bd-2)}
.model-row.active{border-color:var(--red);background:rgba(225,29,72,.04);box-shadow:0 0 0 1px var(--red-soft)}
.model-info{min-width:0}
.model-name{display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:600;color:var(--tx)}
.model-name .star{color:var(--red-hot)}
.model-name .badge{font-size:9px;padding:2px 6px;background:rgba(52,211,153,.12);color:var(--ok);border-radius:3px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.model-name .badge.miss{background:var(--bg-3);color:var(--tx-3)}
.model-name .badge.warn{background:rgba(241,193,74,.15);color:#f1c14a;border:1px solid rgba(241,193,74,.3);font-size:9.5px;animation:warnPulse 1.6s ease-in-out infinite}
.model-name .badge.ok{background:rgba(52,211,153,.18);color:var(--ok);font-size:9.5px}
.meta-muted{font-size:11px;color:var(--tx-4);padding:8px 14px;font-style:italic}
@keyframes warnPulse{0%,100%{opacity:1}50%{opacity:.65}}
.status-strip{display:flex;gap:10px;align-items:center;padding:10px 14px;background:var(--bg-2);border:1px solid var(--bd-1);border-radius:var(--r-md);margin-bottom:14px;flex-wrap:wrap}
.status-pill{font-family:var(--f-mono);font-size:11px;padding:4px 10px;background:var(--bg-3);border:1px solid;border-radius:20px;color:var(--tx-2)}
.btn-cleanup{padding:7px 12px;background:rgba(241,193,74,.12);color:#f1c14a;border:1px solid rgba(241,193,74,.3);border-radius:var(--r-sm);font-size:11px;font-weight:700;cursor:pointer;transition:all var(--t-fast)}
.btn-cleanup:hover{background:rgba(241,193,74,.2);border-color:#f1c14a}
.model-meta{font-size:11.5px;color:var(--tx-3);margin-top:4px}
.model-progress{margin-top:6px;font-size:10.5px;color:var(--tx-3);font-family:var(--f-mono)}
.btn-use{padding:8px 14px;background:var(--red);color:#fff;border:0;border-radius:var(--r-sm);font-size:11.5px;font-weight:700;cursor:pointer;transition:all var(--t-fast);letter-spacing:.3px}
.btn-use:hover{background:var(--red-hot)}
.btn-dl{padding:8px 14px;background:var(--bg-3);color:var(--tx);border:1px solid var(--bd-2);border-radius:var(--r-sm);font-size:11.5px;font-weight:600;cursor:pointer;transition:all var(--t-fast)}
.btn-dl:hover{border-color:var(--red);color:var(--tx)}
.drawer{position:fixed;top:0;right:0;bottom:0;width:460px;background:var(--bg-1);border-left:1px solid var(--bd-2);transform:translateX(100%);transition:transform var(--t-slow);z-index:40;display:flex;flex-direction:column;box-shadow:-20px 0 60px rgba(0,0,0,.5)}
.drawer.open{transform:translateX(0)}
.drawer-head{padding:18px 24px;border-bottom:1px solid var(--bd-1);display:flex;align-items:center;justify-content:space-between}
.drawer-head h3{font-family:var(--f-display);font-size:20px;color:var(--tx);margin:0;font-weight:400}
.drawer-head h3 small{display:block;font-family:var(--f-ui);font-size:9.5px;font-weight:600;color:var(--tx-3);letter-spacing:1.5px;text-transform:uppercase;margin-top:3px}
.drawer-body{flex:1;overflow-y:auto;padding:24px}
.drawer-section{margin-bottom:22px}
.drawer-label{display:flex;align-items:center;justify-content:space-between;font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:var(--tx-3);font-weight:700;margin-bottom:8px}
.drawer-label .hint{text-transform:none;letter-spacing:0;color:var(--tx-4);font-weight:500;font-size:11px}
.drawer-input,.drawer-textarea{width:100%;padding:10px 12px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-md);font-family:var(--f-mono);font-size:12.5px;color:var(--tx);outline:none;transition:all var(--t-fast);resize:vertical}
.drawer-textarea{min-height:80px;line-height:1.5}
.drawer-input:focus,.drawer-textarea:focus{border-color:var(--red);box-shadow:0 0 0 3px var(--red-soft)}
.drawer-input::placeholder,.drawer-textarea::placeholder{color:var(--tx-4)}
.drawer-actions{display:flex;gap:8px;padding:16px 24px;border-top:1px solid var(--bd-1);background:var(--bg-1)}
.btn-save{flex:1;padding:10px;background:linear-gradient(135deg,var(--red) 0%,var(--red-deep) 100%);border:0;border-radius:var(--r-md);color:#fff;font-size:12.5px;font-weight:700;letter-spacing:.4px;cursor:pointer;box-shadow:0 3px 10px rgba(225,29,72,.25);transition:all var(--t-fast);text-transform:uppercase}
.btn-save:hover{box-shadow:0 5px 16px rgba(225,29,72,.35);transform:translateY(-1px)}
.btn-preview{padding:10px 16px;background:var(--bg-2);border:1px solid var(--bd-2);border-radius:var(--r-md);color:var(--tx-2);font-size:12px;font-weight:600;cursor:pointer;transition:all var(--t-fast);text-transform:uppercase;letter-spacing:.4px}
.btn-preview:hover{border-color:var(--red);color:var(--tx)}
.prompt-preview{display:none;margin-top:12px;background:var(--bg-2);border:1px solid var(--bd-1);border-radius:var(--r-md);padding:12px;font-family:var(--f-mono);font-size:11px;color:var(--tx-2);max-height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.55}
.prompt-preview.show{display:block;animation:fadeIn var(--t-med)}
.drawer-checkbox-label{display:flex;align-items:flex-start;gap:10px;cursor:pointer;color:var(--tx-2);font-size:12px;user-select:none;padding:4px 0}
.drawer-checkbox-label input[type="checkbox"]{margin-top:3px;cursor:pointer;accent-color:var(--red)}
.drawer-checkbox-label strong{color:var(--tx)}
.drawer-checkbox-label .hint{display:block;margin-top:2px;color:var(--tx-4);font-size:11px}
.toast{position:fixed;top:80px;right:24px;padding:12px 16px;background:var(--bg-2);border:1px solid var(--bd-2);border-left:3px solid var(--red);border-radius:var(--r-md);color:var(--tx);font-size:13px;font-weight:500;box-shadow:0 10px 30px rgba(0,0,0,.4);z-index:60;opacity:0;transform:translateX(20px);transition:all var(--t-med);pointer-events:none;max-width:340px}
.toast.show{opacity:1;transform:translateX(0)}
.toast.err{border-left-color:var(--err)}
.toast.warn{border-left-color:var(--warn)}
.toast.ok{border-left-color:var(--ok)}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{width:12px;height:12px;border:2px solid var(--bd-3);border-top-color:var(--red);border-radius:50%;animation:spin .8s linear infinite;display:inline-block}
.empty-row{text-align:center;padding:40px;color:var(--tx-3);font-style:italic;font-size:12px}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--bd-3)}
::selection{background:var(--red);color:#fff}

/* ─── Compact / narrow fields (model selectors) ──────────────── */
.field.narrow{min-width:148px;flex-shrink:0}
.field.narrow .field-control{padding:0 12px}
.field.narrow .field-icon{width:13px;height:13px;color:var(--tx-3);flex-shrink:0}

/* ─── Toolbar separator ─────────────────────────────────────── */
.toolbar-sep{width:1px;height:32px;background:var(--bd-1);flex-shrink:0;margin:0 2px}
.toolbar-spacer{flex:1;min-width:12px}

/* ─── Toolbar horizontal scroll (when narrow) ─────────────── */
header,.toolbar,main,.stats-bar,footer{min-width:0}
.toolbar{width:100%;max-width:100%;min-width:0;overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;scrollbar-width:thin;scrollbar-color:var(--bd-1) transparent}
.toolbar::-webkit-scrollbar{height:5px}
.toolbar::-webkit-scrollbar-track{background:transparent}
.toolbar::-webkit-scrollbar-thumb{background:var(--bd-1);border-radius:3px}
.toolbar::-webkit-scrollbar-thumb:hover{background:var(--bd-2)}
.toolbar.no-scroll{overflow-x:visible}

/* ─── Stat-pill + footer chip variants for narrow screens ──── */
.stat-pill{flex-shrink:0}
.footer-chip{flex-shrink:0}

/* ═══════════════════════════════════════════════════════════════
   RESPONSIVE — tablet & mobile
   ═══════════════════════════════════════════════════════════════ */

/* Tablet portrait & small laptop (≤1023px) */
@media (max-width:1023px){
  .app{grid-template-rows:64px auto 1fr auto auto;height:100vh}
  .toolbar{flex-wrap:wrap;padding:12px 18px;gap:10px}
  .toolbar-sep{display:none}
  .toolbar-spacer{display:none}
  main{grid-template-columns:1fr;grid-template-rows:1fr 1fr;gap:1px}
  .pane-head{padding:14px 18px}
  .feed{padding:14px 18px}
  .stats-bar{padding:10px 18px;height:auto;flex-wrap:wrap;gap:10px}
  .waveform{order:99;flex:1 0 100%;height:30px}
  footer{padding:8px 18px;height:auto;flex-wrap:wrap;gap:8px}
  .field.narrow{min-width:140px}
  .field{min-width:140px}
}

/* Mobile (≤640px) */
@media (max-width:640px){
  body{font-size:13px}
  .app{grid-template-rows:56px auto 1fr auto auto}
  header{padding:0 14px;gap:10px}
  .brand-mark{width:34px;height:34px}
  .brand-mark svg{width:18px;height:18px}
  .brand-name{font-size:20px}
  .brand-tag{display:none}
  .header-spacer{flex:0 0 8px}
  .status-pill{padding:5px 9px;font-size:10.5px;gap:6px}
  .status-pill #status-text{display:none}
  .status-pill .dot{width:7px;height:7px}
  .status-pill .clock{font-size:10px}
  .header-actions{gap:4px}
  .icon-btn{width:34px;height:34px}
  .icon-btn svg{width:16px;height:16px}

  .toolbar{flex-wrap:wrap;padding:10px 14px;gap:6px}
  .field{flex:1 1 calc(50% - 3px);min-width:0;max-width:calc(50% - 3px)}
  .field.narrow{flex:1 1 calc(50% - 3px);min-width:0;max-width:calc(50% - 3px)}
  .field-label{font-size:9px;letter-spacing:1.1px}
  .field-control{height:38px;padding:0 10px;min-width:0;max-width:100%}
  .field-control select{width:100%;min-width:0;max-width:calc(100% - 24px);text-overflow:ellipsis}
  .field-icon{display:none}
  .swap-btn{display:none}
  .toolbar-spacer{display:none}
  .rec-btn{flex:1 1 100%;justify-content:center;margin-top:4px;height:44px;font-size:11px}

  main{grid-template-rows:1fr 1fr}
  .pane-head{padding:11px 14px;gap:8px}
  .pane-title{gap:8px}
  .pane-label{font-size:10px}
  .pane-lang{font-size:9.5px;padding:2px 6px}
  .pane-btn{padding:5px 9px;font-size:10.5px}
  .feed{padding:12px 14px}
  .empty{padding:24px 12px;gap:10px}
  .empty-icon{width:34px;height:34px}
  .empty-title{font-size:13px}
  .empty-sub{font-size:11px;max-width:240px}
  .seg{padding:9px 11px;gap:10px}
  .seg-text{font-size:13px;line-height:1.45}
  .seg-meta{font-size:9px;gap:6px}

  .stats-bar{padding:8px 12px;gap:6px}
  .stat-pill{padding:4px 8px;font-size:10px}
  .stat-pill .l{font-size:8px}
  .waveform{height:26px;padding:0 6px}

  footer{padding:6px 12px;font-size:10px;gap:8px;justify-content:center}
  .backend-row{gap:4px}
  .footer-chip{padding:3px 6px;font-size:9.5px;gap:4px}
  .footer-chip .dot{width:5px;height:5px}
  .footer-meta{font-size:10px;flex-wrap:wrap;justify-content:center;gap:4px}
  .footer-meta .sep{display:none}

  .toast{top:64px;right:12px;left:12px;max-width:none;font-size:12px;padding:10px 14px}
  .drawer{width:100%;max-width:none}
  .modal{margin:14px;max-height:calc(100vh - 28px)}
  .modal-body{padding:16px}
}

/* Very narrow (≤380px) — keep everything one column even tighter */
@media (max-width:380px){
  .stats-bar{gap:4px}
  .stat-pill{flex:1 1 calc(50% - 2px);justify-content:center}
  .waveform{flex:1 0 100%}
  /* Hide tune icon to make room for status pill */
  #open-tune{display:none}
  .header-actions{gap:2px}
  .icon-btn{width:30px;height:30px}
  .icon-btn svg{width:14px;height:14px}
  .status-pill{padding:4px 8px}
}

/* Touch targets — slightly larger on coarse pointer */
@media (pointer:coarse){
}

/* Reduce-motion: honor user preference */
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:0.01ms !important;animation-iteration-count:1 !important;transition-duration:0.01ms !important}
}

</style>
</head>
<body>
<div class="app">
  <header id="hdr">
    <div class="brand">
      <div class="brand-mark">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
          <path d="M6 4v16M6 4h12M6 12h8"/>
          <circle cx="19" cy="5" r=".8" fill="currentColor" stroke="none"/>
          <circle cx="21" cy="7.5" r=".6" fill="currentColor" stroke="none" opacity=".7"/>
          <circle cx="18" cy="7" r=".5" fill="currentColor" stroke="none" opacity=".5"/>
        </svg>
      </div>
      <div class="brand-text">
        <div class="brand-name">Foundry</div>
        <div class="brand-tag">Realtime Voice Intelligence · <b>The Factory Group</b></div>
      </div>
    </div>
    <div class="header-spacer"></div>
    <div class="status-pill" id="status-pill">
      <span class="dot"></span>
      <span id="status-text">Idle</span>
      <span class="clock" id="clock">00:00:00</span>
    </div>
    <button class="icon-btn" id="open-models" title="Models">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
    </button>
    <button class="icon-btn" id="open-tune" title="Tune translation">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 10v6M4.22 4.22l4.24 4.24m7.08 7.08l4.24 4.24M1 12h6m10 0h6M4.22 19.78l4.24-4.24m7.08-7.08l4.24-4.24"/></svg>
    </button>
  </header>

  <div class="toolbar">
    <div class="field">
      <div class="field-label">Audio Input</div>
      <div class="field-control">
        <svg class="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3"/></svg>
        <select id="device"></select>
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </div>
    <div class="field">
      <div class="field-label">From</div>
      <div class="field-control">
        <select id="src"></select>
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </div>
    <button class="swap-btn" id="swap" title="Swap languages">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
    </button>
    <div class="field">
      <div class="field-label">To</div>
      <div class="field-control">
        <select id="tgt"></select>
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </div>
    <div class="toolbar-sep"></div>
    <div class="field narrow" title="Whisper STT model">
      <div class="field-label">Whisper</div>
      <div class="field-control">
        <svg class="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h2l3-9 4 18 3-9 3 6h5"/></svg>
        <select id="wmodel"></select>
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </div>
    <div class="field narrow" title="Translation model">
      <div class="field-label">Model</div>
      <div class="field-control">
        <svg class="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2a4 4 0 0 0 8 0V6a4 4 0 0 0-4-4zM5 10a7 7 0 0 0 14 0M12 17v5M8 22h8"/></svg>
        <select id="model"></select>
        <svg class="chevron" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </div>
    <div class="toolbar-spacer"></div>
    <button class="rec-btn" id="go">
      <span class="d"></span>
      <span class="stop-ic"></span>
      <span id="go-text">Start Listening</span>
    </button>
  </div>

  <main>
    <section class="pane src">
      <div class="pane-head">
        <div class="pane-title">
          <span class="pane-label">Source</span>
          <span class="pane-lang" id="src-tag">German</span>
        </div>
        <div class="pane-actions">
          <button class="pane-btn" id="clear">Clear</button>
        </div>
      </div>
      <div class="feed" id="src-feed">
        <div class="empty">
          <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3"/></svg>
          <div class="empty-title">Awaiting voice input</div>
          <div class="empty-sub">Press <b>Start Listening</b> to capture audio from BlackHole 2ch.</div>
        </div>
      </div>
    </section>
    <section class="pane tgt">
      <div class="pane-head">
        <div class="pane-title">
          <span class="pane-label">Translation</span>
          <span class="pane-lang" id="tgt-tag">Thai</span>
        </div>
        <div class="pane-actions">
          <button class="pane-btn" id="copy-th">Copy Thai</button>
          <button class="pane-btn" id="copy-both">Copy Both</button>
          <button class="pane-btn" id="save-file">Save</button>
        </div>
      </div>
      <div class="feed" id="tgt-feed">
        <div class="empty">
          <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
          <div class="empty-title">No translations yet</div>
          <div class="empty-sub">Translations will stream here as speech is detected.</div>
        </div>
      </div>
    </section>
  </main>

  <div class="stats-bar">
    <div class="waveform" id="waveform"></div>
    <div class="stat-pill" id="stat-stt"><span class="l">STT</span><span class="v">—</span></div>
    <div class="stat-pill" id="stat-llm"><span class="l">LLM</span><span class="v">—</span></div>
    <div class="stat-pill" id="stat-queue"><span class="l">Q</span><span class="v">0</span></div>
    <div class="stat-pill" id="stat-model"><span class="l">Model</span><span class="v">—</span></div>
  </div>

  <footer>
    <div class="backend-row" id="backend-chips"></div>
    <div class="footer-meta">
      <span><b>Foundry</b> v3</span>
      <span class="sep">·</span>
      <span>The Factory Group</span>
      <span class="sep">·</span>
      <span id="count">0 segments</span>
    </div>
  </footer>
</div>

<div class="modal-bg" id="modal-bg">
  <div class="modal">
    <div class="modal-head">
      <h2>Models<small>The Factory Group · Foundry</small></h2>
      <button class="modal-close" id="modal-close">&times;</button>
    </div>
    <div class="modal-tabs">
      <button data-tab="whisper" class="active">Whisper (STT)</button>
      <button data-tab="ollama">Ollama (Translate)</button>
      <button data-tab="mlx">MLX (fallback)</button>
    </div>
    <div class="modal-body" id="modal-body"><div class="empty-row">Loading models…</div></div>
  </div>
</div>

<div class="drawer" id="tune-drawer">
  <div class="drawer-head">
    <h3>Tune Translation<small>The Factory Group · Foundry</small></h3>
    <button class="modal-close" id="tune-close">&times;</button>
  </div>
  <div class="drawer-body">
    <div class="drawer-section">
      <div class="drawer-label">Glossary <span class="hint">term=translation, comma-separated</span></div>
      <textarea class="drawer-textarea" id="glossary" rows="3" placeholder="z.B. Krankenversicherung=Krankenversicherung, M2024=Meeting 2024"></textarea>
    </div>
    <div class="drawer-section">
      <div class="drawer-label">Custom system prompt</div>
      <textarea class="drawer-textarea" id="custom-prompt" rows="4" placeholder="Additional rules for the translator (tone, formality, terminology)…"></textarea>
    </div>
    <div class="drawer-section">
      <label class="drawer-checkbox-label">
        <input type="checkbox" id="concise-translation">
        <div>
          <strong>Concise Translation (แปลกระชับ)</strong>
          <span class="hint">Shortens the translation to reduce latency and speed up response times. (ช่วยลดความหน่วง/สั้นขึ้น)</span>
        </div>
      </label>
    </div>
    <div class="drawer-section">
      <label class="drawer-checkbox-label">
        <input type="checkbox" id="use-mlx-whisper">
        <div>
          <strong>Use MLX-Whisper (In-Process STT)</strong>
          <span class="hint">If unchecked, runs external whisper-cli which releases RAM instantly after transcription. (ช่วยประหยัด RAM ~1.6GB)</span>
        </div>
      </label>
    </div>
    <div class="drawer-section">
      <label class="drawer-checkbox-label">
        <input type="checkbox" id="use-mlx-lm">
        <div>
          <strong>Use MLX-LLM (In-Process Translate)</strong>
          <span class="hint">If unchecked, delegates translation to Ollama server, avoiding high Python memory usage. (ช่วยประหยัด RAM ~5.5GB)</span>
        </div>
      </label>
    </div>
    <div class="drawer-section">
      <div class="drawer-label">Prompt preview</div>
      <div class="prompt-preview" id="prompt-preview"></div>
    </div>
  </div>
  <div class="drawer-actions">
    <button class="btn-preview" id="preview-prompts">Preview</button>
    <button class="btn-save" id="save-tune">Save Changes</button>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
(function(){
  "use strict";
  const $ = (id) => document.getElementById(id);
  const LANGS = ["English","Thai","Japanese","Chinese (Simplified)","Chinese (Traditional)","Korean","French","German","Spanish","Portuguese","Russian","Italian","Vietnamese","Indonesian","Arabic","Hindi","Burmese","Lao","Khmer","Malay","Turkish","Dutch","Polish","Swedish"];
  const WHISPER_CATALOG = [
    {name:"tiny",           size:75,    desc:"Fastest — low accuracy"},
    {name:"base",           size:141,   desc:"Fast — adequate"},
    {name:"small",          size:466,   desc:"Balanced"},
    {name:"medium",         size:1500,  desc:"Accurate"},
    {name:"large-v3-turbo", size:1600,  desc:"⭐ Recommended — fast & accurate"},
    {name:"large-v3",       size:3100,  desc:"Most accurate, slowest"},
  ];
  const OLLAMA_CATALOG = [
    {name:"qwen2.5:0.5b",   desc:"Fastest — low quality"},
    {name:"gemma3:1b",      desc:"⭐ Recommended (Gemma 3) — very light & fast"},
    {name:"qwen2.5:1.5b",   desc:"⭐ Recommended (Qwen 2.5) — very light & fast"},
    {name:"gemma2:2b",      desc:"Light — good quality"},
    {name:"qwen2.5:3b",     desc:"Balanced — accurate & fast"},
    {name:"gemma3:4b",      desc:"⭐ Recommended (Gemma 3) — smart & fast"},
    {name:"qwen2.5:7b",     desc:"Accurate, slower"},
    {name:"gemma2:9b",      desc:"Strong, slower"},
    {name:"llama3.1:8b",    desc:"Multilingual"},
    {name:"mistral:7b",     desc:"Fast, good"},
    {name:"phi3:mini",      desc:"Lightweight"},
  ];

  // ---------- State ----------
  const state = {
    listening: false, model: "", wmodel: "", device: "",
    src_lang: "German", tgt_lang: "Thai",
    history: [], glossary: "", custom_prompt: "",
    concise_translation: false,
    use_mlx_whisper: false, use_mlx_lm: true,
    backends: null,
  };

  // ---------- Helpers ----------
  async function postControl(payload){
    const r = await fetch("/api/control", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
    return r.json();
  }
  function escapeHTML(s){
    return (s||"").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }
  function formatTime(ts){
    return new Date(ts*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
  }
  function basename(p){ return (p||"").split("/").pop(); }
  function fmtSize(mb){ return mb >= 1000 ? (mb/1024).toFixed(1)+" GB" : mb+" MB"; }

  // ---------- Build waveform bars ----------
  const WAVE_BARS = 56;
  const waveformEl = $("waveform");
  const waveBars = [];
  for (let i=0; i<WAVE_BARS; i++){
    const b = document.createElement("div");
    b.className = "wave-bar";
    b.style.height = "2px";
    waveformEl.appendChild(b);
    waveBars.push(b);
  }
  let waveTarget = new Array(WAVE_BARS).fill(0);
  let waveCurrent = new Array(WAVE_BARS).fill(0);
  function pushWaveLevel(v){
    // shift left, append new at the right
    waveTarget.shift();
    waveTarget.push(Math.max(0.02, Math.min(1, v*1.4)));
  }
  function tickWave(){
    for (let i=0; i<WAVE_BARS; i++){
      // smooth current towards target
      waveCurrent[i] += (waveTarget[i] - waveCurrent[i]) * 0.35;
      const h = Math.max(2, waveCurrent[i] * 32);
      waveBars[i].style.height = h + "px";
    }
    requestAnimationFrame(tickWave);
  }
  tickWave();

  // ---------- Initial setup ----------
  function initSelects(){
    const srcSel = $("src"), tgtSel = $("tgt");
    LANGS.forEach(L => { srcSel.add(new Option(L,L)); tgtSel.add(new Option(L,L)); });
    srcSel.value = state.src_lang; tgtSel.value = state.tgt_lang;
  }
  async function loadModels(){
    try {
      const r = await fetch("/api/models");
      const models = await r.json();
      const sel = $("model");
      if (!sel) return;
      sel.innerHTML = "";
      if (!models.length){ sel.add(new Option("— Ollama offline —", "")); return; }
      models.forEach(m => sel.add(new Option(m, m)));
      // Restore selection from state (SSE may have fired before options populated)
      if (state.model && [...sel.options].some(o => o.value === state.model)){
        sel.value = state.model;
      }
    } catch(e){ console.error(e); }
  }
  async function loadWmodels(){
    try {
      const r = await fetch("/api/wmodels");
      const models = await r.json();
      const sel = $("wmodel");
      if (!sel) return;
      sel.innerHTML = "";
      if (!models.length){ sel.add(new Option("— no model —", "")); return; }
      models.forEach(p => {
        // Display: "ggml-large-v3-turbo" (no .bin) — saves space in narrow UIs
        const label = basename(p).replace(/^ggml-|\.bin$/g, "");
        sel.add(new Option(label, p));
      });
      // Restore selection from state (SSE may have fired before options populated)
      if (state.wmodel && [...sel.options].some(o => o.value === state.wmodel)){
        sel.value = state.wmodel;
      }
    } catch(e){ console.error(e); }
  }
  async function loadDevices(){
    try {
      const r = await fetch("/api/devices");
      const devs = await r.json();
      const sel = $("device");
      sel.innerHTML = "";
      devs.forEach(d => sel.add(new Option(d.name + " (" + d.channels + "ch)", d.name)));
      if (state.device) sel.value = state.device;
    } catch(e){ console.error(e); }
  }

  // ---------- Status & clock ----------
  function setStatus(text, kind){
    $("status-text").textContent = text;
    const pill = $("status-pill");
    pill.classList.toggle("live", state.listening);
  }
  let startTime = null;
  function tickClock(){
    if (state.listening && startTime){
      const sec = Math.floor((Date.now() - startTime)/1000);
      const h = String(Math.floor(sec/3600)).padStart(2,"0");
      const m = String(Math.floor((sec%3600)/60)).padStart(2,"0");
      const s = String(sec%60).padStart(2,"0");
      $("clock").textContent = h+":"+m+":"+s;
    } else {
      const now = new Date();
      $("clock").textContent = now.toTimeString().slice(0,8);
    }
  }
  setInterval(tickClock, 1000);

  // ---------- Recording state ----------
  function setListeningUI(on){
    state.listening = on;
    $("hdr").classList.toggle("recording", on);
    waveformEl.classList.toggle("active", on);
    $("status-pill").classList.toggle("live", on);
    const btn = $("go");
    btn.classList.toggle("live", on);
    $("go-text").textContent = on ? "Stop Listening" : "Start Listening";
    if (on){ startTime = Date.now(); }
    else { startTime = null; waveTarget = new Array(WAVE_BARS).fill(0); }
  }

  // ---------- Stat pills ----------
  function setStatSTT(s){ $("stat-stt").querySelector(".v").textContent = s != null ? s.toFixed(1)+"s" : "—"; }
  function setStatLLM(s){ $("stat-llm").querySelector(".v").textContent = s != null ? s.toFixed(1)+"s" : "—"; }
  function setStatQueue(n){ $("stat-queue").querySelector(".v").textContent = n; }
  function setStatModel(m){ $("stat-model").querySelector(".v").textContent = m; }

  // ---------- Render entries ----------
  function renderEntry(e, fresh){
    document.querySelectorAll(".empty").forEach(n => n.remove());
    const lat = e.latency || {};
    const metaHtml = '<div class="entry-meta">'
      + '<span class="ts">'+formatTime(e.ts)+'</span>'
      + '<span class="dot"></span>'
      + (lat.stt != null ? '<span class="lat">STT '+lat.stt.toFixed(1)+'s</span><span class="dot"></span>' : '')
      + (lat.llm != null ? '<span class="lat">LLM '+lat.llm.toFixed(1)+'s</span>' : '')
      + '</div>';

    const src = document.createElement("div");
    src.className = "entry" + (fresh ? " fresh" : "");
    src.innerHTML = metaHtml + '<div class="entry-text">'+escapeHTML(e.src)+'</div>';
    $("src-feed").prepend(src);

    const tgt = document.createElement("div");
    tgt.className = "entry" + (fresh ? " fresh" : "");
    tgt.innerHTML = metaHtml + '<div class="entry-text">'+escapeHTML(e.tgt)+'</div>';
    $("tgt-feed").prepend(tgt);

    // cap visible to 50
    ["src-feed","tgt-feed"].forEach(id => {
      const f = $(id);
      while (f.children.length > 50) f.lastChild.remove();
    });
    $("count").textContent = state.history.length + " segments";
  }
  function rebuildFeeds(){
    $("src-feed").innerHTML = ""; $("tgt-feed").innerHTML = "";
    if (!state.history.length){
      $("src-feed").innerHTML = '<div class="empty"><svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3"/></svg><div class="empty-title">Awaiting voice input</div><div class="empty-sub">Press <b>Start Listening</b> to capture audio from BlackHole 2ch.</div></div>';
      $("tgt-feed").innerHTML = '<div class="empty"><svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg><div class="empty-title">No translations yet</div><div class="empty-sub">Translations will stream here as speech is detected.</div></div>';
      return;
    }
    [...state.history].reverse().forEach(e => renderEntry(e, false));
  }

  // ---------- Backend chips ----------
  function renderBackendChips(b){
    if (!b) return;
    const chips = [
      ['vad',        b.vad,         'VAD', false],
      ['denoise',    b.denoise,     'Denoise', false],
      ['memory',     (b.memory_size||0) > 0, 'Mem×'+(b.memory_size||0), false],
      ['mlx-stt',    b.mlx_whisper, 'MLX-STT', true],
      ['mlx-lm',     b.mlx_lm,      'MLX-LLM', true],
    ];
    $("backend-chips").innerHTML = chips.map(([k,on,label,mlx]) => {
      const cls = on ? "on" : "";
      const mlxCls = (mlx && on) ? "mlx on" : cls;
      const finalCls = mlx ? mlxCls : cls;
      return '<span class="backend-chip '+finalCls+'"><span class="d"></span>'+label+'</span>';
    }).join("");
  }

  // ---------- Toast ----------
  let _toastTimer = null;
  function toast(msg, kind){
    const t = $("toast");
    t.textContent = msg;
    t.className = "toast " + (kind || "ok") + " show";
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
  }

  // ---------- Model manager modal ----------
  function showModal(open){ $("modal-bg").classList.toggle("open", open); }
  async function openModelManager(tab){
    showModal(true);
    document.querySelectorAll(".modal-tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
    const body = $("modal-body");
    body.innerHTML = '<div class="empty-row">Loading models…</div>';
    // Aggregate status from server (ollama + whisper + mlx + RAM + incomplete).
    let status = null;
    try { status = await (await fetch("/api/models_status")).json(); }
    catch(e){ status = null; }
    const ramFree = status && status.system && status.system.ram_free_mb;
    const ramTotal = status && status.system && status.system.ram_total_mb;
    const ramWarn  = ramFree && ramTotal && ramFree < 4000;  // < 4 GB free
    const incompl  = (status && status.incomplete_downloads) || [];
    // Build a header strip with system info + incomplete-warning + cleanup button
    let header = "";
    if (ramTotal){
      const pct = ramFree ? Math.round(100 * ramFree / ramTotal) : 0;
      const ramColor = ramWarn ? "#ff5e5e" : (pct > 30 ? "#5edda5" : "#f1c14a");
      header += '<div class="status-strip">'
        + '<span class="status-pill" style="border-color:'+ramColor+'">RAM: '
        + (ramFree||'?')+' MB free / '+ramTotal+' MB total ('+pct+'%)</span>';
      if (incompl.length){
        const totalMb = incompl.reduce((a,b)=>a+b.size_mb,0);
        header += '<button class="btn-cleanup" data-action="cleanup">'
          + '🧹 Clean '+incompl.length+' incomplete ('+totalMb+' MB)</button>';
      }
      header += '</div>';
    }
    if (tab === "whisper"){
      const installed = (status && status.whisper && status.whisper.installed) || [];
      const installedSet = new Set(installed.map(p => basename(p).replace(/^ggml-|\.bin$/g, "")));
      const currentName = basename(state.wmodel).replace(/^ggml-|\.bin$/g, "");
      body.innerHTML = header;
      WHISPER_CATALOG.forEach(item => {
        const has = installedSet.has(item.name);
        const active = item.name === currentName;
        // RAM check: model size * 2 (rough heuristic — whisper loads model fully)
        const ramNeeded = item.size * 2;
        const tooBig = ramFree && ramNeeded > ramFree;
        const row = document.createElement("div");
        row.className = "model-row" + (active ? " active" : "");
        const ramWarnSpan = tooBig
          ? '<span class="badge warn">⚠ needs ~'+fmtSize(ramNeeded)+' RAM (only '+fmtSize(ramFree)+' free)</span>'
          : '';
        row.innerHTML =
          '<div class="model-info">'
            + '<div class="model-name">'
              + (active ? '<span class="star">★</span>' : '')
              + '<span>ggml-'+item.name+'.bin</span>'
              + '<span class="badge '+(has?"":"miss")+'">'+ (has?"INSTALLED":"NOT INSTALLED") +'</span>'
              + ramWarnSpan
            + '</div>'
            + '<div class="model-meta">'+item.desc+' · '+fmtSize(item.size)+'</div>'
            + '<div class="model-progress" data-progress></div>'
          + '</div>'
          + (has
            ? '<button class="btn-use" data-action="use" data-name="'+item.name+'">'+(active?"✓ In use":"Use")+'</button>'
            : '<button class="btn-dl" data-action="dl" data-name="'+item.name+'"'+(tooBig?' title="Not enough RAM — close apps first"':'')+'>⬇ Download</button>');
        body.appendChild(row);
      });
    } else if (tab === "ollama"){
      let installed = (status && status.ollama && status.ollama.installed) || [];
      const installedSet = new Set(installed);
      body.innerHTML = header;
      OLLAMA_CATALOG.forEach(item => {
        const has = installedSet.has(item.name);
        const active = item.name === state.model;
        const row = document.createElement("div");
        row.className = "model-row" + (active ? " active" : "");
        row.innerHTML =
          '<div class="model-info">'
            + '<div class="model-name">'
              + (active ? '<span class="star">★</span>' : '')
              + '<span>'+item.name+'</span>'
              + '<span class="badge '+(has?"":"miss")+'">'+ (has?"INSTALLED":"NOT INSTALLED") +'</span>'
            + '</div>'
            + '<div class="model-meta">'+item.desc+'</div>'
            + '<div class="model-progress" data-progress></div>'
          + '</div>'
          + (has
            ? '<button class="btn-use" data-action="use-ollama" data-name="'+item.name+'">'+(active?"✓ In use":"Use")+'</button>'
            : '<button class="btn-dl" data-action="pull" data-name="'+item.name+'">⬇ Pull</button>');
        body.appendChild(row);
      });
    } else if (tab === "mlx"){
      // Show MLX in-process backends used when Ollama is down/too slow.
      const mlxInstalled = (status && status.mlx && status.mlx.installed) || [];
      const mlxCatalog = (status && status.mlx && status.mlx.catalog) || [];
      const installedMap = {};
      mlxInstalled.forEach(m => { installedMap[m.hf_repo] = m; });
      body.innerHTML = header;
      mlxCatalog.forEach(item => {
        const inst = installedMap[item.hf_repo];
        const have = !!inst;
        const need = item.ram_mb || item.size_mb;
        const tooBig = ramFree && need > ramFree;
        const row = document.createElement("div");
        row.className = "model-row";
        const ramWarnSpan = tooBig
          ? '<span class="badge warn">⚠ needs ~'+fmtSize(need)+' RAM (only '+fmtSize(ramFree)+' free)</span>'
          : (have ? '<span class="badge ok">✓ '+fmtSize(inst.size_mb)+' cached</span>' : '');
        row.innerHTML =
          '<div class="model-info">'
            + '<div class="model-name">'
              + '<span>'+item.name+'</span>'
              + '<span class="badge '+(have?"":"miss")+'">'+ (have?"DOWNLOADED":"NOT DOWNLOADED") +'</span>'
              + ramWarnSpan
            + '</div>'
            + '<div class="model-meta">'+item.hf_repo+' · '+fmtSize(item.size_mb)+' disk · '+fmtSize(need)+' RAM</div>'
            + '<div class="model-progress" data-progress></div>'
          + '</div>'
          + (have
            ? '<span class="meta-muted">auto-loaded</span>'
            : '<span class="meta-muted">downloads on first use</span>');
        body.appendChild(row);
      });
      if (incompl.length){
        const note = document.createElement("div");
        note.className = "model-row";
        note.innerHTML = '<div class="model-info"><div class="model-meta" style="color:#f1c14a">'
          +'⚠ '+incompl.length+' incomplete download(s) in HF cache ('+incompl.reduce((a,b)=>a+b.size_mb,0)+' MB). Click "Clean" above to free space.'
          +'</div></div>';
        body.appendChild(note);
      }
    }
  }
  async function handleModalClick(e){
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action, name = btn.dataset.name;
    // cleanup button may live outside .model-row (it's in the status strip)
    const row = btn.closest(".model-row");
    const prog = row ? row.querySelector("[data-progress]") : null;
    if (action === "cleanup"){
      btn.disabled = true; btn.textContent = "Cleaning…";
      try {
        const r = await (await fetch("/api/cleanup_incomplete", {method:"POST"})).json();
        if (r.ok){
          toast("✓ Removed "+r.removed+" file(s), freed "+r.freed_mb+" MB", "ok");
          openModelManager("whisper");
        } else { toast("✗ Cleanup failed", "err"); btn.disabled = false; btn.textContent = "🧹 Clean"; }
      } catch(e){ toast("✗ "+e.message, "err"); btn.disabled = false; btn.textContent = "🧹 Clean"; }
      return;
    }
    if (action === "use"){
      const installed = await (await fetch("/api/wmodels")).json();
      const match = installed.find(p => p.includes("ggml-"+name+".bin"));
      if (match){
        await postControl({wmodel: match});
        state.wmodel = match;
        toast("✓ Switched to "+basename(match), "ok");
        openModelManager("whisper");
      }
    } else if (action === "use-ollama"){
      await postControl({model: name});
      state.model = name;
      const msel = $("model"); if (msel) msel.value = name;
      toast("✓ Switched to "+name, "ok");
      openModelManager("ollama");
    } else if (action === "dl"){
      btn.disabled = true; btn.textContent = "Starting…";
      prog.innerHTML = "Initialising download…";
      try {
        await fetch("/api/download_wmodel", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({name})
        });
      } catch(e){ prog.textContent = "✗ "+e.message; btn.disabled = false; return; }
      const waitFor = setInterval(async () => {
        const installed = await (await fetch("/api/wmodels")).json();
        const have = installed.find(p => p.includes("ggml-"+name+".bin"));
        if (have){ clearInterval(waitFor); toast("✓ Downloaded ggml-"+name+".bin", "ok"); openModelManager("whisper"); }
      }, 3000);
    } else if (action === "pull"){
      btn.disabled = true; btn.textContent = "Pulling…";
      prog.textContent = "Running ollama pull… (may take minutes)";
      try {
        const r = await fetch("/api/ollama_pull", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({name})
        });
        const data = await r.json();
        if (data.ok){ prog.textContent = "✓ Pulled — refreshing…"; toast("✓ Pulled "+name, "ok"); setTimeout(()=>openModelManager("ollama"), 1000); }
        else { prog.textContent = "✗ "+(data.error||"failed"); btn.disabled = false; btn.textContent = "⬇ Pull"; }
      } catch(e){ prog.textContent = "✗ "+e.message; btn.disabled = false; btn.textContent = "⬇ Pull"; }
    }
  }

  // ---------- Tune drawer ----------
  function showDrawer(open){ $("tune-drawer").classList.toggle("open", open); }
  async function hydrateTune(){
    try {
      const s = await (await fetch("/api/state")).json();
      $("glossary").value = s.glossary || "";
      $("custom-prompt").value = s.custom_prompt || "";
      $("concise-translation").checked = !!s.concise_translation;
      $("use-mlx-whisper").checked = s.use_mlx_whisper !== false;
      $("use-mlx-lm").checked = s.use_mlx_lm !== false;
    } catch(e){}
  }

  // ---------- Event wiring ----------
  $("go").addEventListener("click", async () => {
    const willListen = !state.listening;
    if (willListen) {
      // Pre-flight: peek at the input device and refuse if it's silent.
      // This catches "system audio not routed to BlackHole" before the user
      // waits 15+ seconds for transcripts that will never come.
      try {
        const deviceName = $("device").value || state.device;
        const devices = await (await fetch("/api/devices")).json();
        const info = devices.find(d => d.name === deviceName);
        if (info) {
          // Check device sample rate — BlackHole should be 48000
          // (no direct way to query peak from JS, but check device exists + non-zero input channels)
          if (!info.max_input_channels || info.max_input_channels < 1) {
            toast("⚠ Device has no input channels", "err");
          }
        } else {
          toast("⚠ Device not found: " + deviceName, "err");
        }
      } catch(e) { /* non-fatal */ }
    }
    try {
      const r = await postControl({listening: willListen});
      if (r.ok){
        setListeningUI(willListen);
        toast(willListen ? "● Listening started — if no transcript in 15s, check audio routing (system audio → "+(state.device||'BlackHole 2ch')+")" : "■ Stopped", willListen ? "ok" : "warn");
      } else if (r.error) {
        toast(r.error, "err");
      }
    } catch(e){ toast("Control error: "+e.message, "err"); }
  });
  $("device").addEventListener("change", async (e) => {
    await postControl({device: e.target.value});
    state.device = e.target.value;
    toast("Input → "+e.target.value, "ok");
  });
  $("src").addEventListener("change", async (e) => {
    await postControl({src_lang: e.target.value});
    state.src_lang = e.target.value;
    $("src-tag").textContent = e.target.value;
    toast("Source → "+e.target.value, "ok");
  });
  $("tgt").addEventListener("change", async (e) => {
    await postControl({tgt_lang: e.target.value});
    state.tgt_lang = e.target.value;
    $("tgt-tag").textContent = e.target.value;
    toast("Target → "+e.target.value, "ok");
  });
  // Translation model picker — push to server + clear stale status / errors.
  // Listen to BOTH "change" (mouse/keyboard) and "input" (programmatic) so we
  // never miss a selection, even on browsers that fire only one of them.
  async function onModelChange(e){
    const v = e.target.value;
    if (!v) return;
    console.log("[model] change →", v);
    const r = await postControl({model: v});
    console.log("[model] postControl →", r);
    if (!r || r.ok === false) {
      toast((r && r.error) || "Model switch failed", "err");
      // Revert the dropdown to actual server state
      e.target.value = state.model || "";
      return;
    }
    // Trust the server's response — re-sync dropdown + stat from r.state
    const newModel = (r.state && r.state.model) || v;
    state.model = newModel;
    e.target.value = newModel;  // force-sync visual to server value
    setStatModel(newModel);
    toast("Model → "+newModel, "ok");
    // Re-test Ollama so a fresh error appears immediately if the new model is broken
    try {
      const j = await (await fetch("/api/test_ollama")).json();
      if (j.ok) toast(j.msg, "ok");
      else { setStatus(j.msg); toast(j.msg, "err"); }
    } catch(err) { console.warn("test_ollama failed", err); }
  }
  $("model").addEventListener("change", onModelChange);
  $("model").addEventListener("input", onModelChange);
  // Whisper model picker
  async function onWmodelChange(e){
    const v = e.target.value;
    if (!v) return;
    console.log("[wmodel] change →", v);
    const r = await postControl({wmodel: v});
    if (!r || r.ok === false) {
      toast((r && r.error) || "Whisper model switch failed", "err");
      e.target.value = state.wmodel || "";
      return;
    }
    const newVal = (r.state && r.state.wmodel) || v;
    state.wmodel = newVal;
    e.target.value = newVal;
    toast("Whisper → "+basename(newVal), "ok");
  }
  $("wmodel").addEventListener("change", onWmodelChange);
  $("wmodel").addEventListener("input", onWmodelChange);
  $("swap").addEventListener("click", async () => {
    const a = $("src").value, b = $("tgt").value;
    $("src").value = b; $("tgt").value = a;
    await postControl({src_lang: b, tgt_lang: a});
    state.src_lang = b; state.tgt_lang = a;
    $("src-tag").textContent = b; $("tgt-tag").textContent = a;
    toast("Languages swapped", "ok");
  });
  $("clear").addEventListener("click", async () => {
    await postControl({clear: true});
    state.history = [];
    rebuildFeeds();
    $("count").textContent = "0 segments";
    toast("History cleared", "ok");
  });

  // Modal wiring
  $("open-models").addEventListener("click", () => openModelManager("whisper"));
  $("modal-close").addEventListener("click", () => showModal(false));
  $("modal-bg").addEventListener("click", e => { if (e.target.id === "modal-bg") showModal(false); });
  document.querySelectorAll(".modal-tabs button").forEach(b => b.addEventListener("click", () => openModelManager(b.dataset.tab)));
  $("modal-body").addEventListener("click", handleModalClick);

  // Tune drawer wiring
  $("open-tune").addEventListener("click", () => { showDrawer(true); hydrateTune(); });
  $("tune-close").addEventListener("click", () => showDrawer(false));
  $("save-tune").addEventListener("click", async () => {
    const hasDenoise = $("tog-denoise") ? $("tog-denoise").classList.contains("on") : true;
    const hasVad = $("tog-vad") ? $("tog-vad").classList.contains("on") : true;
    const hasMemory = $("tog-memory") ? $("tog-memory").classList.contains("on") : true;
    await postControl({
      glossary: $("glossary").value,
      custom_prompt: $("custom-prompt").value,
      concise_translation: $("concise-translation").checked,
      use_mlx_whisper: $("use-mlx-whisper").checked,
      use_mlx_lm: $("use-mlx-lm").checked,
      denoise: hasDenoise,
      vad_chunk: hasVad,
      use_memory: hasMemory,
    });
    toast("✓ Tune settings saved", "ok");
    showDrawer(false);
  });
  $("preview-prompts").addEventListener("click", async () => {
    const r = await fetch("/api/prompts");
    const d = await r.json();
    const pre = $("prompt-preview");
    pre.textContent = "=== Whisper prompt ===\n\n" + (d.whisper_prompt||"") +
                      "\n\n=== Translation prompt ===\n\n" + (d.translation_prompt||"");
    pre.classList.add("show");
  });

  // Copy / save
  async function copyToClipboard(text){
    try { await navigator.clipboard.writeText(text); return true; }
    catch(e){
      try {
        const ta = document.createElement("textarea");
        ta.value = text; ta.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
        document.body.appendChild(ta); ta.focus(); ta.select();
        const ok = document.execCommand("copy"); document.body.removeChild(ta);
        return ok;
      } catch(e2){ return false; }
    }
  }
  $("copy-th").addEventListener("click", async () => {
    if (!state.history.length){ toast("Nothing to copy yet", "warn"); return; }
    const txt = [...state.history].reverse().map(e => e.tgt).join("\n\n");
    if (await copyToClipboard(txt)) toast("📋 Copied "+state.history.length+" Thai lines", "ok");
    else toast("Copy failed — try ⌘+C manually", "err");
  });
  $("copy-both").addEventListener("click", async () => {
    if (!state.history.length){ toast("Nothing to copy yet", "warn"); return; }
    const items = [...state.history].reverse();
    const txt = items.map(e => "["+formatTime(e.ts)+"]\n"+e.src+"\n→ "+e.tgt).join("\n\n");
    if (await copyToClipboard(txt)) toast("📋 Copied "+items.length+" pairs", "ok");
    else toast("Copy failed", "err");
  });
  $("save-file").addEventListener("click", () => {
    if (!state.history.length){ toast("Nothing to save yet", "warn"); return; }
    const items = [...state.history].reverse();
    const txt = items.map(e => "["+formatTime(e.ts)+"]\n"+e.src+"\n→ "+e.tgt).join("\n\n");
    const blob = new Blob([txt], {type:"text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "foundry_transcript_"+new Date().toISOString().replace(/[:.]/g,"-").slice(0,19)+".txt";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("💾 Saved "+a.download, "ok");
  });

  // Keyboard: Space toggles
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
    if (e.code === "Space"){ e.preventDefault(); $("go").click(); }
  });

  // ---------- SSE ----------
  const es = new EventSource("/api/events");
  es.addEventListener("state", ev => {
    const s = JSON.parse(ev.data);
    Object.assign(state, s);
    const modelSel = $("model");
    if (modelSel && state.model && modelSel.value !== state.model) modelSel.value = state.model;
    const wmodelSel = $("wmodel");
    if (wmodelSel && state.wmodel && wmodelSel.value !== state.wmodel) wmodelSel.value = state.wmodel;
    if (state.device && $("device").value !== state.device) $("device").value = state.device;
    if (state.src_lang){ $("src").value = state.src_lang; $("src-tag").textContent = state.src_lang; }
    if (state.tgt_lang){ $("tgt").value = state.tgt_lang; $("tgt-tag").textContent = state.tgt_lang; }
    setListeningUI(state.listening);
    setStatus(state.listening ? "● Recording" : (state.history.length ? "Ready" : "Idle"));
    rebuildFeeds();
    $("count").textContent = state.history.length + " segments";
    setStatModel(state.model ? state.model : "—");
    if (s.backends) renderBackendChips(s.backends);
  });
  es.addEventListener("transcript", ev => {
    const e = JSON.parse(ev.data);
    state.history.push(e);
    renderEntry(e, true);
    if (e.latency){
      setStatSTT(e.latency.stt);
      setStatLLM(e.latency.llm);
      $("stat-stt").classList.add("ok");
      $("stat-llm").classList.add("ok");
      setTimeout(() => { $("stat-stt").classList.remove("ok"); $("stat-llm").classList.remove("ok"); }, 1500);
    }
  });
  es.addEventListener("level", ev => {
    const v = JSON.parse(ev.data).v;
    pushWaveLevel(v);
  });
  es.addEventListener("info", ev => {
    try {
      const m = JSON.parse(ev.data).msg;
      if (m.includes("silence") || m.includes("no speech")) return; // skip noise
      setStatus(m);
    } catch(e){}
  });
  es.addEventListener("error", ev => {
    try {
      const m = JSON.parse(ev.data).msg;
      setStatus(m);
      toast(m, "err");
    } catch(e){}
  });
  es.addEventListener("ok", ev => {
    try { setStatus(JSON.parse(ev.data).msg); } catch(e){}
  });
  es.addEventListener("wmodels", ev => {
    try {
      const d = JSON.parse(ev.data);
      state.wmodel = d.current;
      loadWmodels();
      toast("✓ Models updated", "ok");
    } catch(e){}
  });
  es.addEventListener("dl_progress", ev => {
    try {
      const p = JSON.parse(ev.data);
      document.querySelectorAll(".model-row").forEach(row => {
        const btn = row.querySelector("button[data-action='dl']");
        if (btn && btn.dataset.name === p.name){
          const prog = row.querySelector("[data-progress]");
          if (prog) prog.innerHTML =
            '<div style="height:5px;background:var(--bg-3);border-radius:3px;overflow:hidden;margin-bottom:4px"><div style="width:'+p.pct+'%;height:100%;background:var(--red);transition:width .2s"></div></div>'
            + '<div style="font-size:10px;color:var(--tx-3);font-family:var(--f-mono)">'+p.mb+' / '+p.total_mb+' MB · '+p.pct+'%</div>';
          btn.textContent = p.pct + "%";
        }
      });
    } catch(e){}
  });

  // ---------- Init ----------
  (async () => {
    initSelects();
    await loadDevices();
    await loadModels();
    await loadWmodels();
    try {
      const s = await (await fetch("/api/state")).json();
      Object.assign(state, s);
      if (state.device) $("device").value = state.device;
      $("src").value = state.src_lang;
      $("tgt").value = state.tgt_lang;
      $("src-tag").textContent = state.src_lang;
      $("tgt-tag").textContent = state.tgt_lang;
      rebuildFeeds();
      setListeningUI(state.listening);
      setStatModel(state.model || "—");
      if (s.backends) renderBackendChips(s.backends);
      setStatus(state.listening ? "● Recording" : "Ready");
    } catch(e){ console.error(e); }
  })();
})();
</script>
</body>
</html>
"""

class _FastHTTPServer(http.server.ThreadingHTTPServer):
    """HTTPServer variant that skips the socket.getfqdn() reverse-DNS call in
    Python 3.14's HTTPServer.server_bind — it adds a ~35s DNS timeout on every
    bind to '127.0.0.1'. We don't use the FQDN anywhere, so just leave it as
    the host string."""
    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host if host else "127.0.0.1"
        self.server_port = port


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_a, **_k):
        pass  # silence access logs

    def log_error(self, format, *args):
        # Suppress connection-reset noise from SSE/browser disconnects
        msg = (args[0] if args else "") or format
        if any(s in str(msg) for s in ("Connection reset", "Broken pipe", "errno 54")):
            return
        super().log_error(format, *args)

    def handle_one_request(self):
        """Wrap parent to swallow connection-reset errors that normally
        happen when browsers refresh / navigate during SSE streams."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.close_connection = True
        except OSError as e:
            if getattr(e, "errno", None) in (54, 32, 104):  # ECONNRESET/BROKENPIPE/ECONNABORTED
                self.close_connection = True
            else:
                raise

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/api/state":
            self._send_json(_public_state())
        elif self.path == "/api/devices":
            self._send_json(list_input_devices())
        elif self.path == "/api/models":
            self._send_json(fetch_models())
        elif self.path == "/api/models_status":
            # Aggregate status of all model backends + system RAM.
            self._send_json(models_status())
        elif self.path.startswith("/api/check_model"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            name = (qs.get("name") or [""])[0]
            if name:
                self._send_json(check_model_availability(name))
            else:
                self._send_json({"error": "missing ?name="})
        elif self.path == "/api/backends":
            # diagnostic info about which backends are active
            self._send_json({
                "whisper_cli":     {"available": os.path.exists(WHISPER_BIN),
                                     "binary": WHISPER_BIN, "threads": WHISPER_THREADS},
                "mlx_whisper":     {"available": HAS_MLX_WHISPER},
                "mlx_lm":          {"available": HAS_MLX_LM},
                "webrtcvad":       {"available": True, "aggressiveness": VAD_AGGR},
                "noisereduce":     {"available": True, "props": DENOISE_PROPS},
                "ollama":          {"url": OLLAMA_URL},
                "vad_enabled":     True,   # always-on (no UI toggle)
                "denoise_enabled": True,   # always-on (no UI toggle)
                "memory_enabled":  True,   # always-on (no UI toggle)
            })
        elif self.path == "/api/prompts":
            # return current prompts for debugging / preview in GUI
            with _state_lock:
                src = state["src_lang"]
                tgt = state["tgt_lang"]
            self._send_json({
                "whisper_prompt":    get_whisper_prompt(src),
                "translation_prompt": get_translation_prompt(src, tgt),
            })
        elif self.path == "/api/test_capture":
            path, vol, err = test_capture()
            if path:
                self._send_json({"ok": True, "path": path, "level": vol})
            else:
                self._send_json({"ok": False, "error": err, "level": 0.0})
        elif self.path == "/api/test_ollama":
            ok, elapsed, msg = test_ollama()
            self._send_json({"ok": ok, "elapsed": elapsed, "msg": msg})
        elif self.path == "/api/wmodels":
            self._send_json(list_whisper_models())
        elif self.path == "/api/events":
            self._sse()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/control":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body or b"{}")
            except Exception:
                data = {}
            self._apply_control(data)
        elif self.path == "/api/open":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body or b"{}")
                p = data.get("path", "")
                if p and os.path.exists(p):
                    subprocess.Popen(["open", p])
                    self._send_json({"ok": True})
                else:
                    self._send_json({"ok": False, "error": "path not found"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
        elif self.path == "/api/ollama_pull":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body or b"{}")
                name = data.get("name", "").strip()
                if not name or not all(c.isalnum() or c in ":._-" for c in name):
                    self._send_json({"ok": False, "error": "invalid model name"})
                else:
                    broadcast("info", {"msg": f"Running: ollama pull {name}"})
                    r = subprocess.run(["ollama", "pull", name],
                                       capture_output=True, text=True, timeout=1800)
                    if r.returncode == 0:
                        broadcast("ok", {"msg": f"✓ Pulled {name}"})
                        self._send_json({"ok": True})
                    else:
                        broadcast("error", {"msg": f"ollama pull failed: {r.stderr.strip()[:200]}"})
                        self._send_json({"ok": False, "error": r.stderr.strip()[:200]})
            except subprocess.TimeoutExpired:
                self._send_json({"ok": False, "error": "pull timed out (30 min)"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
        elif self.path == "/api/download_wmodel":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body or b"{}")
                name = data.get("name", "large-v3-turbo")
            except Exception:
                name = "large-v3-turbo"
            # respond immediately, download in background
            self._send_json({"ok": True, "msg": "Download started"})
            def _do_dl():
                broadcast("info", {"msg": f"Downloading ggml-{name}.bin…"})
                path, err = download_whisper_model(name)
                if path:
                    with _state_lock:
                        state["wmodel"] = path
                    broadcast("ok", {"msg": f"✓ Downloaded {os.path.basename(path)} — now active"})
                    broadcast("wmodels", {"models": list_whisper_models(), "current": path})
                else:
                    broadcast("error", {"msg": f"Download failed: {err}"})
            threading.Thread(target=_do_dl, daemon=True).start()
        elif self.path == "/api/cleanup_incomplete":
            n, freed = cleanup_incomplete_downloads()
            self._send_json({"ok": True, "removed": n, "freed_mb": freed})
        else:
            self.send_error(404)

    def _apply_control(self, data):
        with _state_lock:
            if "listening" in data:
                state["listening"] = bool(data["listening"])
            if "device" in data:
                state["device"] = data["device"]
            if "src_lang" in data:
                state["src_lang"] = data["src_lang"]
                _whisper_prompt_cache["src"] = None  # invalidate
            if "tgt_lang" in data:
                state["tgt_lang"] = data["tgt_lang"]
                _translation_prompt_cache["key"] = None
            if "model" in data:
                state["model"] = data["model"]
            if "wmodel" in data:
                p = data["wmodel"]
                if os.path.exists(p):
                    state["wmodel"] = p
            # denoise, vad_chunk, use_memory are always-on (no toggle)
            if "use_mlx_whisper" in data:
                state["use_mlx_whisper"] = bool(data["use_mlx_whisper"])
            if "use_mlx_lm" in data:
                state["use_mlx_lm"] = bool(data["use_mlx_lm"])
                if not state["use_mlx_lm"]:
                    # Free the cached MLX-LM model to reclaim RAM immediately
                    _mlx_lm_cache["loaded"] = False
                    _mlx_lm_cache["model_obj"] = None
                    _mlx_lm_cache["tokenizer"] = None
                    import gc
                    gc.collect()
            if "concise_translation" in data:
                state["concise_translation"] = bool(data["concise_translation"])
                _translation_prompt_cache["key"] = None
            if "glossary" in data:
                state["glossary"] = str(data["glossary"])[:2000]
                _translation_prompt_cache["key"] = None
                _whisper_prompt_cache["src"] = None
            if "custom_prompt" in data:
                state["custom_prompt"] = str(data["custom_prompt"])[:4000]
                _translation_prompt_cache["key"] = None
                _whisper_prompt_cache["src"] = None
            if data.get("clear"):
                state["history"] = []
        broadcast("state", _public_state())
        self._send_json({"ok": True, "state": _public_state()})

    def _send_json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.write(
                f"event: state\ndata: {json.dumps(_public_state(), ensure_ascii=False)}\n\n".encode()
            )
            self.wfile.flush()
        except Exception:
            return
        client_q: queue.Queue = queue.Queue(maxsize=300)
        with clients_lock:
            sse_clients.add(client_q)
        try:
            while True:
                try:
                    msg = client_q.get(timeout=15)
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with clients_lock:
                sse_clients.discard(client_q)


def main():
    # default model — prefer accuracy over speed for first run
    # Priority order: gemma > llama3 > mistral > qwen > anything else
    models = fetch_models()
    if models:
        PRIORITY = (
            "gemma4", "gemma3", "gemma2", "gemma",
            "llama3.1", "llama3.2", "llama3",
            "mistral", "mixtral",
            "aya", "qwen2.5", "qwen3",
        )
        preferred = None
        for prefix in PRIORITY:
            hit = next((m for m in models if m.startswith(prefix)), None)
            if hit:
                preferred = hit
                break
        if not preferred:
            preferred = models[0]
        with _state_lock:
            state["model"] = preferred

    # P1 — prefer large-v3-turbo if installed, else whatever's already set
    installed_wmodels = list_whisper_models()
    if installed_wmodels:
        turbo = next((p for p in installed_wmodels if "large-v3-turbo" in p), None)
        large = next((p for p in installed_wmodels if "large-v3" in p and "turbo" not in p), None)
        small = next((p for p in installed_wmodels if "small" in p), None)
        chosen = turbo or large or small or installed_wmodels[0]
        with _state_lock:
            state["wmodel"] = chosen
        broadcast("info", {"msg": f"🎤 Whisper default → {os.path.basename(chosen)}"})

    # announce backend availability
    backend_info = []
    backend_info.append("VAD: webrtcvad")
    backend_info.append("denoise: on")
    backend_info.append("memory: on")
    backend_info.append("MLX-Whisper: ✓" if HAS_MLX_WHISPER else "MLX-Whisper: ✗")
    backend_info.append("MLX-LM: ✓" if HAS_MLX_LM else "MLX-LM: ✗")
    broadcast("info", {"msg": "Backends — " + " · ".join(backend_info)})

    # start threads — P2 pipeline: audio → stt → llm (3 stages)
    threading.Thread(target=audio_thread, daemon=True, name="audio").start()
    threading.Thread(target=stt_thread,   daemon=True, name="stt").start()
    threading.Thread(target=llm_thread,   daemon=True, name="llm").start()

    # warmup: pre-load model by sending a test prompt
    def warmup():
        time.sleep(0.6)
        broadcast("info", {"msg": "Pre-loading Ollama model…"})
        ok, elapsed, msg = test_ollama()
        broadcast("ok" if ok else "error", {"msg": msg})
    threading.Thread(target=warmup, daemon=True, name="warmup").start()

    # try to bind to fixed port; give up with clear message if taken
    try:
        server = _FastHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"\n❌ Cannot start on port {PORT}: {e}")
        print(f"   Port {PORT} is likely already in use.")
        print(f"   → kill the other process:  lsof -ti:{PORT} | xargs kill -9")
        print(f"   → or change PORT in the script (top of file).")
        sys.exit(1)
    port = PORT
    url = f"http://localhost:{port}/"
    print(f"\n⚡ Foundry — Realtime Voice Intelligence · The Factory Group")
    print(f"   running at {url}")
    print("   BlackHole 2ch → VAD + denoise → whisper (large-v3-turbo) → Ollama/MLX")
    print("   Press Ctrl+C to stop.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
        server.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
