# Foundry — Realtime Voice Intelligence

Real-time speech translation pipeline: BlackHole 2ch audio → Whisper STT → LLM translation → live web GUI.

## Requirements

- macOS (Apple Silicon recommended for MLX acceleration)
- [BlackHole 2ch](https://existential.audio/blackhole/) virtual audio device
- [Ollama](https://ollama.com/) with a chat model (e.g. `gemma2:9b`, `qwen2.5:1.5b`)
- Python 3.10+
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (`brew install whisper-cpp`)

## Quick Start

```bash
python3 realtime_translator.py
```

On first run it auto-creates a `.venv`, installs dependencies, and starts the web server at `http://127.0.0.1:8765`.

### Optional MLX Acceleration (Apple Silicon)

```bash
pip install mlx-whisper mlx-lm
```

MLX backends are auto-detected at runtime (STT → MLX-Whisper, Translate → MLX-LM).

## Pipeline

```
System Audio → BlackHole 2ch
  → noisereduce (denoise)
  → webrtcvad (drop silence)
  → 5s chunks + overlap
  → Whisper STT (MLX or whisper.cpp)
  → Context + Translation Memory
  → LLM Translate (MLX-LM or Ollama)
  → SSE → Live Web UI
```

## Features

- VAD chunking with configurable aggressiveness
- Noise reduction per chunk
- Language-pair translation prompts (DE→TH / EN→TH / JA→TH / generic)
- Glossary & custom prompt support (editable from GUI)
- Translation memory (last 5 segments as context)
- Output validation + 1 retry on failure
- Whisper model download manager (built-in GUI)
- Responsive dark-theme web UI

## Configuration

Edit constants at the top of `realtime_translator.py`:
- `SAMPLERATE`, `CHUNK_DURATION`, `VAD_AGGR`
- `WHISPER_THREADS`, `PORT`, `DEFAULT_DEVICE`
- Language pairs, hotwords, prompts

---

Built by **The Factory Group**
