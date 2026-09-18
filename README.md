# Realtime Japanese → English Translator

Listens to **Windows system audio** (browser / stream / game) and continuously
translates **Japanese speech → English text** into the CMD/terminal window.

```
System audio (WASAPI loopback)
        ↓
5s / 1s rolling buffer  (configurable)
        ↓
kotoba-tech/kotoba-whisper-bilingual-v1.0 (faster-whisper / CTranslate2, CUDA)
        ↓
Japanese → English  (task=translate, language=en)
        ↓
deduplicated English text → CMD
```

## Requirements

- Windows 10/11
- Python 3.10+ (a working install, no virtualenv required)
- NVIDIA GPU + CUDA (the app uses CUDA exclusively and will **not** fall back to CPU)
- PyTorch + CUDA already installed (used only for the CUDA check; never modified)

## Install

The app does **not** install, upgrade, or modify PyTorch/CUDA. It only adds the
packages it needs:

```powershell
python -m pip install faster-whisper PyAudioWPatch numpy scipy
```

`scipy` and `numpy` are usually already present; `pip` skips anything installed.

## Run

```powershell
python main.py                       # client UI (thin); "Local" auto-starts a server
python main.py --client              # client UI (thin, no Whisper/Torch/CUDA)
python main.py --server-mode         # server UI (model + audio input + LAN broadcast)
python main.py --server-headless     # headless caption server (for local mode / remote)
python main.py --cli                 # terminal ASR pipeline (original, no GUI)
```

### Architecture: client captures audio, server runs the model

The **client** captures Windows system audio (WASAPI loopback) and streams it to
the **server**, which runs Whisper and streams translated captions back. The client
UI never loads or runs Whisper/Torch/CUDA. In **Local** mode the client spawns a
headless server subprocess (`--server-headless`) and connects to it over localhost;
**LAN** mode connects to a remote server; **Remote/Colab** is reserved.

- **Client** (`main.py` / `--client`): **Audio Input** dropdown (loopback devices)
  + connection/caption settings; RUN → borderless overlay showing the returned captions.
- **Server** (`--server-mode` / `--server-headless`): receives streamed audio, runs
  the model, returns captions; `--server-mode` shows server status.

Wire protocol: handshake + audio frames (client→server, 16 kHz mono float32) and
caption lines (server→client) over one full-duplex TCP connection.

The background fixer is **off by default** (no-fixer mode) to minimize overhead;
enable it with `--fixer` where applicable.

### GUI

`python main.py` opens the settings window. Configure caption style and
connection (Local / LAN / Remote), then press **RUN** to hide the settings
window and show the borderless caption overlay:

- left-drag to move it, right-click for the menu (`Show Main Window`,
  `Settings`, `Clear Caption`, `Pause Caption`, `Close Transcript`), `ESC` to close.
- captions replace (do not append), wrap to the configured width, and the top
  edge stays stable as the caption grows.
- settings persist to `settings.json`.

The GUI consumes the *stabilized* transcript (it never drives Whisper directly),
and the ASR pipeline keeps running on background threads so the UI stays
responsive. The client UI does not load Whisper/Torch/CUDA at all.

### CLI options

```powershell
python main.py --cli --buffer 5 --step 1        # minimum delay (default)
python main.py --cli --buffer 7 --step 1.5      # better quality, slightly more delay
python main.py --cli --compute-type float16     # higher precision (slower)
python main.py --cli --debug                    # print per-window confidence metrics
```

| Flag                     | Default        | Meaning                                            |
| ------------------------ | -------------- | -------------------------------------------------- |
| `--buffer`               | `5`            | Rolling audio window (seconds) fed to the model    |
| `--step`                 | `1`            | How often inference runs (seconds)                 |
| `--compute-type`         | `int8_float16` | `float16`, `int8_float16`, or `int8`               |
| `--vad` / `--no-vad`     | on             | Silero VAD to skip silence / drained-buffer gaps   |
| `--min-logprob`          | `-1.0`         | Drop segments below this average log-probability   |
| `--max-compression-ratio`| `2.4`          | Drop repetitive segments above this ratio          |
| `--max-temperature`      | `1.0`          | Drop segments decoded at this temperature fallback |
| `--min-words`            | `1`            | Drop segments shorter than this many words         |
| `--silence-threshold`    | `0.001`        | Skip inference when window RMS is below this       |
| `--stability`            | `1`            | Windows a sentence must repeat before emission     |
| `--overlap`              | `0.5`          | Word overlap (0-1) treated as the same sentence    |
| `--prompt`               | off            | Feed previous text back as a decoder prompt (causes repetition; experimental) |

Lower `--buffer`/`--step` = lower latency but more fragmented/less accurate
translations. Higher = more context and cleaner output, but more delay.

## Reducing junk / hallucinated text

Short rolling windows can produce fragmented or hallucinated output (e.g. filler
like "now, but ..."). The app filters this using **signals the model already
computes during decoding** — no extra model or latency:

1. **Silero VAD** (`--vad`, on by default) — strips silence and near-silence, so a
   drained buffer or a gap between sentences produces nothing instead of junk.
2. **Confidence filtering** — segments are dropped when any of these trip:
   `avg_logprob < --min-logprob`, `compression_ratio > --max-compression-ratio`,
   or `temperature >= --max-temperature`. A segment decoded at temperature `1.0`
   means the model exhausted all fallback temperatures and is almost always a
   hallucination.
3. **RMS gate** (`--silence-threshold`) — skips inference entirely when the whole
   window is quiet, saving GPU work.
4. **Fuzzy dedup** (`--stability`, `--overlap`) — emits each distinct sentence once
   and suppresses re-wordings of the same sentence across overlapping windows.

Tune on your live loop with `python main.py --debug`, which prints the RMS, VAD
duration, and per-segment `logp / comp / nsp / temp` alongside the text, so you
can see exactly why a line was kept or dropped. `--stability 2` gives cleaner but
sparser output; `--stability 1` (default) is more responsive.

### Why not a separate "error detection" model?

A third-party text-quality classifier (a small BERT, a perplexity scorer, etc.) was
evaluated and rejected: it would add a model download/load, a second forward pass
per inference (latency), and extra VRAM/CPU — while the log-probability,
compression-ratio, no-speech-probability, and temperature signals are **already
computed for free** by faster-whisper during decoding and correlate strongly with
hallucination.

## How it works

- **Audio capture** (`audio_capture.py`) — a background thread records the default
  output device's WASAPI **loopback** via `PyAudioWPatch`, downmixes stereo→mono,
  resamples 48 kHz→16 kHz float32, and keeps a rolling in-RAM buffer. No WAV files.
- **Translation** (`translator.py`) — loads the CTranslate2 model **once** at startup
  on CUDA (`int8_float16`), then translates each window with
  `language="en", task="translate", beam_size=1`.
- **Dedup** (`text_dedup.py`) — removes the overlap between consecutive windows via
  word-suffix matching and only prints complete sentences.
- **Inference loop** (`main.py`) — a worker thread runs inference every `--step`
  seconds on the latest `--buffer` seconds; audio capture keeps running in parallel
  during GPU inference.

### Model

Uses the faster-whisper/CTranslate2 weights:

```
kotoba-tech/kotoba-whisper-bilingual-v1.0-faster
```

(the FP16 CTranslate2 conversion of `kotoba-tech/kotoba-whisper-bilingual-v1.0`;
the parent repo only contains Transformers safetensors).

The model downloads once (~1.5 GB) into the Hugging Face cache on first run.

## Troubleshooting

- **GUI (local/server) crashes with no error output** — this is a PyQt5/CUDA DLL
  init conflict: CTranslate2 must be loaded *before* Qt. The app already does this
  (`_run_with_model` loads the model before importing `ui`). If you import the UI
  yourself, call `asr_engine.load_model()` before importing `PyQt5`.
- **`Library cublas64_12.dll is not found or cannot be loaded`** — the app already
  points CTranslate2 at the CUDA DLLs bundled with PyTorch (`torch/lib`). If it still
  fails, make sure the PyTorch install actually has `torch\lib\cublas64_12.dll`.
  pip install --no-deps nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
- **No output / silence** — confirm audio is actually playing to the selected output
  device. The startup banner prints the loopback device name; use that device for
  playback.
- **`Invalid sample rate`** — the WASAPI loopback device only accepts its native rate
  (typically 48000); the app handles this internally by resampling to 16 kHz.
- **No default output device found** — the app prints the available loopback devices
  and asks you to pick one by index.
