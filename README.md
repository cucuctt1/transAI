# Realtime Japanese → English Translator

Listens to **Windows system audio** (browser / stream / game) and continuously
translates **Japanese speech → English text** into the CMD/terminal window.

```
System audio (WASAPI loopback)
        ↓
10s / 5s rolling buffer  (configurable)
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
python main.py
```

Press `Ctrl+C` to stop cleanly (releases the audio device, model, and GPU).

### Options

```powershell
python main.py --buffer 5 --step 1              # minimum delay (default)
python main.py --buffer 7 --step 1.5            # better quality, slightly more delay
python main.py --compute-type float16           # higher precision (slower)
```

| Flag            | Default         | Meaning                                          |
| --------------- | --------------- | ------------------------------------------------ |
| `--buffer`      | `5`             | Rolling audio window (seconds) fed to the model  |
| `--step`        | `1`             | How often inference runs (seconds)               |
| `--compute-type`| `int8_float16`  | `float16`, `int8_float16`, or `int8`             |

Lower `--buffer`/`--step` = lower latency but more fragmented/less accurate
translations. Higher = more context and cleaner output, but more delay.

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

- **`Library cublas64_12.dll is not found or cannot be loaded`** — the app already
  points CTranslate2 at the CUDA DLLs bundled with PyTorch (`torch/lib`). If it still
  fails, make sure the PyTorch install actually has `torch\lib\cublas64_12.dll`.
- **No output / silence** — confirm audio is actually playing to the selected output
  device. The startup banner prints the loopback device name; use that device for
  playback.
- **`Invalid sample rate`** — the WASAPI loopback device only accepts its native rate
  (typically 48000); the app handles this internally by resampling to 16 kHz.
- **No default output device found** — the app prints the available loopback devices
  and asks you to pick one by index.
