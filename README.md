# Realtime Japanese to English Translator

Captures Windows system audio and displays live English captions using
`kotoba-tech/kotoba-whisper-bilingual-v1.0-faster`. Audio stays on your local
machine in Local mode; LAN mode sends audio to your selected caption server.

## Guide contents

- [Windows installation](#windows-installation)
- [Example commands by use case](#example-commands-by-use-case)
- [Complete command-line reference](#complete-command-line-reference)
- [Optional Moonshine backend](#optional-moonshine-backend-english-captions)
- [Commit-only finalization](#commit-only-finalization)
- [Japanese correction before OPUS-MT](#japanese-correction-before-opus-mt-moonshine)
- [GUI and connections](#gui-and-connections)
- [Troubleshooting](#troubleshooting)

## Windows installation

### 1. Choose what this computer will do

| Installation | Required packages | Models on this computer? |
| --- | --- | --- |
| LAN display/audio client only | `requirements-client.txt` | No |
| Local UI with automatic server, or Kotoba server/CLI | `requirements.txt` | Kotoba |
| Moonshine server/local UI/CLI | `requirements.txt` plus `requirements-moonshine.txt`, existing Torch 2.6+ | Moonshine and OPUS-MT |
| Any pipeline with Qwen | Above plus optional fixer runtime | Qwen GGUF |

Use **64-bit Windows and Python 3.10 or 3.11** for the paths tested here. The app
captures Windows playback/loopback audio, not an audio filename or microphone CLI
input. No Docker or WSL is required. First model downloads need internet and
free disk space; allow several GB for dependencies, model files and caches.
GPU mode needs an NVIDIA GPU/driver and the runtime appropriate to each backend.
CPU mode is supported, but slower inference may lose audio if it exceeds history.

All commands below are **PowerShell**, run from the repository:

```powershell
Set-Location D:\transAI
python --version
python -c "import sys; print(sys.executable)"
```

Substitute your repository path if different. Use `python -m pip`, not a bare
`pip`, so installation targets the same interpreter that runs the app. If Python
is not on PATH, use the full path to your existing Python executable.

### 2. Reuse your existing Torch (do not reinstall it)

If Torch is already working, verify it before changing any environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.__file__); print('CUDA available:', torch.cuda.is_available())"
```

This workspace was checked with `2.7.1+cu128`. None of the project's requirements
files asks pip to install Torch. Moonshine needs Torch 2.6+; this app also uses
Torch to check the GPU when launching either speech backend with `--device cuda`.
If Torch is missing, stop before following the Moonshine/GPU steps. Configure it
deliberately for that Python using the [official PyTorch instructions](https://pytorch.org/get-started/locally/),
or use Kotoba's CPU path. Do not replace a working CUDA Torch installation just
because Qwen needs a different runtime: Qwen uses llama.cpp, not Torch.

Create an environment that inherits the existing Torch while isolating new packages:

```powershell
python -m venv --system-site-packages .runtime/moonshine-env
.runtime/moonshine-env/Scripts/python -m pip install --cache-dir .cache/pip -r requirements.txt
.runtime/moonshine-env/Scripts/python -m pip install --cache-dir .cache/pip -r requirements-moonshine.txt
.runtime/moonshine-env/Scripts/python -c "import torch, transformers; print(torch.__version__, torch.__file__); print(transformers.__version__)"
```

Skip creation/installation if this environment is already prepared. Inheritance
also exposes other base-environment packages, so it is not a completely clean
environment; unrelated dependency warnings may need investigation. Use the same
base Python that owns your working Torch. Activation is optional: invoking the
environment's Python directly avoids PowerShell activation-policy changes.

### 3. Kotoba/faster-whisper only

For an existing interpreter/environment:

```powershell
python -m pip install --cache-dir .cache/pip -r requirements.txt
python main.py --cli --device cpu
```

The CPU command is an initial functional check and starts playback capture; play
Japanese audio, then press Ctrl+C to stop. For CUDA, first check Torch as above.
Current faster-whisper builds require CUDA 12 cuBLAS and cuDNN 9; a working Torch
GPU check alone does not prove that CTranslate2 can load those DLLs. Follow the
[faster-whisper Windows GPU setup](https://github.com/SYSTRAN/faster-whisper#gpu)
for matching runtime libraries. The app searches repo-local `cuda12/` and `dlls/`,
Torch libraries, system CUDA 12 directories, NVIDIA package directories and PATH.
Do not copy DLLs from arbitrary downloads or mix incompatible CUDA major versions.

```powershell
python main.py --cli --device cuda --debug
```

Wait for model loading before judging first-caption latency. A missing DLL,
unsupported compute type or native initialization problem must be fixed first;
changing caption timing will not repair it.

### 4. Optional Qwen CPU fixer

Install the optional runtime locally (uses the official CPU wheel index):

```powershell
python -m pip install --target .runtime --cache-dir .cache/pip --only-binary=:all: -r requirements-fixer.txt
python llm_fixer.py
```

The second command downloads the pinned Qwen GGUF, not a new Torch. Use the
prepared environment's Python instead of `python` if that is your chosen
interpreter. The app inserts `.runtime` into its import path for the fixer.
CPU Qwen uses two inference threads; it is separate from the `--device` setting.
`--fixer` alone does **not** use Qwen; use `--llm-fixer` to enable the model.

### 5. Optional Qwen CUDA fixer

First install the CPU fixer dependencies above. Install a compatible CUDA
llama-cpp-python wheel separately under `.runtime/fixer-cuda`; do not overwrite
Torch or the CPU wheel. See [Fixer device](#fixer-device) for the installation
command and compatibility caveats. `--fixer-cuda` enables Qwen and requests full
offloading. If the runtime is CPU-only, the app logs the error and uses unedited
captions; the flag does not download/install a CUDA runtime automatically.

With 4 GB VRAM, ASR/MT and Qwen can compete for memory. Start with CPU Qwen and
measure before enabling CUDA. Available memory, not just model file size, matters.

### 6. Lightweight LAN client only

On the display/playback computer (a different computer from the model server):

```powershell
python -m venv .runtime/client-env
.runtime/client-env/Scripts/python -m pip install --cache-dir .cache/pip -r requirements-client.txt
.runtime/client-env/Scripts/python main.py --client
```

Choose **LAN**, enter the server's address and pairing code, select the playback
loopback device, then press RUN. Do not choose Local with this client-only install:
Local launches a model server and therefore needs server dependencies.

### 7. Model storage and offline use

| Repository path | Contents |
| --- | --- |
| `models/whisper/` | Kotoba/faster-whisper weights |
| `models/moonshine/` | Pinned Moonshine and OPUS-MT snapshots |
| `models/fixer/` | Qwen Q4 GGUF |
| `models/huggingface/` | General Hugging Face/Xet caches |
| `.runtime/` | Optional fixer packages and project Python environments |
| `.runtime/fixer-cuda/` | Optional CUDA llama.cpp wheel |
| `.cache/pip/` | Pip downloads when using the guide's cache option |

These paths are Git-ignored. Existing C: model caches are not moved or deleted.
Temporary OS/installer files can still use Windows temp directories; this is not
a guarantee that no bytes are ever written on C:. Keep the repository on the
drive where you want the persistent models. Downloads are automatic on first use.
To request offline operation **after all required files have been downloaded**:

```powershell
$env:HF_HUB_OFFLINE = '1'
.runtime/moonshine-env/Scripts/python main.py --backend moonshine
Remove-Item Env:HF_HUB_OFFLINE
```

Missing cached files will fail offline; remove the variable and finish downloading.

## Example commands by use case

Commands in this section use the already-prepared `.runtime/moonshine-env`
interpreter for both speech backends. If using another complete environment,
replace that executable with its Python path. Run one example at a time.

### A. Simplest local UI, no fixer, live captions

```powershell
.runtime/moonshine-env/Scripts/python main.py
.runtime/moonshine-env/Scripts/python main.py --backend moonshine
```

Pick Local, choose the loopback device matching your playback, then RUN. The UI
starts its own headless server and forwards model/pipeline arguments to it.
Default: CUDA, live mode, two-second updates, five-second inactivity clear.
Closing the UI shuts down its own child server, not unrelated servers.

### B. CPU-only translation

```powershell
.runtime/moonshine-env/Scripts/python main.py --device cpu
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --device cpu
```

Omit `--fixer-cuda` to keep the entire inference pipeline on CPU. Kotoba converts
FP16 compute settings to INT8 on CPU. Moonshine/OPUS-MT use FP32.

### C. Deterministic English cleanup, without an LLM

```powershell
.runtime/moonshine-env/Scripts/python main.py --fixer
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --fixer
```

This cleans spacing and obvious repeated loops. It is not semantic correction.
Commit-only mode bypasses this deterministic English-cleanup stage.

### D. Live captions with English Qwen editing

```powershell
.runtime/moonshine-env/Scripts/python main.py --llm-fixer --caption-timeout 8
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --llm-fixer --fixer-language en --caption-timeout 8
```

English editing happens after translation. Raw previews are hidden while editing;
eight seconds gives the CPU fixer more time before silence expiry drops its result.

### E. Japanese Qwen correction before English translation

```powershell
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --fixer-language ja --llm-fixer --caption-timeout 8
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --fixer-language ja
```

The first uses Japanese Qwen, then OPUS-MT. The second performs deterministic
Japanese preparation only (no LLM). Neither runs Qwen again on the English output.
This option is not supported by Kotoba.

### F. Full phrases versus one-time commits

```powershell
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --phrase-only --phrase-pause 1
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --commit-pause 0.45
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --fixer-language ja --llm-fixer --caption-timeout 8
.runtime/moonshine-env/Scripts/python main.py --commit-only --llm-fixer --caption-timeout 8
```

Phrase-only uses pauses plus punctuation/long-pause fallback. Commit-only uses
shorter pauses and one-time phrase IDs. The last command is Kotoba with English
finalization. Do not combine `--live`, `--phrase-only`, and `--commit-only`.

### G. Qwen CUDA, after installing its CUDA runtime

```powershell
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --fixer-language ja --fixer-cuda --caption-timeout 8
```

`--llm-fixer` is implied. `--device cpu --fixer-cuda` would deliberately run speech
inference on CPU but Qwen on GPU; those controls are independent.

### H. Terminal output and debug logging

```powershell
.runtime/moonshine-env/Scripts/python main.py --cli --backend moonshine --debug
.runtime/moonshine-env/Scripts/python main.py --cli --backend moonshine --commit-only --fixer-language ja --llm-fixer --caption-timeout 8 --debug
```

CLI captures this computer's default loopback audio directly, without a server or
Qt overlay. Live captions replace the terminal line; phrase/commit modes print
separate lines. `--debug` includes text, so avoid saving logs containing private
speech unless intended. Ctrl+C requests graceful shutdown.

### I. Headless LAN server and another computer's client

On the model computer:

```powershell
.runtime/moonshine-env/Scripts/python main.py --server-headless --backend moonshine --commit-only --fixer-language ja --llm-fixer --caption-timeout 8 --port 8765 --pairing-code "choose-your-code"
```

On the playback/display computer, run the client from installation step 6. In the
UI choose LAN and set, for example, `192.168.1.50:8765` (replace with the model
computer's actual private IP), with the same pairing code. Allow the chosen TCP
port through Windows Firewall on the private network if necessary. The server
receives the client's audio; starting a headless server alone does not capture
the server computer's playback. Set model/fixer options on the server, not the LAN
client. The server binds all interfaces; the protocol is plain TCP, not TLS.
Use a trusted LAN. A pairing code does not encrypt audio; do not expose it by
public port forwarding. The server processes the active client's audio, not
independent simultaneous model sessions per client.

### J. Server with a settings/status window

```powershell
.runtime/moonshine-env/Scripts/python main.py --server-mode --backend moonshine --commit-only --fixer-language ja --llm-fixer --caption-timeout 8
```

Press RUN in the server UI and connect a client. **Current limitation:** server UI
uses port 8765 and the pairing code from its settings; CLI `--port` and
`--pairing-code` are not passed into that UI path. Use the headless server for a
custom port. For Local auto-launch, leave port/code at defaults; a busy 8765 causes
the launcher to choose a private free port. Avoid mixing launch flags: although
the parser accepts some combinations, only one launch branch actually runs.

### K. Example tuning profiles

```powershell
# More context/history, without enabling fixers
.runtime/moonshine-env/Scripts/python main.py --buffer 10 --audio-buffer 20 --step 1
# Earlier commit at short pauses; easier to split a thought too soon
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --commit-pause 0.3 --debug
# Fewer caption changes, longer time to read before inactivity clear
.runtime/moonshine-env/Scripts/python main.py --step 1.5 --caption-timeout 5
# Kotoba terminology hints (ignored by Moonshine)
.runtime/moonshine-env/Scripts/python main.py --translation-prompt "Akihabara, Kaito, OpenAI"
# Custom compatible ChatML GGUF fixer, path quoted for spaces
.runtime/moonshine-env/Scripts/python main.py --llm-fixer --fixer-model "D:\transAI\models\fixer\custom.gguf"
```

These are examples, not measured optimal profiles. More context, wider beams,
longer pauses and LLM stages can improve some outputs while adding delay.

## Run

Default server launches use the slower, context-oriented live profile: CUDA `int8_float16`,
10 seconds of context, 20 seconds of audio history, a 2-second minimum update
interval, beam 1, and VAD enabled. Both fixers and phrase-only mode are off.
This applies to the automatically launched local server, `--server-mode`, and
`--server-headless`. CPU execution converts the compute type to `int8`.
The profile reduces redundant decoding while retaining sentence context; it is
not a measured optimum for every recording or GPU.

```powershell
python main.py                         # client UI; Local starts a caption server
python main.py --cli                   # terminal captions, same translation pipeline
python main.py --server-headless       # caption server without GUI
python main.py --server-mode           # caption server with status UI
python main.py --client                # thin client; no model loaded in this process
```

Install server dependencies with `pip install -r requirements.txt`, or client-only
dependencies with `pip install -r requirements-client.txt`. Use Python 3.10+.
CUDA is the default; `--device cpu` is supported, including in CLI mode.
The model downloads into the Hugging Face cache on first use.

## Optional Moonshine backend (English captions)

```powershell
python -m pip install -r requirements-moonshine.txt
python main.py --backend moonshine
python main.py --server-headless --backend moonshine
python main.py --cli --backend moonshine --device cpu
```

`--backend faster-whisper` remains the default. Local mode forwards the option
to its child server; for LAN, select it on the server command. Moonshine requires
Transformers 5.2 and PyTorch 2.6+; retain a CUDA-enabled PyTorch installation for
`--device cuda`. No Docker is needed. Use a separate virtual environment if you
do not want to upgrade Transformers in your existing environment. The optional
requirements deliberately do not install or upgrade Torch. For example, reuse
your installed Torch while isolating the new dependencies:

```powershell
python -m venv --system-site-packages .runtime/moonshine-env
.runtime/moonshine-env/Scripts/python -m pip install -r requirements-moonshine.txt
.runtime/moonshine-env/Scripts/python main.py --backend moonshine
```

This pairs [Moonshine Streaming Tiny Japanese](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny-ja)
with [OPUS-MT Japanese-to-English](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en).
Moonshine alone does **not** translate. Both revisions are pinned and cached in
`models/moonshine/` inside this repository. First launch needs internet. Both
stages use FP32 on the selected CPU/CUDA device.
Local auto-launch allows up to ten minutes for this two-model cold start, while
still detecting process exit and cancellation; it does not wait once ready.

The integration reuses rolling-window scheduling, caption dedup, optional fixers,
phrase mode, and safe shutdown. It uses window-based `generate`, not persistent
incremental encoder state. ASR is greedy with a duration-based output token cap;
MT uses beam 1 in live mode and the scheduled beam in phrase mode. Identical
consecutive Japanese hypotheses reuse their English translation. Whisper's
`--compute-type`, `--vad`, decoder quality thresholds, `--min-words`, and
`--translation-prompt` do not apply; the scheduler energy gate still applies.

This is an experimental alternative, not a promised speed/quality improvement:
ASR errors propagate into translation, particularly on noisy/short speech.
Compare recordings with `--debug`, measuring final English-caption latency.
Verification on this machine: CUDA model loading, non-speech ASR generation,
Japanese-to-English text generation, offline cached loading and normal process
exit succeeded with the existing Torch 2.7.1+cu128. These are smoke checks, not
real-speech quality or comparative speed benchmarks. The inherited environment
prints an optional OpenCV/NumPy compatibility warning during Transformers'
tokenizer checks; it did not prevent inference. Torch/NumPy/OpenCV were not changed.

## Scheduling and quality

- Incoming audio is downmixed and resampled once, then stored in a sample-bounded
  16 kHz ring. Local and network sources share the same buffer implementation.
  Snapshots copy only the requested samples, independent of network frame sizes.
- One worker owns inference and pulls fresh audio when ready. There is no queue
  of expensive, stale windows. Inference time counts toward the preview interval;
  slow inference is not followed by another artificial wait.
- A cheap 20 ms energy gate checks only new samples. With default settings,
  previews can start after approximately two seconds of speech, before the full
  ten-second window is available. Energy detection is not speech recognition:
  music or background noise can still trigger it; Silero VAD in the decoder
  provides a second filter.
- Live mode uses beam 1 for both previews and the final pause update. It never
  switches to phrase-mode beam search or waits for sentence punctuation. Use
  `--live` to select it explicitly. Startup logs report the active mode.
- Once that final pass completes, continuing silence triggers no more inference.
  The next utterance excludes the previous utterance's audio where a pause was
  observed. Shorter thinking pauses retain context. Continuous speech uses an
  ten-second rolling window; slow decodes can recover unread audio plus 1.5
  seconds of overlap from the 20-second history. If history runs out, a warning
  reports the unavailable duration. No bounded buffer can recover overwritten audio.
- The fixer is currently off in the CLI defaults; `--fixer` enables deterministic cleanup
  on results, with no extra model or polling thread. It normalizes spacing and
  collapses obvious loops of three or more consecutive phrases (at least three
  words per phrase). Two repeats, short emphasis, negation and numbers are kept.
  A deliberate triple repeated phrase can also be collapsed; use `--no-fixer`
  to disable this cleanup. Meaning corrections come from the audio decoder, not
  guessed grammar edits.
- Captions replace the current window instead of appending loosely matched
  words. Deduplication preserves word order, numbers and negation. Returning to
  a recently displayed wording needs a second observation, or the final decode.
  Identical words in a new utterance are allowed.
- Display text retains punctuation and is limited to 32 words. Captions clear
  after five seconds without detected audio activity (including stopped input).
  Use `--caption-timeout` to adjust this. Clearing remains responsive during ASR
  inference and bypasses display pacing. Music/noise above the energy threshold
  can count as activity. Identical updates are not
  broadcast again. CLI captions overwrite the previous line in an interactive
  terminal; redirected output prints changed captions as separate lines.

The decoder remains deterministic, with `condition_on_previous_text=False`.
Prior translated captions are not fed back as a prompt. The earlier hard
three-word repetition ban was removed because it could block legitimate speech.

## Tuning

### Commit-only finalization

```powershell
python main.py --commit-only --llm-fixer --caption-timeout 8
# Prepared Moonshine environment (Japanese source strengthens boundary checks):
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --llm-fixer --caption-timeout 8
```

`--commit-only` is separate from and mutually exclusive with `--live` and
`--phrase-only`. It works in CLI, Local UI, and either server launch mode. It
does not enable the LLM automatically: without `--llm-fixer`, deterministic
boundary checks still provide one-time phrase publication.

The finalizer receives a phrase ID, audio timestamps/pause state, current English,
Moonshine's Japanese source (when available), and up to four **actually published**
phrases as read-only history. Its structured result is `action: wait|commit` and
`new_text`. The app owns the commit ledger and audio cut: repeated IDs, stale
IDs and overlapping audio intervals cannot be committed again. Separate phrases
with intentionally identical wording are preserved. This does not guarantee
that a recognition/translation model never hallucinates repeated wording.

Commit mode uses `--commit-pause 0.45`, independently of phrase-only mode's
`--phrase-pause`. It accepts plain Japanese endings and unpunctuated English at
audio pauses; punctuation/polite endings are not mandatory. Japanese topic/case
particles and conjunctive/te-form endings can delay an initial commit **once**.
Predicate/negation endings can arrive late; omitted subjects must not be invented.
These are deliberately limited suffix heuristics, not a grammatical parser.
See the [Japanese sentence-analysis reference](https://aclanthology.org/C80-1078/)
for predicate-final structure and omission. Kotoba has no Japanese transcript in
this pipeline, so it uses audio boundaries and the current English draft instead.

A continuation wait retains the audio and draft. Resumed speech extends the same
phrase, but the next qualifying pause commits it without another veto. Continuing
silence releases the cached draft after max(0.9 seconds, twice `--commit-pause`).
Once the app accepts a boundary, the LLM can clean the text but cannot delay it.
Trailing endpoint silence is trimmed to 120 ms for decoding, and commit decoding
uses at most beam 2. The finalizer has a bounded token budget; invalid edits or
added content words fall back to the raw translation. Numbers, negation and new
personal pronouns remain guarded. This cannot fix hallucinations already present
in the ASR/translation output.

Use `--debug` to inspect `[COMMIT]` reasons, Japanese source, raw English and edited
English. The 30-second limit remains only for speech with no detected pauses;
continuous background sound may defeat the energy gate. Lower `--commit-pause`
for shorter pauses or adjust `--silence-threshold` to the recording, rather than
assuming punctuation identifies a safe audio cut.

Commit mode serializes phrase finalization/publication instead of replacing
pending phrases. Slow ASR/LLM work can exhaust audio history; the engine warns
when that happens. Silence expiry still wins and drops unpublished late results;
use a longer `--caption-timeout` for the CPU fixer. Expired phrases never enter
committed history. More conservative completion adds latency; no real-speech
quality improvement is claimed by the automated regression tests.

### Optional local model fixer

#### Japanese correction before OPUS-MT (Moonshine)

```powershell
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --fixer-language ja --llm-fixer --caption-timeout 8
```

`--fixer-language ja` splits the Moonshine path into Japanese ASR, source
normalization/time-aware overlap checks, optional Japanese Qwen editing, then
OPUS-MT English translation. Use `--fixer-cuda` instead of `--llm-fixer` only if
the CUDA runtime is installed. Without either LLM flag, source stabilization still
runs, but there is no Qwen call. The default `--fixer-language en` keeps the old
English-fixer path. Japanese mode is rejected with faster-whisper/Kotoba.

In commit-only mode, Japanese boundary checks run before MT; only accepted phrases
are corrected and translated. Up to four published Japanese phrases are tracked;
the Japanese Qwen prompt receives only the latest one to stay small. Character
overlap is stripped only against **committed, temporally overlapping** audio,
never across separate phrases. Live drafts replace previous uncommitted drafts;
their repeated prefixes are not incorrectly removed. Live/phrase-only modes do
not label earlier drafts as committed source history. Identical consecutive source
drafts reuse correction and translation results. Japanese correction is serialized
before translation, so a slow CPU fixer increases caption delay.

Japanese validation preserves digit/kanji-number sequences and common negative
endings, bounds character changes, and rejects added content characters. These
are conservative heuristics, not proof of meaning preservation; uncertain edits
fall back to the original Japanese. English Qwen editing is **not** run a second
time. `--debug` prints `[JA-FIX]` source/corrected/English comparisons. Torch and
model weights are unchanged.

#### Overlay history animation

The previous displayed caption becomes a dim, ellipsized single line above the
current caption. It eases upward and scales down over 450 ms, holds briefly, and
fades out by four seconds. A reserved row prevents layout jumps during fading.
Duplicate captions do not restart the animation; manual clear and no-input clear
remove both lines immediately. This is display history, not the LLM's committed
history, and works for local and network captions alike.

#### Fixer device

Qwen uses CPU by default. Add `--fixer-cuda` to enable the fixer and request full
GPU layer offloading, independently of the speech model's `--device` setting:

```powershell
.runtime/moonshine-env/Scripts/python main.py --backend moonshine --commit-only --fixer-cuda --caption-timeout 8
```

This requires a CUDA build of `llama-cpp-python`; the existing
`requirements-fixer.txt` installs a **CPU** build. The flag does not install any
packages or change Torch. A CUDA wheel may be installed separately under
`.runtime/fixer-cuda/`, which takes priority only with the flag. Follow the
[official CUDA installation instructions](https://github.com/abetlen/llama-cpp-python#installation-configuration)
and choose a wheel matching your Python/driver. For example, when a matching
CUDA 12.5 Windows wheel is available:

```powershell
python -m pip install --target .runtime/fixer-cuda --cache-dir .cache/pip --no-deps --only-binary=:all: --index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 llama-cpp-python
```

Restart the server after changing the native runtime. CUDA mode enables native
loading logs so actual layer offloading/VRAM allocation can be inspected. A
non-CUDA runtime produces an explicit `[FIXER]` error and the existing unedited
caption fallback; it does not silently label CPU editing as CUDA. GPU allocation
failures likewise use the normal fixer-failure handling, not a second CPU model.
Qwen shares VRAM with ASR/translation; on a 4 GB GPU, full offloading may not fit.
Omit the flag to retain the CPU fixer. GPU execution has not been benchmarked
with the installed CPU-only runtime.

`python main.py --live --llm-fixer` enables asynchronous Qwen2.5-0.5B-Instruct
Q4_K_M caption editing. It also works with `--phrase-only`. This is separate from
`--fixer`, which enables only the original deterministic cleanup.

The model edits the current English caption using the preceding caption as
labelled context. It can repair repetitions, overlapping words, grammar and
small wording drift. It cannot recover Japanese meaning missing from Whisper's
translation because it does not hear the audio or receive a Japanese transcript.
It is a 0.5B model, not Qwen 0.8B. Its proposed edits are checked for numbers,
negation, excessive rewriting and invalid/truncated output; these checks reduce
risk but do not guarantee meaning preservation.

With `--llm-fixer`, raw previews stay hidden until editing finishes. A two-thread
CPU worker handles one active edit while the engine keeps only the newest pending
caption. New previews do not invalidate every running correction, so a slow fixer
can still publish during continuous speech. ASR keeps running independently.
Results are displayed at most once every two seconds by default (`--step` controls this).
Unchanged validated results also display. On model/edit failure, original text is
used as a fallback and a `[FIXER]` message explains the problem. Input expiry clears
pending display work and invalidates late corrections; a slow final edit may thus
be dropped after silence. Increase `--caption-timeout` for a slower CPU fixer.
Phrase mode uses the same bounded latest-pending policy, not an archival queue.

Install the optional runtime on D: (already installed in this workspace):

```powershell
python -m pip install --target .runtime --cache-dir .cache/pip --only-binary=:all: -r requirements-fixer.txt
python llm_fixer.py                      # download the pinned official Q4 model
python main.py --phrase-only --llm-fixer
```

The model is `models/fixer/qwen2.5-0.5b-instruct-q4_k_m.gguf`. Use
`--fixer-model D:\\path\\custom.gguf` to override it (a ChatML instruction model
is required). New Whisper downloads go to `models/whisper`; Hugging Face caches
are under `models/huggingface`. Existing C: caches are left untouched. Models,
runtime packages and pip caches are ignored by Git. No hosted inference is used.

One local smoke test removed a repeated sentence in about 2.9 seconds; this is
not a translation-quality benchmark. Sources: [official Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

### Full-phrase mode

Run `python main.py --phrase-only` for the overlay, or
`python main.py --cli --phrase-only` for one complete phrase per terminal line.
This mode waits for one second of quiet, then privately translates the captured
phrase. Normally output is published and its audio logically removed when the
text also ends with sentence punctuation (`.`, `?`, `!`, or Japanese equivalents).
If text is unfinished, its audio is retained and translated again after more
speech and another pause. Internal periods, decimals, common titles/initialisms,
and ellipses do not count as endings. Punctuation is a heuristic, not a grammar
parser. If silence lasts twice the configured pause (at least two seconds), the
saved translation is released even without punctuation, without a second model
call. An empty result clears the ended phrase without displaying invented text.
This prevents missing punctuation from holding unrelated speech until 30 seconds.
Audio arriving during inference is kept and processed in order. There are no displayed previews
and no 32-word caption truncation. Live mode remains the default.

`--phrase-pause 0.8` changes the silence needed before checking a phrase. Audio
boundaries use energy detection; music can delay the pause signal.
`--max-phrase-seconds 30` is a fallback: reaching this limit can publish a
nonempty translation without confirmed completion, with a log message. Phrase mode
allocates at least this limit plus 15 seconds of capture history. If inference
falls behind that history, a warning reports audio loss. Phrase probes use the
configured beam, capped at 2 when measured inference exceeds the pause duration.

The overlay shows the latest complete phrase; CLI phrase mode retains each phrase
on its own line. These options also apply to headless and server UI modes.

These options apply to CLI/headless/server UI mode. In the normal Local client
UI they are forwarded to its own server. If the preferred local port is already
occupied, a private port is used rather than inheriting an unrelated server's
mode/settings. Existing servers are not terminated. LAN clients cannot change
a remote server's options.

```powershell
python main.py --translation-prompt "Akihabara, Kaito, OpenAI"
python main.py --cli --buffer 7 --audio-buffer 9 --step 1
python main.py --cli --beam-size 1          # faster final decode, less search
python main.py --cli --debug               # audio times, beam, duration and text
python main.py --server-headless --device cpu
```

## Complete command-line reference

Run `python main.py --help` with your chosen interpreter. Times below are seconds.
Options are per launch, not automatically saved to `settings.json`. Local UI
forwards pipeline options to its child; a LAN client cannot reconfigure the server.

### Launch and connection

| Argument | Default | Purpose / applicability |
| --- | --- | --- |
| `-h`, `--help` | — | Print available arguments without loading models. |
| No launch flag | Local client UI | Model loads in its automatic headless child when Local RUN is pressed. |
| `--client` | off | Client UI; choose LAN for a model-free client. Local still launches a server. |
| `--server-headless` | off | Receive client audio and serve captions without a GUI. |
| `--server-mode` | off | Model server with GUI; press RUN after loading. Not local playback capture. |
| `--cli` | off | Capture default local loopback audio and print captions directly. |
| `--port INTEGER` | 8765 | Headless server TCP port; choose a valid free port. Server UI does not use this CLI value; Local launcher chooses its own. |
| `--pairing-code TEXT` | empty | Headless connection code; empty means no code check. Server UI/LAN client use their saved UI setting. Not encryption. |
| `--parent-control` | off, hidden | Internal child-process STOP/EOF pipe; do not pass manually. |

Choose **one** launch flag. Current branch precedence if accidentally combined is
client, server UI, headless server, CLI. This is not a supported multi-mode launch.

### Backend and decoder

| Argument | Default | Purpose / applicability |
| --- | --- | --- |
| `--backend faster-whisper\|moonshine` | faster-whisper | Kotoba direct Japanese→English, or Moonshine Japanese ASR→OPUS-MT English. |
| `--device cuda\|cpu` | cuda | Speech backend device; Moonshine applies it to ASR and MT. Does not select Qwen's device. |
| `--compute-type float16\|int8_float16\|int8` | int8_float16 | Kotoba/CTranslate2 precision. CPU converts FP16 options to INT8. Ignored by Moonshine, which uses FP32. |
| `--beam-size INTEGER` | 5 | Maximum search beam. Live always uses 1; phrase-only uses up to this maximum (capped at 2 when slow); commit-only uses at most 2. Moonshine ASR is always greedy; its MT uses the scheduled beam. Values below 1 are clamped. |
| `--vad` | on | Enable Kotoba decoder Silero VAD. Separate from scheduler energy detection. |
| `--no-vad` | off | Disable that Kotoba VAD; does not disable the scheduler energy gate. |
| `--translation-prompt TEXT` | empty | Kotoba English names/terms hint, normalized and bounded to 400 characters. Not guaranteed glossary enforcement. Ignored by Moonshine. |
| `--min-logprob FLOAT` | -1.0 | Kotoba segment confidence cutoff: higher/less negative rejects more. Lower may keep quiet speech but more errors. |
| `--max-compression-ratio FLOAT` | 2.4 | Kotoba repetition filter: lower is stricter; higher can admit more repetition. |
| `--max-temperature FLOAT` | 1.0 | Kotoba rejects segments at or above this decoder temperature. Current decoding uses temperature 0, so this is not a generation-creativity knob; setting 0 rejects temperature-0 segments. |
| `--min-words INTEGER` | 1 | Kotoba minimum accepted segment word count; increasing can drop valid short replies. |

Moonshine ignores the Kotoba-only decoder filters above. It caps ASR tokens using
audio duration instead. Do not interpret these filters as universal confidence
scores or expect `--compute-type int8` to quantize Moonshine.

### Scheduling and output

| Argument | Default | Purpose / applicability |
| --- | --- | --- |
| `--live` | on | Rolling draft replacement with stabilization; least waiting for complete phrases. |
| `--phrase-only` | off | Pause/punctuation-based phrase output, without visible previews. |
| `--commit-only` | off | One publication per phrase ID with committed-history tracking. Does not enable Qwen by itself. |
| `--buffer FLOAT` | 10 | Live decoding context. Smaller can reduce work but lose context; phrase/commit modes keep the current phrase instead. Must be positive. |
| `--audio-buffer FLOAT` | 20 | Available recovery history. Automatically at least `--buffer`, and in phrase/commit modes at least `--max-phrase-seconds + 15`. More history does not make inference faster. |
| `--step FLOAT` | 2 | Minimum live preview interval and nonempty display update interval. Phrase/commit decoding follows boundaries instead. Positive; clearing can bypass pacing. |
| `--caption-timeout FLOAT` | 5 | Clear both caption/history lines after inactivity; expire late fixes. Positive. Increase for slow Qwen, e.g. 8. Not a forced inference timeout. |
| `--phrase-pause FLOAT` | 1 | Quiet interval for phrase-only probes and live final-pass endpoint. Commit-only uses its own pause below. |
| `--commit-pause FLOAT` | 0.45 | Commit-only quiet interval. Lower splits sooner; higher waits longer. Continuation hints can delay once; extended-silence fallback is max(0.9, twice this value). |
| `--max-phrase-seconds FLOAT` | 30 | Safety split when phrase/commit speech lacks boundaries. Must exceed positive pause settings. Does not shorten the live window. |
| `--silence-threshold FLOAT` | 0.001 | Audio RMS activity threshold. Raise to ignore more quiet noise, at risk of missing soft speech; lower to keep quiet speech, at risk of never detecting pauses. Not an ASR confidence threshold. |

`--live`, `--phrase-only` and `--commit-only` are mutually exclusive. A short pause
is not proof of grammatical completion. Background music can keep the energy gate
active until the duration limit. History exhaustion means samples are already lost.

### Fixers

| Argument | Default | Purpose / applicability |
| --- | --- | --- |
| `--fixer` | off | Deterministic English spacing/repeated-loop cleanup in live/phrase-only; no model download. Bypassed in commit-only. |
| `--no-fixer` | on by default | Disable deterministic cleanup only; does **not** turn off Qwen. |
| `--llm-fixer` | off | Enable optional Qwen editing on CPU unless `--fixer-cuda` is supplied. |
| `--fixer-language en\|ja` | en | English editing after translation, or Moonshine Japanese preparation/editing before MT. `ja` requires Moonshine. Does not enable Qwen by itself. |
| `--fixer-cuda` | off | Enable Qwen and request all layers on CUDA with a compatible llama.cpp runtime. Independent of `--device`; no runtime installation is performed by this flag. |
| `--fixer-model PATH` | pinned Qwen model | Override with a local ChatML instruction GGUF. Quote paths containing spaces. Does not enable Qwen by itself; compatibility/quality of other models is not guaranteed. |
| `--fixer-interval FLOAT` | 0.5 | Minimum spacing between async fixer jobs, clamped to at least 0.1. Does not change model generation speed or display pacing. |
| `--fixer-context FLOAT` | 7 | Legacy configuration value; the current deterministic cleanup does not consume timed context. Does not resize Qwen history/token context. |

To disable Qwen, omit **both** `--llm-fixer` and `--fixer-cuda`. There is no
`--no-llm-fixer` argument. The deterministic and LLM fixers are independent.
English live mode uses one in-flight edit plus latest pending text. Japanese mode
must finish correction before MT; commit-only serializes finalization/publication.

### Diagnostics and compatibility

| Argument | Default | Purpose / applicability |
| --- | --- | --- |
| `--debug`, `--debug-logs` | off | Equivalent switches for extra inference, commit and Japanese-fixer diagnostics. May include caption/source text. |
| `--latency` | on | Enable periodic latency status reporting. |
| `--no-latency` | off | Disable periodic status only; does not speed up model inference or disable errors/debug logs. |
| `--warning-latency FLOAT` | 1 | Timing monitor's warning threshold; not an inference deadline. |
| `--backlog-latency FLOAT` | 2 | Timing monitor's backlog threshold; does not create or resize a model queue. |
| `--rollback FLOAT` | 3 | Legacy rollback setting accepted for compatibility; current live window replacement does not use the old rollback algorithm. |

There are no CLI flags for font, colors, window position, playback-device index,
Qwen CPU thread count, Qwen partial GPU layer count, or animation duration. Use
the GUI/settings for supported display/device preferences; the remaining values
are implementation settings, not undocumented command-line options.

## GUI and connections

Select a loopback device and connection in the settings window, then press RUN.
Right-click the overlay for settings, clear, pause and close actions; drag to
move it. Caption styling and connection preferences persist in `settings.json`.
Remote/Colab mode is not implemented; use Local or LAN.

Clients send 16 kHz mono float32 audio over TCP and receive caption JSON.
The server receives audio from the active client. Local mode uses localhost port
8765. A LAN server can use `--port` and `--pairing-code`.

## Model and limitations

Kotoba bilingual uses `language="en", task="translate"` for Japanese-to-English
translation: in this particular model the language token selects the target.
Do not substitute the generic Whisper source-language convention.

The model can still mistranslate names, omit details, or struggle with music,
overlapping speakers and sentences longer than the context window. The current
default pipeline is direct speech translation. The optional Moonshine backend
instead uses separate Japanese ASR and English translation models.

Design references:
- [Kotoba bilingual model card](https://huggingface.co/kotoba-tech/kotoba-whisper-bilingual-v1.0)
- [faster-whisper decoding and VAD](https://github.com/SYSTRAN/faster-whisper)
- [Whisper-Streaming agreement and adaptive latency](https://github.com/ufal/whisper_streaming)

## Verification

`pytest` is a development dependency, not included in the runtime requirements.
To run tests in the prepared environment:

```powershell
.runtime/moonshine-env/Scripts/python -m pip install --cache-dir .cache/pip pytest
.runtime/moonshine-env/Scripts/python -m pytest -q
```

Alternatively run `python -m pytest -q` in your chosen environment. Regression tests cover ring wraparound and variable
frame sizes, silence/short utterances, adaptive scheduling, corrections involving
negation/numbers, repeated captions, deterministic cleanup, and the shared engine
with a fake model. They do not measure translation accuracy on real recordings.

A local synthetic CPU check (48 kHz input, five-second window, one-second step)
measured resampling at approximately 2.42 ms per second of audio before versus
0.86 ms after, and five-second ring snapshots at about 0.009 ms. This measures
preprocessing only; GPU decoding usually dominates total latency.

## Troubleshooting

| Symptom | Check / next step |
| --- | --- |
| `ModuleNotFoundError` despite installing dependencies | Print `sys.executable`; install with that executable's `-m pip`. The base Python and `.runtime/moonshine-env` are different environments. |
| Moonshine dependency/version error | Use the prepared interpreter and `requirements-moonshine.txt`; Transformers 4.x does not provide this integration. Check existing Torch 2.6+ without replacing it automatically. |
| `CUDA is not available` | Run the Torch verification command in that exact interpreter. Use `--device cpu` while investigating the driver/build mismatch. |
| Kotoba hangs/errors on its first decode | Check matching CTranslate2 CUDA/cuBLAS/cuDNN DLLs, not just Torch's CUDA result. See the linked faster-whisper GPU instructions. |
| Qwen CUDA requested but rejected | The selected llama.cpp package is CPU-only/non-CUDA, or its GPU backend is unavailable. Install a matching CUDA wheel in the separate folder, restart, and inspect native offload logs. |
| `No matching distribution found` for the optional CUDA wheel | A compatible Windows/Python wheel may not exist at that index/version. Do not remove `--only-binary` unless you deliberately intend a C++/CUDA source build. Use CPU Qwen or follow official build instructions. |
| GPU allocation/OOM error | Remove `--fixer-cuda` first so Qwen runs on CPU; avoid multiple model-server instances. Reduce workload or use `--device cpu`. Quantization flags only affect Kotoba. |
| No captions, server appears idle | A headless/server UI process needs client audio. Check RUN, LAN address/code, playback loopback device and that audio is actually playing. |
| First Local launch times out | Model download/import may be slow. Start the same backend once in CLI/headless mode and wait for loading; then stop gracefully and retry Local. Moonshine Local startup allows up to ten minutes; the default Kotoba launcher allows one minute. |
| Captions disappear before a slow fixer finishes | Increase `--caption-timeout`, e.g. 8. The inactivity deadline invalidates unpublished results, intentionally. |
| Commit happens only at the duration limit | Look for continuous noise/music above the energy threshold; inspect `--debug`. Try a shorter `--commit-pause` or a recording-appropriate silence threshold. |
| Unrelated or incorrect English words | Compare `[JA-FIX]` Japanese source, corrected source and English, or `[COMMIT]` raw/final text. Test without Qwen to locate the stage introducing the error. Neither model nor heuristics guarantee accuracy. |
| Japanese characters fail when printed in the Windows terminal | For that PowerShell session, set `$env:PYTHONIOENCODING = 'utf-8'` before launching. The GUI handles Unicode separately. |
| OpenCV/NumPy warning during Moonshine imports | This workspace inherited an unrelated optional OpenCV binary mismatch; the recorded smoke test still succeeded. If it becomes a fatal import error, diagnose dependencies in the chosen environment rather than replacing Torch blindly. |
| GUI changes do not alter server model/mode | Backend/fixer/mode arguments are launch settings. Restart the model server with the new command; LAN clients cannot change remote inference options. |
| `--no-fixer` still runs Qwen | That flag disables only deterministic cleanup. Omit both `--llm-fixer` and `--fixer-cuda` to disable Qwen. |
| Local pairing fails | The auto-server code and client's saved pairing code must match. For the simplest Local setup leave both empty; use explicit headless/LAN setup for a custom code/port. |

For dependency diagnostics only (no installation):

```powershell
.runtime/moonshine-env/Scripts/python -m pip check
.runtime/moonshine-env/Scripts/python main.py --help
```

Closing the main window, the overlay's Close Transcript action, or ESC now exits
the app through cooperative cleanup. Use Show Main Window/Settings to return to
settings without quitting. A Closing status remains visible while active model
inference finishes; no new captions are emitted during shutdown. Socket readers,
capture, ASR and the optional fixer are joined before exit. Ctrl+C requests the
same cleanup; repeated Ctrl+C does not interrupt native teardown.

An automatically launched server receives STOP over a private parent pipe, and
also shuts down on parent EOF. Normal close no longer calls terminate/kill on
the model process. This prevents the known shutdown races; a genuinely hung
native driver may still leave the app waiting for inference. OS forced process
termination/power loss cannot guarantee cleanup. Existing crash dumps are not
deleted and Windows crash reporting is not disabled.

- Load the model before importing PyQt5 in a server process. On Windows, loading
  CTranslate2/CUDA after Qt can crash natively; the entry points preserve this order.
- CTranslate2 requires matching CUDA runtime DLLs. If CUDA 12 DLLs are missing,
  install `nvidia-cublas-cu12 nvidia-cuda-runtime-cu12`, or use `--device cpu`.
- If there is no audio, check that playback uses the loopback device selected in
  the client. The capture resampler handles its native sample rate.
