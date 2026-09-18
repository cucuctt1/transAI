import os
import re
import site
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import torch  # optional: CTranslate2 does not need torch
except Exception:
    torch = None

_LETTER = re.compile(r"[A-Za-z]")

# --- CUDA runtime resolution for CTranslate2 ---------------------------------
# faster-whisper's backend (CTranslate2) is a CUDA-12 build: it loads
# cublas64_12.dll / cudart64_12.dll lazily on the first inference. If torch bundles
# a different CUDA major (e.g. cu13 -> cublas64_13.dll) there is no cublas64_12.dll,
# and CTranslate2 HANGS on the first translate() instead of raising. So collect
# every directory that can provide the CUDA 12 runtime DLLs and add them to the
# process-local DLL search path. The torch/CUDA install is not modified.

_DLL_HANDLES = []


def _candidate_dll_dirs():
    dirs = []
    here = os.path.dirname(os.path.abspath(__file__))
    # 1) project-local drop-in folder: just copy cublas64_12.dll / cudart64_12.dll here
    for name in ("cuda12", "dlls"):
        d = os.path.join(here, name)
        if os.path.isdir(d):
            dirs.append(d)
    # 2) torch's bundled CUDA libs
    if torch is not None:
        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(d):
            dirs.append(d)
    # 3) a system CUDA Toolkit install (any v12.x)
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(cuda_base):
        try:
            for v in os.listdir(cuda_base):
                if v.startswith("v12"):
                    d = os.path.join(cuda_base, v, "bin")
                    if os.path.isdir(d):
                        dirs.append(d)
        except OSError:
            pass
    # 4) nvidia-*-cu12 pip wheels: site-packages/nvidia/<pkg>/bin
    try:
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        roots = []
    for root in roots:
        nv = os.path.join(root, "nvidia")
        if not os.path.isdir(nv):
            continue
        for pkg in os.listdir(nv):
            for sub in ("bin", "lib", os.path.join("bin", "x64")):
                d = os.path.join(nv, pkg, sub)
                if os.path.isdir(d):
                    dirs.append(d)
    return dirs


def _register_cuda_dirs():
    found = set()
    for d in _candidate_dll_dirs():
        try:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(d))
        except Exception:
            pass
        try:
            for f in os.listdir(d):
                low = f.lower()
                if low.startswith("cublas64") or low.startswith("cudart64"):
                    found.add(low)
        except Exception:
            pass
    return found


_CUDA_DLLS = _register_cuda_dirs()
if os.name == "nt" and "cublas64_12.dll" not in _CUDA_DLLS:
    _found = ", ".join(sorted(f for f in _CUDA_DLLS if f.startswith("cublas64"))) or "none"
    print(
        f"[CUDA] cublas64_12.dll NOT found (found: {_found}).\n"
        "[CUDA] CTranslate2 is a CUDA-12 build and will HANG on the first inference.\n"
        "[CUDA] Fix:  pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12\n"
        "[CUDA] Or run the server with --device cpu.",
        file=sys.stderr,
        flush=True,
    )

from faster_whisper import WhisperModel


@dataclass
class SegmentResult:
    text: str
    start: float
    end: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temperature: float


@dataclass
class TranslateResult:
    text: str
    duration: float
    duration_after_vad: float
    segments: list = field(default_factory=list)


class Translator:
    """Loads the Kotoba-Whisper-Bilingual model once and translates Japanese speech to English.

    Junk (hallucinated/fragmented) output is filtered using signals the model already
    computes during decoding (avg log-prob, compression ratio, no-speech prob), plus an
    optional Silero VAD pass. No extra model or added latency is required.
    """

    def __init__(
        self,
        model_id="kotoba-tech/kotoba-whisper-bilingual-v1.0-faster",
        compute_type="int8_float16",
        vad=True,
        min_words=1,
        min_logprob=-1.0,
        max_compression_ratio=2.4,
        max_temperature=1.0,
        device="cuda",
    ):
        self.model_id = model_id
        self.device = device
        if device == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"  # float16 is not supported on CPU
        self.compute_type = compute_type
        self.vad = vad
        self.min_words = min_words
        self.min_logprob = min_logprob
        self.max_compression_ratio = max_compression_ratio
        self.max_temperature = max_temperature
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        self._warmup()

    def _warmup(self):
        # Trigger CUDA kernel/cuBLAS initialization now so the first real
        # inference does not pay a one-time warm-up spike.
        try:
            segments, _ = self.model.transcribe(
                np.zeros(16000, dtype=np.float32),
                language="en",
                task="translate",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=self.vad,
            )
            for _ in segments:
                pass
        except Exception:
            pass

    def _is_junk(self, seg):
        text = (seg.text or "").strip()
        if not text:
            return True
        if not _LETTER.search(text):
            return True
        if len(text.split()) < self.min_words:
            return True
        if seg.avg_logprob is not None and seg.avg_logprob < self.min_logprob:
            return True
        if seg.compression_ratio is not None and seg.compression_ratio > self.max_compression_ratio:
            return True
        if seg.temperature is not None and seg.temperature >= self.max_temperature:
            return True
        return False

    def translate(self, audio, prompt=None):
        audio = np.asarray(audio, dtype=np.float32)
        kwargs = dict(
            language="en",
            task="translate",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=self.vad,
        )
        if prompt:
            kwargs["initial_prompt"] = prompt

        segments, info = self.model.transcribe(audio, **kwargs)

        parts = []
        seg_results = []
        for seg in segments:
            sr = SegmentResult(
                text=(seg.text or "").strip(),
                start=seg.start,
                end=seg.end,
                avg_logprob=seg.avg_logprob,
                compression_ratio=seg.compression_ratio,
                no_speech_prob=seg.no_speech_prob,
                temperature=seg.temperature,
            )
            seg_results.append(sr)
            if not self._is_junk(seg):
                if sr.text:
                    parts.append(sr.text)

        return TranslateResult(
            text=" ".join(parts).strip(),
            duration=info.duration,
            duration_after_vad=info.duration_after_vad,
            segments=seg_results,
        )
