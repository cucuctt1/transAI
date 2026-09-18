import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch

_LETTER = re.compile(r"[A-Za-z]")

# CTranslate2 loads cuBLAS lazily from the system. Torch bundles the CUDA 12
# runtime (cublas64_12.dll, cudart64_12.dll, ...) under torch/lib. Point the
# process-local DLL search path at it so CTranslate2 can resolve cuBLAS without
# touching the existing CUDA/PyTorch installation.
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
_dll_handle = None
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _dll_handle = os.add_dll_directory(_torch_lib)

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
