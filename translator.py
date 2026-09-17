import os

import numpy as np
import torch

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


class Translator:
    """Loads the Kotoba-Whisper-Bilingual model once and translates Japanese speech to English."""

    def __init__(self, model_id="kotoba-tech/kotoba-whisper-bilingual-v1.0-faster", compute_type="int8_float16"):
        self.model_id = model_id
        self.compute_type = compute_type
        self.model = WhisperModel(model_id, device="cuda", compute_type=compute_type)
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
            )
            for _ in segments:
                pass
        except Exception:
            pass

    def translate(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            task="translate",
            beam_size=1,
            condition_on_previous_text=False,
        )
        parts = [seg.text.strip() for seg in segments]
        return " ".join(p for p in parts if p).strip()
