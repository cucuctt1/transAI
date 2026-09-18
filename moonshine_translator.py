"""Optional Japanese ASR -> English MT backend, without importing CTranslate2.

Uses the documented Transformers window-generation API. Audio scheduling and
shutdown remain owned by ASREngine; this is not a persistent encoder-cache stream.
"""
from dataclasses import dataclass, field

import numpy as np

from model_storage import configure

ASR_MODEL = "moonshine-ai/moonshine-streaming-tiny-ja"
ASR_REVISION = "a82871d17fede2efad4540d4a32eabb515fea5fa"
MT_MODEL = "Helsinki-NLP/opus-mt-ja-en"
MT_REVISION = "0770961a39ba6bd66305b149c3f4110bcafca2e6"


@dataclass
class MoonshineResult:
    text: str
    duration: float
    duration_after_vad: float
    segments: list = field(default_factory=list)
    source_text: str = ""


class MoonshineTranslator:
    def __init__(self, device="cuda", beam_size=5, log_cb=None):
        cache = str(configure() / "moonshine")
        try:
            import torch
            from transformers import (AutoProcessor, MoonshineStreamingForConditionalGeneration,
                                      MarianMTModel, MarianTokenizer)
            import sentencepiece  # noqa: F401 - fail early with an actionable message
            import sacremoses  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Moonshine dependencies unavailable. Run: "
                               "python -m pip install -r requirements-moonshine.txt "
                               "(requires Transformers 5.2 and PyTorch 2.6+).") from exc
        self._torch = torch
        self.device = device
        self.beam_size = max(1, beam_size)
        self.compute_type = "float32"
        self.vad = False
        self.model_id = ASR_MODEL
        log = log_cb or (lambda line: None)
        options = dict(cache_dir=cache, revision=ASR_REVISION)
        log("Loading Moonshine Japanese ASR (repo-local cache)...")
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL, **options)
        # Keep checkpoint precision: tiny ASR is sensitive on short utterances.
        self.model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            ASR_MODEL, dtype=torch.float32, **options).to(device).eval()
        options = dict(cache_dir=cache, revision=MT_REVISION)
        log("Loading OPUS-MT Japanese -> English (repo-local cache)...")
        self.tokenizer = MarianTokenizer.from_pretrained(MT_MODEL, **options)
        self.mt_model = MarianMTModel.from_pretrained(
            # This pinned repository ships PyTorch weights only. Do not trigger
            # a Hub-side safetensors conversion (and another download) at launch.
            MT_MODEL, dtype=torch.float32, use_safetensors=False, **options).to(device).eval()
        self._last_translation = None

    def _translate_text(self, text, beam):
        if not text:
            return ""
        key = (text, beam)
        if self._last_translation is not None and self._last_translation[0] == key:
            return self._last_translation[1]
        # Never silently truncate a long transcript. Split token IDs within
        # Marian's positional limit, reserving room for its EOS token.
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        limit = min(512, self.mt_model.config.max_position_embeddings)
        size = limit - self.tokenizer.num_special_tokens_to_add(pair=False)
        parts = []
        for start in range(0, len(ids), size):
            chunk = self.tokenizer.build_inputs_with_special_tokens(ids[start:start + size])
            input_ids = self._torch.tensor([chunk], dtype=self._torch.long, device=self.device)
            output = self.mt_model.generate(
                input_ids=input_ids, attention_mask=self._torch.ones_like(input_ids),
                num_beams=beam, do_sample=False, max_new_tokens=limit - 1,
            )
            parts.append(self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip())
        translated = " ".join(p for p in parts if p)
        self._last_translation = (key, translated)
        return translated

    def transcribe_japanese(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Moonshine expects finite mono 16 kHz audio")
        duration = len(audio) / 16000
        if not len(audio) or not np.any(audio):
            return MoonshineResult("", duration, duration)
        with self._torch.inference_mode():
            inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
            if "attention_mask" not in inputs:
                raise RuntimeError("Moonshine processor did not return its required attention_mask")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            # Model-card bound prevents repetition loops on noisy/short clips.
            samples = inputs["attention_mask"].sum(dim=-1).max().item()
            tokens = max(2, int(samples * 6.5 / 16000) + 2)
            generated = self.model.generate(**inputs, max_new_tokens=tokens,
                                            num_beams=1, do_sample=False)
            source = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return MoonshineResult(source, duration, duration, source_text=source)

    def translate_source(self, source, beam_size=None):
        beam = self.beam_size if beam_size is None else max(1, beam_size)
        with self._torch.inference_mode():
            return self._translate_text(source, beam)

    def translate(self, audio, prompt=None, beam_size=None):
        result = self.transcribe_japanese(audio)
        result.text = self.translate_source(result.source_text, beam_size)
        return result
