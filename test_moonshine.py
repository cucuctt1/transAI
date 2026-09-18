"""Backend regressions without network, model weights, CUDA, or Qt."""
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from moonshine_translator import MoonshineTranslator


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def to(self, device):
        return self

    def sum(self, dim):
        return Tensor(self.data.sum(axis=dim))

    def max(self):
        return Tensor(self.data.max())

    def item(self):
        return self.data.item()


def backend():
    obj = MoonshineTranslator.__new__(MoonshineTranslator)
    obj.device, obj.beam_size = 'cpu', 5
    obj._last_translation = None
    obj._torch = SimpleNamespace(inference_mode=nullcontext, long='long',
                                 tensor=lambda data, **kw: Tensor(data),
                                 ones_like=lambda t: Tensor(np.ones_like(t.data)))
    class Processor:
        def __call__(self, audio, **kw):
            assert kw == dict(return_tensors='pt', sampling_rate=16000)
            return dict(input_values=Tensor([audio]), attention_mask=Tensor([np.ones(len(audio))]))

        def batch_decode(self, output, **kw):
            return ['こんにちは。']
    obj.processor = Processor()
    obj.asr_calls, obj.mt_calls = [], []
    obj.model = SimpleNamespace(generate=lambda **kw: obj.asr_calls.append(kw))
    class Tokenizer:
        def encode(self, text, **kw):
            return list(range(12))

        def num_special_tokens_to_add(self, **kw):
            return 1

        def build_inputs_with_special_tokens(self, ids):
            return ids + [99]

        def batch_decode(self, output, **kw):
            return ['Hello.']
    obj.tokenizer = Tokenizer()
    obj.mt_model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8),
                                  generate=lambda **kw: obj.mt_calls.append(kw))
    return obj


def test_audio_mask_token_cap_english_and_translation_cache():
    obj = backend()
    result = obj.translate(np.ones(16000, dtype=np.float32), beam_size=1)
    assert result.source_text == 'こんにちは。'
    assert result.text == 'Hello. Hello.'
    assert result.duration == 1
    assert obj.asr_calls[0]['max_new_tokens'] == 8
    assert 'attention_mask' in obj.asr_calls[0]
    assert obj.asr_calls[0]['num_beams'] == 1
    # Input chunks retain all 12 tokens, with one EOS each.
    assert [len(call['input_ids'].data[0]) for call in obj.mt_calls] == [8, 6]
    assert all(call['num_beams'] == 1 for call in obj.mt_calls)
    obj.translate(np.ones(16000), beam_size=1)
    assert len(obj.mt_calls) == 2
    obj.translate(np.ones(16000), beam_size=3)
    assert len(obj.mt_calls) == 4


def test_silent_empty_invalid_input():
    obj = backend()
    assert obj.translate([]).text == ''
    assert obj.translate(np.zeros(16000)).text == ''
    assert not obj.asr_calls
    for audio in ([float('nan')], np.ones((2, 3))):
        with pytest.raises(ValueError):
            obj.translate(audio)


def test_backend_cli_and_factory(monkeypatch):
    from main import parse_args
    from asr_engine import load_model
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    assert parse_args().backend == 'faster-whisper'
    monkeypatch.setattr(sys, 'argv', ['main.py', '--backend', 'moonshine'])
    assert parse_args().backend == 'moonshine'
    calls = []
    monkeypatch.setattr('moonshine_translator.MoonshineTranslator',
                        lambda **kw: calls.append(kw) or 'moonshine-instance')
    # Importing translator would pull in CTranslate2; Moonshine must not do so.
    monkeypatch.setitem(sys.modules, 'translator', None)
    instance, device = load_model(backend='moonshine', device='cpu', log_cb=lambda s: None)
    assert instance == 'moonshine-instance' and device == 'CPU'
    assert calls[0]['device'] == 'cpu'
    with pytest.raises(ValueError):
        load_model(backend='unknown', device='cpu')


def test_local_server_forwards_backend(monkeypatch):
    from server import LocalServerLauncher
    calls = []
    launcher = LocalServerLauncher(extra_args=['--backend', 'moonshine', '--fixer-cuda'])
    assert launcher.startup_timeout == 600
    assert LocalServerLauncher(extra_args=['--backend=moonshine']).startup_timeout == 600
    assert LocalServerLauncher().startup_timeout == 60
    monkeypatch.setattr(launcher, 'is_up', lambda: True)
    monkeypatch.setattr('server.subprocess.Popen',
                        lambda command, **kw: calls.append(command) or SimpleNamespace(poll=lambda: None))
    assert launcher.ensure()
    index = calls[0].index('--backend')
    assert calls[0][index + 1] == 'moonshine'
    assert '--fixer-cuda' in calls[0]
