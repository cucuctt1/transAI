import threading
from types import SimpleNamespace

import numpy as np
import pytest

from japanese_fixer import JapaneseContext, validate_japanese_edit
from streaming import AudioRing, DecodeJob


def test_overlap_requires_committed_audio_overlap():
    context = JapaneseContext()
    first = DecodeJob(None, 0, 2, True, 1)
    context.commit(first, '明日は東京に行きます。')
    assert context.prepare('東京に行きます。その後帰ります。',
                           DecodeJob(None, 1, 3, True, 2)) == 'その後帰ります。'
    assert context.prepare('東京に行きます。その後帰ります。',
                           DecodeJob(None, 2, 4, True, 2)).startswith('東京に行きます。')
    assert context.prepare('明日は東京に行きます。', first) == ''
    fresh = JapaneseContext()
    assert fresh.prepare('はいはい。', first) == 'はいはい。'


def test_japanese_guard_keeps_numbers_negation_and_content():
    text = '今日は3人で行きません。'
    for candidate in ('今日は4人で行きません。', '今日は3人で行きます。',
                      '今日は3人で東京に行きません。', '', 'I will go.'):
        assert validate_japanese_edit(text, candidate) == text
    assert validate_japanese_edit('これはは本です。', 'これは本です。') == 'これは本です。'


def test_japanese_flag_requires_moonshine(monkeypatch):
    from main import parse_args, pipeline_config
    monkeypatch.setattr('sys.argv', ['main.py', '--fixer-language', 'ja'])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr('sys.argv', ['main.py', '--backend', 'moonshine', '--fixer-language', 'ja'])
    assert pipeline_config(parse_args()).fixer_language == 'ja'


@pytest.mark.parametrize('commit', [False, True])
def test_japanese_correction_precedes_mt_without_english_qwen(monkeypatch, commit):
    from asr_engine import ASREngine
    from transcript import Config
    from llm_fixer import AsyncFixer
    calls, states, shown, logs = [], [], [], []
    done = threading.Event()
    class Fixer:
        def fix_japanese(self, state):
            states.append(state)
            calls.append('ja-fix')
            return 'これは本です。'

        def close(self):
            pass
    monkeypatch.setattr('llm_fixer.AsyncFixer', lambda *a, **kw: AsyncFixer(factory=Fixer))
    class Source(AudioRing):
        def start(self):
            self.feed(np.tile(np.concatenate([np.full(16000, .1), np.zeros(16000)]), 2))

        def stop(self):
            pass
    class Model:
        beam_size = 1
        def transcribe_japanese(self, audio):
            calls.append('asr')
            return SimpleNamespace(text='これはは本です。', source_text='これはは本です。')

        def translate_source(self, source, beam):
            assert source == 'これは本です。'
            calls.append('mt')
            return 'This is a book.'
    def emit(text):
        if text:
            shown.append(text)
            if len(shown) == (2 if commit else 1):
                done.set()
    engine = ASREngine(Config(commit_only=commit, fixer_language='ja', enable_llm_fixer=True,
                              caption_timeout=8), Model(), 'fake', Source(45), caption_cb=emit,
                       log_cb=logs.append)
    engine.start()
    try:
        assert done.wait(4), logs
    finally:
        engine.stop()
    assert calls[:3] == ['asr', 'ja-fix', 'mt']
    if commit:
        assert states[1]['history'][0]['text'] == 'これは本です。'
    assert not any('[ERROR]' in item for item in logs)
