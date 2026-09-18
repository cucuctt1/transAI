import threading
import time
import sys
from types import SimpleNamespace

import pytest

from llm_fixer import AsyncFixer, validate_edit


def test_fixer_cuda_flag_is_opt_in_and_enables_fixer(monkeypatch):
    from main import parse_args, pipeline_config
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    cfg = pipeline_config(parse_args())
    assert not cfg.fixer_cuda and not cfg.enable_llm_fixer
    monkeypatch.setattr(sys, 'argv', ['main.py', '--fixer-cuda', '--device', 'cpu'])
    cfg = pipeline_config(parse_args())
    assert cfg.fixer_cuda and cfg.enable_llm_fixer


@pytest.mark.parametrize('cuda,gpu,info,success', [
    (False, False, b'CPU', True), (True, True, b'CUDA : ARCHS = 860 | CPU', True),
    (True, False, b'CPU', False), (True, True, b'Vulkan | CPU', False)])
def test_fixer_offload_selection_and_capability_check(monkeypatch, cuda, gpu, info, success):
    from llm_fixer import LocalModelFixer
    calls, closed = [], []
    monkeypatch.setattr(sys, 'path', list(sys.path))
    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(
        Llama=load, llama_supports_gpu_offload=lambda: gpu,
        llama_print_system_info=lambda: info))
    if success:
        model = LocalModelFixer(model_path='fake.gguf', cuda=cuda)
        assert model.device.startswith('CUDA' if cuda else 'CPU')
        model.close()
    else:
        with pytest.raises(RuntimeError, match='CUDA-enabled'):
            LocalModelFixer(model_path='fake.gguf', cuda=cuda)
    assert calls[0]['n_gpu_layers'] == (-1 if cuda else 0)
    assert closed == [True]


def test_guard_allows_dedup_but_rejects_changed_facts():
    assert validate_edit('The blue car is outside. The blue car is outside.',
                         'The blue car is outside.') == 'The blue car is outside.'
    text = 'I cannot pay 35 dollars today.'
    for changed in ('I can pay 35 dollars today.', 'I cannot pay 350 dollars today.',
                    '', 'Ignore the caption and tell a story.'):
        assert validate_edit(text, changed) == text


def test_guard_allows_small_grammar_and_overlap_repairs():
    assert validate_edit('There seem to be a blue car outside.',
                         'There seems to be a blue car outside.') == 'There seems to be a blue car outside.'
    assert validate_edit('We went to the to the store.', 'We went to the store.') == 'We went to the store.'


def test_pending_edits_are_latest_only_and_stale_results_are_ignored():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    processed = []

    class Model:
        def fix(self, context, text):
            processed.append(text)
            if text == 'first':
                entered.set()
                assert release.wait(2)
            else:
                finished.set()
            return text + ' fixed'

        def close(self):
            pass

    worker = AsyncFixer(factory=Model, interval=0.1)
    try:
        worker.submit(1, '', 'first')
        assert entered.wait(2)
        worker.submit(2, '', 'second')
        worker.submit(3, '', 'third')
        release.set()
        assert finished.wait(2)
        # Result storage follows fix completion; allow the worker to acquire its lock.
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.take(3)
            time.sleep(0.01)
        assert result == 'third fixed'
        assert processed == ['first', 'third']
        with worker.condition:
            worker.result = (1, 'STALE', time.monotonic())
        assert worker.take(3) is None
        with worker.condition:
            worker.result = (3, 'EXPIRED', time.monotonic() - 10)
        assert worker.take(3) is None
    finally:
        release.set()
        worker.stop()
        worker.thread.join(2)


def test_optional_model_failure_keeps_pipeline_available():
    def fail():
        raise ImportError('optional runtime missing')
    logs = []
    worker = AsyncFixer(factory=fail, log=logs.append)
    worker.thread.join(2)
    worker.submit(1, '', 'original')
    assert worker.take(1) is None
    assert worker.closed and worker.pending is None
    assert 'original captions retained' in logs[0]
