import threading
from types import SimpleNamespace

import numpy as np
import pytest

from phrase_commit import CommitLedger, CommitOutput, CommitPolicy, boundary_ready, finalization_state
from streaming import AudioRing, DecodeJob, PhrasePolicy


@pytest.mark.parametrize('source,ready', [
    ('私は', False), ('駅に。', False), ('雨なので', False), ('行って', False),
    ('行きません。', True), ('行きました', True), ('そうですか？', True),
    ('はい。', True), ('寒い。', True), ('ですが', False)])
def test_japanese_boundary_hints(source, ready):
    assert boundary_ready('A sentence.', source) is ready


@pytest.mark.parametrize('source', ['明日行く', '寒い', 'ありがとう', 'はい', '行くよ', ''])
def test_plain_endings_and_unpunctuated_english_do_not_block(source):
    assert boundary_ready('We can go tomorrow', source)


def test_short_conversational_pauses_commit_without_safety_split():
    policy = CommitPolicy()
    audio = np.tile(np.concatenate([np.full(16000, .1), np.zeros(8000)]), 3)
    jobs = []
    while True:
        job = policy.next(audio, 4.5)
        if job is None:
            break
        assert not policy.forced_split
        assert not policy.should_wait(job, 'We can go', '行く')
        assert len(job.audio) / 16000 <= 1.33  # includes small pre/post roll, not all silence
        assert policy.accept_result(job, 'We can go', complete=True)
        jobs.append(job)
    assert len(jobs) == 3
    assert jobs[0].end == pytest.approx(1.46)
    assert all(b.start >= a.end for a, b in zip(jobs, jobs[1:]))


def test_continuation_can_wait_only_once_across_resumed_speech():
    policy = CommitPolicy()
    audio = np.tile(np.concatenate([np.full(16000, .1), np.zeros(8000)]), 2)
    first = policy.next(audio, 3)
    assert policy.should_wait(first, 'Because it rains', '雨なので')
    assert not policy.accept_result(first, 'Because it rains', complete=False)
    second = policy.next(audio, 3)
    assert second.utterance == first.utterance
    assert second.cached_text is None
    assert not policy.should_wait(second, 'But it rains', '雨ですが')
    assert policy.accept_result(second, 'But it rains', complete=True)


def test_continuation_long_pause_releases_before_one_second():
    policy = CommitPolicy()
    audio = np.concatenate([np.full(16000, .1), np.zeros(16000)])
    first = policy.next(audio, 2)
    assert policy.should_wait(first, 'I was going to', '私は')
    policy.accept_result(first, 'I was going to', complete=False)
    second = policy.next(audio, 2)
    assert second.cached_text == 'I was going to'
    assert second.end == pytest.approx(1.9)


def test_finalizer_rejects_out_of_context_added_words():
    from llm_fixer import validate_commit_edit
    raw = 'We will go to the station.'
    assert validate_commit_edit(raw, 'We will go to the station tomorrow.') == raw
    assert validate_commit_edit('We went to the to the store.', 'We went to the store.') == 'We went to the store.'


def test_ledger_rejects_duplicate_stale_and_audio_overlap_not_intentional_repeats():
    ledger = CommitLedger()
    assert ledger.record(1, 0, 2, 'Yes.')
    assert not ledger.record(1, 0, 2, 'Different text.')
    assert not ledger.record(2, 1, 3, 'Overlap.')
    assert ledger.record(2, 2, 4, 'Yes.')
    assert [x['text'] for x in ledger.history] == ['Yes.', 'Yes.']


def test_history_records_only_publication_and_expiry_drops_pending():
    shown = []
    out = CommitOutput(shown.append)
    out.observe(np.full(16000, .1), 1, 0)
    job = DecodeJob(None, 0, 1, True, 1)
    assert out.offer_commit(job, 'Hello.')
    assert out.history() == []
    out.tick(0)
    assert len(out.history()) == 1
    assert out.offer_commit(job, 'Duplicate.')
    out.tick(1)
    assert shown == ['Hello.']
    assert out.offer_commit(DecodeJob(None, 1, 2, True, 2), 'Late.')
    out.tick(3)
    assert shown == ['Hello.', '']
    assert len(out.history()) == 1


def test_incomplete_clause_retains_audio_then_long_pause_releases_once():
    policy = PhrasePolicy()
    audio = np.concatenate([np.full(16000, .1), np.zeros(32000)])
    first = policy.next(audio, 3)
    assert not policy.accept_result(first, 'Because it rains.', complete=False)
    assert policy.consumed == 0
    second = policy.next(audio, 3)
    assert second.cached_text == 'Because it rains.'
    assert policy.accept_result(second, second.cached_text, complete=True)
    assert policy.consumed == pytest.approx(3)


def test_commit_cli_is_explicit_and_enables_phrase_buffer(monkeypatch):
    from main import parse_args, pipeline_config
    monkeypatch.setattr('sys.argv', ['main.py', '--commit-only', '--llm-fixer'])
    cfg = pipeline_config(parse_args())
    assert cfg.commit_only and cfg.phrase_only and cfg.enable_llm_fixer
    assert cfg.audio_buffer >= cfg.max_phrase_seconds + 15
    monkeypatch.setattr('sys.argv', ['main.py', '--commit-only', '--live'])
    with pytest.raises(SystemExit):
        parse_args()


def test_engine_finalizes_in_order_with_published_history(monkeypatch):
    from asr_engine import ASREngine
    from llm_fixer import AsyncFixer
    from transcript import Config
    states, shown, errors = [], [], []
    done = threading.Event()

    class Fixer:
        def finalize(self, state):
            states.append(state)
            return {'action': 'commit', 'new_text': state['current']}

        def close(self):
            pass

    monkeypatch.setattr('llm_fixer.AsyncFixer', lambda *args, **kwargs: AsyncFixer(factory=Fixer))

    class Source(AudioRing):
        def start(self):
            self.feed(np.tile(np.concatenate([np.full(16000, .1), np.zeros(16000)]), 2))

        def stop(self):
            pass

    def emit(text):
        if text:
            shown.append(text)
            if len(shown) == 2:
                done.set()

    model = SimpleNamespace(beam_size=1, translate=lambda *a, **kw:
                            SimpleNamespace(text='Yes.', source_text='はい。'))
    engine = ASREngine(Config(commit_only=True, enable_llm_fixer=True, caption_timeout=8),
                       model, 'fake', Source(45), caption_cb=emit, log_cb=errors.append)
    engine.start()
    try:
        assert done.wait(4), errors
    finally:
        engine.stop()
    assert shown == ['Yes.', 'Yes.']
    assert states[0]['committed_history'] == []
    assert states[1]['committed_history'][0]['text'] == 'Yes.'
    assert states[1]['start'] >= states[0]['end']


def test_structured_finalizer_wait_guard_and_forced_fallback():
    from llm_fixer import LocalModelFixer
    fixer = LocalModelFixer.__new__(LocalModelFixer)
    state = finalization_state(DecodeJob(None, 0, 2, True, 1), 'I cannot pay 35.', '', [])
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content':
                '{"action":"commit","new_text":"I can pay 350."}'}}]}
    fixer.model = SimpleNamespace(create_chat_completion=lambda **kw: response,
                                  tokenize=lambda *a, **kw: [], n_ctx=lambda: 2048)
    assert fixer.finalize(state)['new_text'] == state['current']
    response['choices'][0]['message']['content'] = '{"action":"wait","new_text":""}'
    assert fixer.finalize(state)['action'] == 'wait'
    state['forced_boundary'] = True
    assert fixer.finalize(state) == {'action': 'commit', 'new_text': state['current']}
    state['forced_boundary'] = False
    state['audio_boundary_accepted'] = True
    assert fixer.finalize(state) == {'action': 'commit', 'new_text': state['current']}
