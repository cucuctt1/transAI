"""Regression tests for real audio scheduling and caption policies, without a GPU."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from asr_engine import ASREngine
from streaming import AudioRing, StreamPolicy, CaptionStabilizer, DecodeJob, PhrasePolicy
from transcript import BackgroundFixer, Config


def test_ring_variable_blocks_wrap_and_snapshot_isolation():
    ring = AudioRing(1, rate=10)
    ring.feed(np.arange(7))
    assert ring.latest(1) is None
    ring.feed(np.arange(7, 14))
    data, end = ring.latest(1)
    np.testing.assert_array_equal(data, np.arange(4, 14))
    assert end == 1.4
    ring.feed(np.arange(14, 40))
    np.testing.assert_array_equal(ring.latest(1)[0], np.arange(30, 40))
    np.testing.assert_array_equal(data, np.arange(4, 14))
    assert ring.available_seconds() == 1


def simulate(blocks, inference=0):
    ring = AudioRing(12)
    policy = StreamPolicy()
    policy.record_duration(inference)
    jobs = []
    for level in blocks:
        ring.feed(np.full(3200, level, dtype=np.float32))
        audio, end = ring.latest(ring.available_seconds())
        job = policy.next(audio, end)
        if job:
            jobs.append(job)
        # Repeated polling of the same buffer cannot schedule work.
        assert policy.next(audio, end) is None
    return jobs


def test_no_silence_decodes_and_one_final_pass():
    assert simulate([0] * 100) == []
    jobs = simulate([0.1] * 10 + [0] * 40)
    assert sum(j.final for j in jobs) == 1
    assert len(jobs) <= 4
    assert jobs[0].end <= 1.2  # no initial 5-second wait
    assert jobs[-1].end <= 3.2


def test_short_utterance_and_new_utterance_excludes_old_audio():
    jobs = simulate([0.1] * 2 + [0] * 10 + [0.1] * 2 + [0] * 10)
    assert len(jobs) == 2 and all(j.final for j in jobs)
    assert jobs[1].utterance > jobs[0].utterance
    assert jobs[1].start >= jobs[0].end


def test_slow_inference_does_not_add_a_second_scheduler_delay():
    policy = StreamPolicy()
    policy.record_duration(2)
    first = policy.next(np.full(16000, 0.1, dtype=np.float32), 1)
    assert first is not None
    # Two seconds pass inside inference. Fresh work must start immediately.
    assert policy.next(np.full(48000, 0.1, dtype=np.float32), 3) is not None
    assert policy.decode_beam(5, final=False) == 1
    assert policy.decode_beam(5, final=True) == 1


def test_short_thinking_pause_keeps_sentence_context():
    jobs = simulate([0.1] * 10 + [0] * 3 + [0.1] * 10 + [0] * 10)
    assert len({j.utterance for j in jobs}) == 1
    assert sum(j.final for j in jobs) == 1
    assert jobs[-1].start == 0


def test_slow_decode_recovers_unread_audio_with_overlap():
    policy = StreamPolicy(window=5)
    assert policy.next(np.full(16000, 0.1, dtype=np.float32), 1)
    # Worker returns after seven seconds; the normal 5s tail would skip 2s.
    seconds = policy.snapshot_seconds(live=8, available=8)
    assert seconds == 8
    result = policy.next(np.full(int(seconds * 16000), 0.1, dtype=np.float32), 8)
    assert result.start == 0
    assert result.end == 8


def test_context_stays_bounded_for_continuous_speech():
    jobs = simulate([0.1] * 200)
    assert all(len(j.audio) <= 8 * 16000 for j in jobs)
    assert jobs[-1].start > 0


def job(end, final=False, utterance=1):
    return DecodeJob(None, max(0, end - 5), end, final, utterance)


def test_caption_corrections_negation_numbers_and_stale_results():
    c = CaptionStabilizer()
    assert c.update('I can go at 3.', job(1))
    assert c.update('I cannot go at 3.', job(2)) == 'I cannot go at 3.'
    assert c.update('I cannot go at 4.', job(3)) == 'I cannot go at 4.'
    assert c.update('I cannot go at 4!', job(4)) is None
    assert c.update('old result', job(2)) is None


def test_caption_oscillation_requires_confirmation_and_repeat_after_pause_allowed():
    c = CaptionStabilizer()
    assert c.update('The blue one.', job(1))
    assert c.update('The green one.', job(2))
    assert c.update('The blue one.', job(3)) is None
    assert c.update('The blue one.', job(4)) == 'The blue one.'
    assert c.update('The green one.', job(5, final=True)) == 'The green one.'
    assert c.update('The green one.', job(6, utterance=2)) == 'The green one.'


def test_caption_bound_and_punctuation():
    c = CaptionStabilizer(max_words=4)
    assert c.update('Hello everyone. Welcome back today!', job(1)) == 'everyone. Welcome back today!'
    assert len(c.history) <= 8


def test_fixer_removes_loops_preserves_emphasis_numbers_and_negation():
    fix = BackgroundFixer().fix_unstable_transcript
    assert fix('', 'Thank you for watching. ' * 3) == 'Thank you for watching.'
    text = 'No, no. I had had 3.5 dollars. I cannot go.'
    assert fix('', text) == text
    assert fix('', 'I love you. I love you.') == 'I love you. I love you.'
    assert fix('', 'Hello  , world !') == 'Hello, world!'


@pytest.mark.parametrize('phrase_only, missing_punctuation', [(False, False), (True, False), (True, True)])
def test_shared_engine_preview_final_cleanup_dedup_and_shutdown(phrase_only, missing_punctuation):
    expected_text = 'Thank you for watching' + ('' if missing_punctuation else '.')
    class Source:
        def __init__(self):
            self.ring = AudioRing(5)
            self.blocks = iter([0.1] * 10 + [0] * 20)
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def available_seconds(self):
            self.ring.feed(np.full(3200, next(self.blocks, 0), dtype=np.float32))
            return self.ring.available_seconds()

        def live_position(self):
            return self.ring.live_position()

        def latest(self, seconds):
            return self.ring.latest(seconds)

    class Model:
        beam_size = 5

        def __init__(self):
            self.beams = []

        def translate(self, audio, beam_size=None):
            self.beams.append(beam_size)
            return SimpleNamespace(text=((expected_text + ' ') * 3 if len(audio) >= 48000 else 'Thanks.'))

    source, model = Source(), Model()
    captions, errors = [], []
    finished = threading.Event()

    def caption(text):
        captions.append(text)
        if text == expected_text:
            finished.set()

    engine = ASREngine(Config(whisper_step=1, phrase_only=phrase_only, enable_background_fixer=True), model, 'fake', source,
                       caption_cb=caption, log_cb=errors.append)
    engine.start()
    try:
        assert finished.wait(4), errors
    finally:
        engine.stop()
    assert not engine.is_running()
    assert source.stopped
    assert model.beams[0] == (5 if phrase_only else 1)
    assert model.beams[-1] == (5 if phrase_only else 1)
    assert captions == ([expected_text] if phrase_only else ['Thanks.', expected_text])
    if phrase_only:
        assert len(model.beams) == 1  # including the missing-punctuation fallback
    assert engine.final_passes == 1
    assert not any('[ERROR]' in e for e in errors)


def test_phrase_has_no_preview_and_keeps_more_than_live_window():
    ring = AudioRing(45)
    policy = PhrasePolicy()
    for _ in range(60):  # twelve seconds of speech
        ring.feed(np.full(3200, 0.1, dtype=np.float32))
        assert policy.next(*ring.latest(ring.available_seconds())) is None
    ring.feed(np.zeros(16000, dtype=np.float32))
    phrase = policy.next(*ring.latest(ring.available_seconds()))
    assert phrase.final and phrase.start == 0
    assert phrase.end == pytest.approx(13)
    assert len(phrase.audio) == 13 * 16000
    assert policy.next(*ring.latest(ring.available_seconds())) is None


def test_phrase_busy_decoder_preserves_order_and_cuts_completed_audio():
    # Two entire phrases arrive while the worker is busy. Each must be returned
    # separately, even though both are in a single snapshot.
    audio = np.concatenate([np.full(16000, 0.1), np.zeros(16000),
                            np.full(16000, 0.2), np.zeros(16000)]).astype(np.float32)
    policy = PhrasePolicy()
    first = policy.next(audio, 4)
    assert policy.accept_result(first, 'First sentence.')
    second = policy.next(audio, 4)
    assert policy.accept_result(second, 'Second sentence.')
    assert first.end == pytest.approx(2)
    assert second.start == pytest.approx(first.end)
    assert second.end == pytest.approx(4)
    assert second.utterance > first.utterance
    assert not np.any(second.audio == np.float32(0.1))
    assert policy.next(audio, 4) is None


def test_phrase_safety_split_and_full_caption():
    policy = PhrasePolicy(maximum=2)
    audio = np.full(48000, 0.1, dtype=np.float32)
    phrase = policy.next(audio, 3)
    assert policy.forced_split and phrase.end == pytest.approx(2)
    assert policy.accept_result(phrase, 'An unfinished thought')
    assert policy.consumed == pytest.approx(2)
    text = ' '.join(f'word{i}' for i in range(60)) + '.'
    assert CaptionStabilizer(None).update(text, phrase) == text


def test_phrase_config_keeps_whole_phrase_plus_inference_headroom():
    assert Config(phrase_only=True).audio_buffer >= 45
    with pytest.raises(ValueError):
        Config(endpoint_silence=0)


@pytest.mark.parametrize('text, complete', [
    ('A complete sentence.', True), ('A question?', True), ('Stop!', True),
    ('He said "hello."', True), ('The price is 3.5', False),
    ('Ask Dr.', False), ('In the U.S.', False), ('Wait...', False),
    ('First sentence. But the next part', False), ('', False),
])
def test_phrase_terminal_punctuation(text, complete):
    assert PhrasePolicy.sentence_complete(text) is complete


def test_unfinished_translation_preserves_audio_across_pause():
    policy = PhrasePolicy()
    audio = np.concatenate([np.full(16000, 0.1), np.zeros(16000)]).astype(np.float32)
    probe = policy.next(audio, 2)
    assert policy.consumed == 0
    assert not policy.accept_result(probe, 'Because I was')
    # Continued silence doesn't run the same translation over and over.
    audio = np.concatenate([audio, np.zeros(8000)]).astype(np.float32)
    assert policy.next(audio, 2.5) is None
    # Keep the beginning when speech resumes and only cut on a complete result.
    audio = np.concatenate([audio, np.full(16000, 0.2), np.zeros(16000)]).astype(np.float32)
    phrase = policy.next(audio, 4.5)
    assert phrase.start == 0
    assert np.any(phrase.audio == np.float32(0.1))
    assert policy.accept_result(phrase, 'Because I was tired, I went home.')
    assert policy.consumed == pytest.approx(4.5)
    assert not policy.accept_result(probe, 'Stale result.')


def test_missing_punctuation_releases_cached_text_after_long_pause():
    policy = PhrasePolicy()
    audio = np.concatenate([np.full(16000, 0.1), np.zeros(16000)]).astype(np.float32)
    probe = policy.next(audio, 2)
    assert not policy.accept_result(probe, 'Thank you for coming')
    audio = np.concatenate([audio, np.zeros(16000)]).astype(np.float32)
    ready = policy.next(audio, 3)
    assert ready.audio is None  # no second inference
    assert ready.cached_text == 'Thank you for coming'
    assert policy.accept_result(ready, ready.cached_text)
    assert policy.start is None and policy.consumed == pytest.approx(3)


def test_empty_translation_does_not_stall_phrase_buffer():
    policy = PhrasePolicy()
    audio = np.concatenate([np.full(16000, 0.1), np.zeros(32000)]).astype(np.float32)
    probe = policy.next(audio, 3)
    assert not policy.accept_result(probe, '')
    ready = policy.next(audio, 3)
    assert ready.cached_text == ''
    assert not policy.accept_result(ready, '')
    assert policy.start is None


def test_phrase_beam_reduces_when_processing_is_slow():
    policy = PhrasePolicy()
    assert policy.decode_beam(5, True) == 5
    policy.record_duration(3)
    assert policy.decode_beam(5, True) == 2


def test_mode_selection_is_explicit(monkeypatch):
    from main import parse_args
    for arguments, expected in [([], False), (['--live'], False), (['--phrase-only'], True)]:
        monkeypatch.setattr('sys.argv', ['main.py', *arguments])
        assert parse_args().phrase_only is expected


def test_local_launcher_does_not_reuse_an_unowned_server(monkeypatch):
    from server import LocalServerLauncher
    calls = []
    launcher = LocalServerLauncher(extra_args=['--live'])
    monkeypatch.setattr(launcher, 'is_up', lambda: True)

    def spawn(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr('server.subprocess.Popen', spawn)
    assert launcher.ensure()
    assert launcher.port != 8765
    assert '--live' in calls[0]
    assert calls[0][-1] == str(launcher.port)
    assert launcher.ensure()
    assert len(calls) == 1  # only its own process can be reused
