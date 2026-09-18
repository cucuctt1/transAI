import numpy as np

from caption_output import CaptionOutput


class Fixer:
    closed = False

    def __init__(self):
        self.jobs = []
        self.results = {}

    def submit(self, version, context, text):
        self.jobs.append((version, context, text))

    def take(self, version, max_age=5):
        return self.results.pop(version, None)


def active(output, now, end=1):
    output.observe(np.full(16000, 0.1, dtype=np.float32), end, now)


def test_caption_pacing_and_clear_bypasses_interval():
    shown = []
    output = CaptionOutput(shown.append, timeout=0.8)
    active(output, 0)
    output.offer('first')
    output.tick(0)
    output.tick(0.05)
    output.offer('second')
    output.tick(0.1)
    output.tick(0.5)
    assert shown == ['first']
    output.tick(0.8)
    output.tick(2)
    assert shown == ['first', '']


def test_silence_and_stopped_stream_expire_but_active_duplicates_do_not():
    shown = []
    output = CaptionOutput(shown.append)
    active(output, 0)
    output.offer('same caption')
    output.tick(0)
    output.tick(0.05)
    active(output, 2, end=3)
    output.tick(3)
    assert shown == ['same caption']
    # Quiet samples do not renew activity; neither does polling frozen audio.
    output.observe(np.zeros(16000), 4, 3)
    output.observe(np.full(16000, 0.1), 4, 4)
    output.tick(5)
    assert shown == ['same caption', '']


def test_fixer_first_survives_newer_previews_and_bounds_pending_work():
    shown, fixer = [], Fixer()
    output = CaptionOutput(shown.append, fixer=fixer)
    active(output, 0)
    output.offer('raw one')
    output.tick(0)
    output.offer('raw two')
    output.offer('raw three')
    output.tick(0.5)
    assert shown == [] and len(fixer.jobs) == 1
    fixer.results[1] = 'fixed one'
    output.tick(1)
    assert shown == ['fixed one']
    assert fixer.jobs[-1][2] == 'raw three'
    fixer.results[2] = 'fixed three'
    output.tick(1.5)
    assert shown == ['fixed one']
    output.tick(2)
    assert shown == ['fixed one', 'fixed three']


def test_late_fix_cannot_resurrect_cleared_caption():
    shown, fixer = [], Fixer()
    output = CaptionOutput(shown.append, fixer=fixer)
    active(output, 0)
    output.offer('old raw')
    output.tick(0)
    old_version = fixer.jobs[0][0]
    output.tick(3)
    fixer.results[old_version] = 'late fix'
    output.tick(4)
    active(output, 5, end=2)
    output.offer('new raw')
    output.tick(5)
    assert shown == ['']
    new_version = fixer.jobs[-1][0]
    assert new_version != old_version
    fixer.results[new_version] = 'new fix'
    output.tick(5.1)
    assert shown == ['', 'new fix']


def test_unchanged_fixer_result_and_failure_fallback_are_displayed():
    shown, fixer = [], Fixer()
    output = CaptionOutput(shown.append, fixer=fixer)
    active(output, 0)
    output.offer('unchanged')
    output.tick(0)
    fixer.results[1] = 'unchanged'
    output.tick(0.1)
    assert shown == ['unchanged']
    output.offer('fallback')
    output.tick(0.2)
    fixer.closed = True
    output.tick(1.1)
    assert shown == ['unchanged', 'fallback']


def test_defaults_and_timeout_argument(monkeypatch):
    from main import parse_args, pipeline_config
    monkeypatch.setattr('sys.argv', ['main.py'])
    cfg = pipeline_config(parse_args())
    assert cfg.whisper_step == 2 and cfg.caption_timeout == 5
    assert cfg.whisper_window == 10 and cfg.audio_buffer == 20
    monkeypatch.setattr('sys.argv', ['main.py', '--caption-timeout', '8', '--step', '2'])
    cfg = pipeline_config(parse_args())
    assert cfg.caption_timeout == 8 and cfg.whisper_step == 2


def test_default_live_policy_waits_two_seconds_between_previews():
    from transcript import Config
    from streaming import StreamPolicy
    cfg = Config()
    policy = StreamPolicy(cfg.whisper_window, cfg.whisper_step)
    assert policy.next(np.full(16000, .1), 1) is None
    assert policy.next(np.full(32000, .1), 2) is not None
    assert policy.next(np.full(48000, .1), 3) is None
    assert policy.next(np.full(64000, .1), 4) is not None


def test_clear_remains_responsive_during_blocked_inference():
    import threading
    from types import SimpleNamespace
    from asr_engine import ASREngine
    from streaming import AudioRing
    from transcript import Config

    entered, release, shown, cleared = [threading.Event() for _ in range(4)]

    class Source(AudioRing):
        def start(self):
            self.feed(np.full(16000, 0.1, dtype=np.float32))

        def stop(self):
            pass

    class Model:
        beam_size = 1
        calls = 0

        def translate(self, audio, beam_size=None):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                assert release.wait(4)
            return SimpleNamespace(text='caption')

    def emit(text):
        (shown if text else cleared).set()

    source = Source(12)
    engine = ASREngine(Config(whisper_step=1, caption_timeout=0.7), Model(), 'fake', source, caption_cb=emit)
    engine.start()
    try:
        assert shown.wait(2)
        source.feed(np.full(16000, 0.1, dtype=np.float32))
        assert entered.wait(2)
        assert cleared.wait(2)
        assert not release.is_set()
    finally:
        release.set()
        engine.stop()
