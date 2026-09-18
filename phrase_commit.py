"""Conservative phrase finalization and application-owned commit ledger."""
from collections import deque
import re

from caption_output import CaptionOutput
from streaming import PhrasePolicy


def boundary_ready(english, japanese=""):
    """Only explicit continuation hints justify a brief wait at an audio pause."""
    source = japanese.strip().rstrip('。！？.!?」』）)\"\' ')
    # Topic/case particles, conjunctive endings and te-forms often continue.
    if source and re.search(r'(?:けれども|けれど|ですが|ますが|ので|のに|けど|て|で|は|を|に|が|と|も|へ)$', source):
        return False
    # Plain verbs, adjectives and short answers need not have punctuation or a
    # polite ending. English punctuation is also unreliable on short ASR clips.
    return bool(english.strip() or source)


class CommitPolicy(PhrasePolicy):
    """Conversational pause endpoint; one bounded continuation wait per phrase."""
    def __init__(self, threshold=0.001, pause=0.45, maximum=30.0):
        super().__init__(threshold, pause, maximum)
        self.release_silence = max(0.9, pause * 2)
        self.waited_utterance = None

    def should_wait(self, job, text, source):
        if (not text or self.forced_split or job.cached_text is not None
                or self.waited_utterance == job.utterance or boundary_ready(text, source)):
            return False
        self.waited_utterance = job.utterance
        return True

    def decode_beam(self, maximum, final):
        return min(maximum, 2)

    def next(self, audio, end):
        job = super().next(audio, end)
        if job is not None and job.audio is not None and self.last_voice is not None:
            # Keep a little trailing context, not the entire endpoint silence.
            # The consumption boundary remains job.end, so no audio is replayed.
            length = max(1, int(round((min(job.end, self.last_voice + 0.12) - job.start) * 16000)))
            job.audio = job.audio[:length]
        return job


class CommitLedger:
    def __init__(self):
        self.history = deque(maxlen=4)
        self.last_id = -1
        self.end = 0.0

    def record(self, phrase_id, start, end, text):
        if phrase_id <= self.last_id or start < self.end - 0.02 or end <= start:
            return False
        self.last_id, self.end = phrase_id, end
        self.history.append(dict(phrase_id=phrase_id, start=start, end=end, text=text[-300:]))
        return True


class CommitOutput(CaptionOutput):
    """One finalized phrase awaiting publication; never replace it with a preview."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = CommitLedger()
        self.pending_commit = None

    def history(self):
        with self.lock:
            return [dict(item) for item in self.ledger.history]

    def offer_commit(self, job, text):
        with self.lock:
            if self.expired or self.pending_commit is not None:
                return False
            self.pending_commit = (job, text)
            return True

    def busy(self):
        with self.lock:
            return self.pending_commit is not None

    def tick(self, now):
        with self.lock:
            super().tick(now)
            if self.expired:
                self.pending_commit = None
            if self.pending_commit and now - self.last_emit >= self.interval:
                job, text = self.pending_commit
                self.pending_commit = None
                if self.ledger.record(job.utterance, job.start, job.end, text):
                    self.displayed = text
                    self.last_emit = now
                    self.emit(text)


def finalization_state(job, text, source, history, forced=False):
    return dict(phrase_id=job.utterance, start=job.start, end=job.end,
                audio_paused=job.final, boundary_ready=boundary_ready(text, source),
                forced_boundary=forced, committed_history=history,
                japanese=source[-1200:], current=text)
