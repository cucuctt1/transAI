"""Bounded audio storage and pure policies for live translation (no GPU/Qt)."""

import threading
import re
from collections import deque
from dataclasses import dataclass

import numpy as np


class AudioRing:
    """Sample-count bounded ring: snapshots copy only the requested tail."""

    def __init__(self, seconds, rate=16000):
        self.rate = rate
        self.capacity = max(1, int(seconds * rate))
        self.data = np.zeros(self.capacity, dtype=np.float32)
        self.total = 0
        self.lock = threading.Lock()

    def feed(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self.lock:
            end = self.total + len(samples)
            samples = samples[-self.capacity:]
            start = (end - len(samples)) % self.capacity
            first = min(len(samples), self.capacity - start)
            self.data[start:start + first] = samples[:first]
            self.data[:len(samples) - first] = samples[first:]
            self.total = end

    def available_seconds(self):
        with self.lock:
            return min(self.total, self.capacity) / self.rate

    def live_position(self):
        with self.lock:
            return self.total / self.rate

    def latest(self, seconds):
        n = max(1, int(seconds * self.rate))
        with self.lock:
            if n > min(self.total, self.capacity):
                return None
            start = (self.total - n) % self.capacity
            first = min(n, self.capacity - start)
            out = np.empty(n, dtype=np.float32)
            out[:first] = self.data[start:start + first]
            out[first:] = self.data[:n - first]
            return out, self.total / self.rate


@dataclass
class DecodeJob:
    audio: object
    start: float
    end: float
    final: bool
    utterance: int
    cached_text: object = None


class StreamPolicy:
    """Pull freshest audio when idle; inspect only previously unseen samples.

    A pause triggers one final decode, then silence costs no model calls.
    Frame energy is just a cheap activity gate; decoder VAD still handles noise.
    """

    def __init__(self, window=8.0, step=1.0, threshold=0.001, pause=1.0):
        self.window = window
        self.step = step
        self.threshold = threshold
        self.pause = pause
        self.seen = 0.0
        self.last_decode = 0.0
        self.last_voice = None
        self.start = None
        self.utterance = 0
        self.inference_ema = 0.0

    def record_duration(self, seconds):
        self.inference_ema = (0.8 * self.inference_ema + 0.2 * seconds
                              if self.inference_ema else seconds)

    def next(self, audio, end):
        if end <= self.seen + 1e-6:
            return None
        rate = 16000
        audio_start = end - len(audio) / rate
        offset = max(0, int(round((self.seen - audio_start) * rate)))
        fresh = audio[offset:]
        # 20 ms energy frames retain short words even in a long silence window.
        first_voice = last_voice = None
        for i in range(0, len(fresh), 320):
            frame = fresh[i:i + 320]
            if len(frame) and float(np.mean(frame * frame)) >= self.threshold ** 2:
                first_voice = i if first_voice is None else first_voice
                last_voice = i + len(frame)
        self.seen = end
        if first_voice is not None:
            if self.start is None:
                self.utterance += 1
                self.start = max(audio_start, audio_start + (offset + first_voice) / rate - 0.2)
            self.last_voice = audio_start + (offset + last_voice) / rate
        if self.start is None:
            return None
        final = end - self.last_voice >= self.pause - 1e-6
        if not final and first_voice is None:
            return None  # wait for the final pass instead of decoding silence tails
        # Inference already serializes work. Adding an EMA-derived delay here
        # made a slow decoder wait *again* after it had finished.
        if not final and (end - self.start < min(self.step, self.window)
                          or end - self.last_decode < self.step - 1e-6):
            return None
        # Preserve unread audio plus overlap when a decode took longer than
        # the normal window. The caller supplies bounded extra history.
        boundary = min(end - self.window, self.last_decode - 1.5)
        start = max(audio_start, self.start, boundary)
        chunk = audio[max(0, int(round((start - audio_start) * rate))):]
        self.last_decode = end
        job = DecodeJob(chunk, start, end, final, self.utterance)
        if final:
            self.start = None
            self.last_voice = None
        return job

    def snapshot_seconds(self, live, available):
        """Recover missed audio within the source's bounded history."""
        return min(available, max(self.window, live - self.last_decode + 1.5))

    def decode_beam(self, maximum, final):
        # Live mode must not turn every pause into a costly full-phrase pass.
        return 1


def caption_key(text):
    # Preserve negation, numbers and word order; never use bag-of-words similarity.
    return tuple(w.strip('.,!?;:').casefold() for w in text.split())


class PhrasePolicy(StreamPolicy):
    """Complete phrases using both audio pauses and translated punctuation.

    Stop scanning at the first boundary, even if several phrases arrived while
    inference was busy. `consumed` logically cuts completed audio out of future
    jobs while leaving newer audio in the capture ring untouched. An incomplete
    translation retains the phrase start for the next probe after more speech.
    """

    def __init__(self, threshold=0.001, pause=1.0, maximum=30.0):
        super().__init__(threshold=threshold, pause=pause)
        self.maximum = maximum
        self.consumed = 0.0
        self.forced_split = False
        self._pending = None
        self._checked_voice = -1.0
        self._next_probe = 0.0
        self._held_text = None
        self.release_silence = max(2.0, self.pause * 2)

    @staticmethod
    def sentence_complete(text):
        """Only terminal punctuation counts; internal periods do not end audio."""
        text = text.strip().rstrip('\"\'”’)]}').rstrip()
        if not text or text.endswith(('...', '…')):
            return False
        if text.endswith(('!', '?', '。', '！', '？')):
            return True
        if not text.endswith('.'):
            return False
        last = text.split()[-1]
        if last.casefold() in {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'e.g.', 'i.e.', 'vs.'}:
            return False
        return re.fullmatch(r'(?:[A-Za-z]\.)+', last) is None

    def accept_result(self, job, text, complete=None):
        """Commit the audio boundary only after the translation also ends."""
        if job is not self._pending:
            return False
        self._pending = None
        ready = self.sentence_complete(text) if complete is None else complete
        if not (job.cached_text is not None or self.forced_split or ready):
            self._held_text = text
            return False
        self.consumed = job.end
        self.start = self.last_voice = None
        self._held_text = None
        return bool(text.strip())

    def snapshot_seconds(self, live, available):
        # Keep the current phrase, but don't copy the entire history while idle.
        begin = self.start if self.start is not None else max(self.consumed, self.seen - 0.2)
        return min(available, max(0.1, live - begin))

    def decode_beam(self, maximum, final):
        return min(maximum, 2) if self.inference_ema > self.pause else maximum

    def next(self, audio, end):
        if self._pending is not None or end <= self.seen + 1e-6:
            return None
        self.forced_split = False
        rate = 16000
        audio_start = end - len(audio) / rate
        offset = max(0, int(round((self.seen - audio_start) * rate)))
        for i in range(offset, len(audio), 320):
            frame = audio[i:i + 320]
            frame_start = audio_start + i / rate
            frame_end = frame_start + len(frame) / rate
            voiced = float(np.mean(frame * frame)) >= self.threshold ** 2
            if voiced:
                self._held_text = None
                if self.start is None:
                    self.utterance += 1
                    self.start = max(audio_start, self.consumed, frame_start - 0.2)
                self.last_voice = frame_end
            self.seen = frame_end
            if self.start is None:
                continue
            complete = frame_end - self.last_voice >= self.pause - 1e-6
            limit = frame_end - self.start >= self.maximum - 1e-6
            # Punctuation is evidence, not permission to wait indefinitely.
            # A longer pause releases the existing result without another decode.
            if (self._held_text is not None
                    and frame_end - self.last_voice >= self.release_silence - 1e-6):
                job = DecodeJob(None, max(audio_start, self.start), frame_end,
                                True, self.utterance, self._held_text)
                self._pending = job
                self.last_decode = frame_end
                return job
            # A pause starts a private translation probe, not a committed cut.
            # Don't repeatedly translate unchanged audio during the same pause.
            probe = complete and self.last_voice > self._checked_voice + 1e-6
            if not (probe or limit) or frame_end < self._next_probe - 1e-6:
                continue
            start = max(audio_start, self.start)
            first = max(0, int(round((start - audio_start) * rate)))
            chunk = audio[first:i + len(frame)].copy()
            job = DecodeJob(chunk, start, frame_end, True, self.utterance)
            self.last_decode = frame_end
            self._checked_voice = self.last_voice
            self._next_probe = frame_end + 0.5
            self._pending = job
            # The duration limit is an explicit fallback even during silence.
            self.forced_split = limit
            return job
        return None


class CaptionStabilizer:
    """Coalesce overlap and oscillation without freezing semantic corrections."""

    def __init__(self, max_words=32):
        self.max_words = max_words
        self.text = ""
        self.key = ()
        self.pending = ()
        self.history = deque(maxlen=8)
        self.utterance = None
        self.end = -1.0
        self.duplicates = 0

    def update(self, text, job):
        if job.end <= self.end:
            return None
        self.end = job.end
        if job.utterance != self.utterance:
            self.key = ()
            self.history.clear()
            self.pending = ()
            self.utterance = job.utterance
        words = text.split()
        text = ' '.join(words if self.max_words is None else words[-self.max_words:])
        key = caption_key(text)
        if not key:
            return None
        if key == self.key:
            self.duplicates += 1
            self.pending = ()
            return None
        # A smaller sliding window is not new content. Require confirmation for
        # a previously displayed alternative; final (wider-beam) results win.
        contained = any(self.key[i:i + len(key)] == key
                        for i in range(len(self.key) - len(key) + 1))
        repeated = any(k == key and end > job.start for k, end in self.history)
        if not job.final and (contained or repeated) and self.pending != key:
            self.pending = key
            self.duplicates += 1
            return None
        self.pending = ()
        self.text, self.key = text, key
        self.history.append((key, job.end))
        return text
