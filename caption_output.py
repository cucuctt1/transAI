"""Thread-safe caption pacing, input expiry and fixer-first publication."""
import threading

import numpy as np


class CaptionOutput:
    def __init__(self, emit, interval=1.0, timeout=3.0, fixer=None, threshold=0.001):
        self.emit, self.interval, self.timeout = emit, interval, timeout
        self.fixer, self.threshold = fixer, threshold
        self.lock = threading.RLock()
        self.seen = 0.0
        self.last_voice = None
        self.expired = False
        self.version = 0
        self.waiting = self.queued = self.ready = None
        self.displayed = ""
        self.last_emit = float('-inf')

    def observe(self, audio, end, now):
        with self.lock:
            if end <= self.seen:
                return
            start = end - len(audio) / 16000
            offset = max(0, int(round((self.seen - start) * 16000)))
            self.seen = end
            last_voice = None
            for index in range(offset, len(audio), 320):
                frame = audio[index:index + 320]
                if len(frame) and float(np.mean(frame * frame)) >= self.threshold ** 2:
                    last_voice = start + (index + len(frame)) / 16000
            if last_voice is not None:
                observed = now - max(0, end - last_voice)
                self.last_voice = max(self.last_voice or observed, observed)
                if now - self.last_voice < self.timeout:
                    self.expired = False

    def offer(self, text, context=""):
        with self.lock:
            if text and not self.expired:
                self.queued = (context, text)

    def tick(self, now):
        with self.lock:
            if self.last_voice is not None and now - self.last_voice >= self.timeout:
                if not self.expired:
                    self.expired = True
                    self.version += 1
                    self.waiting = self.queued = self.ready = None
                    self.displayed = ""
                    self.emit("")  # clears UI and network clients, even while inference runs
                return
            if self.waiting is not None:
                # Keep an in-flight correction eligible when newer raw ASR arrives.
                # Silence expiry, not every preview, invalidates this generation.
                fixed = self.fixer.take(self.version, max_age=float('inf'))
                if fixed is not None or self.fixer.closed:
                    self.ready = fixed if fixed is not None else self.waiting[1]
                    self.waiting = None
            if self.ready is not None and now - self.last_emit >= self.interval:
                if self.ready != self.displayed:
                    self.displayed = self.ready
                    self.emit(self.ready)
                    self.last_emit = now
                self.ready = None
            if self.waiting is None and self.queued is not None:
                # Don't overwrite a paced correction before it can be displayed.
                if self.ready is not None:
                    return
                item, self.queued = self.queued, None
                if self.fixer is not None and not self.fixer.closed:
                    self.version += 1
                    self.waiting = item
                    self.fixer.submit(self.version, *item)
                else:
                    self.ready = item[1]
