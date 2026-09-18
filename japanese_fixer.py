"""Conservative source-text checks and time-aware committed overlap removal."""
from collections import deque
from difflib import SequenceMatcher
import re
import unicodedata


def normalize_japanese(text):
    return unicodedata.normalize('NFC', text).strip()


def validate_japanese_edit(original, candidate):
    if not isinstance(candidate, str):
        return original
    candidate = normalize_japanese(candidate)
    if not candidate or any(s in candidate for s in ('<|', '```', '{', '}')):
        return original
    numbers = lambda s: re.findall(r'[0-9０-９〇零一二三四五六七八九十百千万億]+', s)
    negatives = lambda s: re.findall(r'ません|なかった|ない|なく|なければ', s)
    if numbers(original) != numbers(candidate) or negatives(original) != negatives(candidate):
        return original
    if len(candidate) > len(original) * 1.15 + 2:
        return original
    # No new content characters/names supplied by history. Particle/punctuation
    # repairs are allowed, but uncertain spelling changes fall back unchanged.
    content = lambda s: {c for c in s if c.isalnum() and c not in 'はがをにでとへもかの'}
    if content(candidate) - content(original):
        return original
    if SequenceMatcher(None, original, candidate, autojunk=False).ratio() < .8:
        return original
    return candidate


class JapaneseContext:
    def __init__(self):
        self.history = deque(maxlen=4)

    def prepare(self, text, job):
        text = normalize_japanese(text)
        if not self.history:
            return text
        last = self.history[-1]
        if job.start >= last['end'] - .02:
            return text  # separate audio: identical wording can be intentional
        if job.end <= last['end']:
            return ''
        previous = last['text']
        # Exact character alignment only; never strip an uncommitted live draft.
        for size in range(min(len(previous), len(text) - 1), 3, -1):
            if previous[-size:] == text[:size]:
                return text[size:]
        return text

    def commit(self, job, source):
        self.history.append(dict(phrase_id=job.utterance, start=job.start,
                                 end=job.end, text=source[-300:]))
