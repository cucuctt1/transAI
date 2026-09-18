import re
from collections import deque

_SENT_SPLIT = re.compile(r"([.!?\u3002\uff01\uff1f]+)")
_LEADING_PUNCT = " \t,;:!?.\u3001\u3002\uff01\uff1f\uff0c\uff0e\uff5e~-"
_LETTER = re.compile(r"[A-Za-z]")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _normalize(sentence):
    return " ".join(_NON_ALNUM.sub(" ", sentence.lower()).split())


def _split_sentences(text):
    parts = _SENT_SPLIT.split(text)
    sentences = []
    buf = ""
    for part in parts:
        if _SENT_SPLIT.fullmatch(part):
            buf += part
            sentences.append(buf)
            buf = ""
        else:
            buf += part
    if buf.strip():
        sentences.append(buf)
    return [s.strip().lstrip(_LEADING_PUNCT).strip() for s in sentences]


def _tokens(sentence):
    return set(sentence.split())


def _jaccard(a, b):
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class Deduplicator:
    """Emits each distinct sentence once, tolerating small re-wordings between windows.

    A sentence is emitted when it appears (with fuzzy word overlap >= `overlap`) in
    `stability` windows; near-duplicates of already-emitted sentences are suppressed.
    """

    def __init__(self, stability=2, overlap=0.6, history=60):
        self.stability = max(1, stability)
        self.overlap = overlap
        self._window_history = deque(maxlen=max(self.stability, 2))
        self._emitted = deque(maxlen=history)

    def _is_emitted(self, key):
        return any(_jaccard(key, e) >= self.overlap for e in self._emitted)

    def feed(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        sents = []
        for raw in _split_sentences(text):
            key = _normalize(raw)
            if not key or not _LETTER.search(key):
                continue
            sents.append((key, raw))

        self._window_history.append([k for k, _ in sents])

        out = []
        for key, raw in sents:
            if self._is_emitted(key):
                continue
            count = 0
            for hist in self._window_history:
                if any(_jaccard(key, h) >= self.overlap for h in hist):
                    count += 1
            if count >= self.stability:
                self._emitted.append(key)
                out.append(raw)
        return " ".join(out).strip()
