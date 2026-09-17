_SENT_END_CHARS = (".", "!", "?", "\u3002", "\uff01", "\uff1f")


class Deduplicator:
    """Removes overlap between consecutive translation windows and emits complete sentences."""

    def __init__(self):
        self._last = ""
        self._pending = ""

    def feed(self, text):
        text = (text or "").strip()
        if not text:
            return ""

        prev = self._last.split()
        cur = text.split()
        k = 0
        n = min(len(prev), len(cur))
        for kk in range(n, 0, -1):
            if prev[-kk:] == cur[:kk]:
                k = kk
                break
        self._last = text

        if k >= len(cur):
            return ""

        new_text = " ".join(cur[k:])
        full = (self._pending + " " + new_text).strip() if self._pending else new_text

        end = -1
        for ch in _SENT_END_CHARS:
            pos = full.rfind(ch)
            if pos > end:
                end = pos
        if end == -1:
            self._pending = full
            return ""

        emitted = full[: end + 1].strip()
        self._pending = full[end + 1 :].strip()
        return emitted
