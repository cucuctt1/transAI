"""Real-time transcript stabilizer.

Maintains a FINALIZED (immutable) + UNSTABLE (mutable tail) transcript, merges new
Whisper hypotheses with a fast deterministic word-level aligner, and provides a
background fixer interface for optional semantic correction.

Alignment is a few milliseconds of pure token work; it never blocks inference and
never performs I/O.
"""

import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration (single source of truth for timing/limits)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    whisper_window: float = 5.0
    whisper_step: float = 1.0
    audio_buffer: float = 7.0
    rollback_window: float = 3.0
    fixer_interval: float = 0.5
    fixer_context: float = 7.0
    max_whisper_queue: int = 1
    max_whisper_pending: int = 1
    max_fixer_queue: int = 1
    enable_background_fixer: bool = False
    enable_latency_monitor: bool = True
    warning_latency: float = 1.0
    backlog_latency: float = 2.0
    words_per_second: float = 3.0  # rough rate used to bound the unstable region


# ---------------------------------------------------------------------------
# Tokenization (word-level, deterministic, CPU-friendly)
# ---------------------------------------------------------------------------

_STRIP = " \t,;:!?.\u3001\u3002\uff01\uff1f\uff0c\uff0e\uff5e~-()[]\"'"
_LEADING = _STRIP
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_SENT_END = re.compile(r"[.!?\u3002\uff01\uff1f]")
_SENT_SPLIT = re.compile(r"[.!?\u3002\uff01\uff1f]+")


def _last_sentence(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    parts = [p for p in parts if p]
    return parts[-1] if parts else ""


def tokenize(text):
    """Split into raw word tokens, keeping trailing punctuation for display."""
    out = []
    for w in (text or "").split():
        w = w.lstrip(_LEADING)
        if w:
            out.append(w)
    return out


def norm(word):
    """Lowercase and strip non-alphanumerics for comparison."""
    return _NON_ALNUM.sub("", word.lower())


# ---------------------------------------------------------------------------
# Fast deterministic aligner
# ---------------------------------------------------------------------------

def _longest_suffix_prefix(a, b):
    """Longest k such that normalized(a[-k:]) == normalized(b[:k])."""
    an = [norm(w) for w in a]
    bn = [norm(w) for w in b]
    maxk = min(len(an), len(bn))
    for k in range(maxk, 0, -1):
        if an[-k:] == bn[:k]:
            return k
    return 0


def _is_subsequence(sub, full):
    si = 0
    for w in full:
        if si < len(sub) and norm(sub[si]) == norm(w):
            si += 1
    return si == len(sub)


def _end_index_of_subsequence(sub, full):
    """Index in `full` just past the earliest subsequence match of `sub` (0 if none)."""
    si = 0
    end = 0
    for i, w in enumerate(full):
        if si < len(sub) and norm(sub[si]) == norm(w):
            si += 1
            end = i + 1
            if si == len(sub):
                return end
    return end if si == len(sub) else 0


def _longest_prefix_subsequence(sub, full):
    """Length of the longest prefix of `sub` that is a subsequence of `full`."""
    si = 0
    for w in full:
        if si < len(sub) and norm(sub[si]) == norm(w):
            si += 1
    return si


def _jaccard(a_tokens, b_tokens):
    """Token-set similarity (normalized), 0..1."""
    sa = {norm(w) for w in a_tokens}
    sb = {norm(w) for w in b_tokens}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_NEAR_DUPLICATE = 0.7


def align(finalized_tokens, unstable_tokens, new_tokens):
    """Merge a new hypothesis into the unstable tail.

    Returns ``(new_unstable_tokens, confidence)``. ``finalized_tokens`` is never
    modified. ``confidence`` is ``"high"`` for exact/safe merges and ``"low"`` for
    drift/reinterpretation (the ambiguous case the background fixer may re-check).
    """
    combined = finalized_tokens + unstable_tokens

    if not new_tokens:
        return list(unstable_tokens), "high"
    if not combined:
        return list(new_tokens), "high"

    # 1) Exact sliding overlap: old ends where new begins -> append continuation.
    k = _longest_suffix_prefix(combined, new_tokens)
    if k > 0:
        return unstable_tokens + new_tokens[k:], "high"

    # 2) New adds nothing (already contained in current transcript): keep old.
    if _is_subsequence(new_tokens, combined):
        return list(unstable_tokens), "high"

    # 3) Word drift / reinterpretation: keep the finalized prefix new still agrees
    #    with, and replace the rest with the newer hypothesis.
    fc = _longest_prefix_subsequence(finalized_tokens, new_tokens)
    m = _end_index_of_subsequence(finalized_tokens[:fc], new_tokens)
    candidate = new_tokens[m:]

    # 3a) Near-duplicate hypothesis: suppress rather than replace + reprint.
    if _jaccard(candidate, unstable_tokens) >= _NEAR_DUPLICATE:
        return list(unstable_tokens), "high"

    return candidate, "low"


# ---------------------------------------------------------------------------
# Transcript state (FINALIZED + UNSTABLE, versioned, thread-safe)
# ---------------------------------------------------------------------------

class TranscriptState:
    def __init__(self, rollback_window=3.0, words_per_second=3.0):
        self.rollback_window = rollback_window
        self.words_per_second = words_per_second
        self._lock = threading.Lock()
        self._finalized = []      # raw tokens, immutable once committed
        self._unstable = []       # raw tokens, may be rewritten
        self._unstable_since = None
        self._needs_fix = False
        self.noop_count = 0
        self.version = 0

    def _finalize(self):
        now = time.monotonic()

        # age-based: a stable tail older than the rollback window becomes finalized
        if self._unstable and self._unstable_since is not None:
            if now - self._unstable_since >= self.rollback_window:
                self._finalized.extend(self._unstable)
                self._unstable = []
                self._unstable_since = None
                return

        # overflow cap: keep only ~rollback_window worth of words mutable
        cap = max(1, int(self.rollback_window * self.words_per_second))
        while len(self._unstable) > cap:
            moved = False
            for i, tok in enumerate(self._unstable):
                if _SENT_END.search(tok):
                    self._finalized.extend(self._unstable[: i + 1])
                    del self._unstable[: i + 1]
                    moved = True
                    break
            if not moved:
                self._finalized.append(self._unstable.pop(0))

    def snapshot(self):
        with self._lock:
            return (
                self.version,
                " ".join(self._finalized),
                " ".join(self._unstable),
                self._needs_fix,
            )

    def apply_hypothesis(self, new_text):
        """Fast align path: merge a Whisper hypothesis (few ms)."""
        new_tokens = tokenize(new_text)
        with self._lock:
            self._finalize()
            before = self._unstable
            self._unstable, confidence = align(self._finalized, self._unstable, new_tokens)
            self._unstable_since = time.monotonic()
            self._needs_fix = confidence == "low"
            if self._unstable == before:
                self.noop_count += 1  # duplicate/overlapping hypothesis suppressed
            self.version += 1
            self._finalize()
            return (
                self.version,
                " ".join(self._finalized),
                " ".join(self._unstable),
                self._needs_fix,
            )

    def apply_fix(self, version, new_unstable_tokens):
        """Apply a background-fixer result only if the transcript is still at `version`."""
        with self._lock:
            if version != self.version:
                return False  # stale
            self._unstable = list(new_unstable_tokens)
            self._unstable_since = time.monotonic()
            self.version += 1
            self._finalize()
            return True

    def latest_caption(self):
        """Current phrase for the overlay: the last sentence of the current transcript."""
        with self._lock:
            text = " ".join(self._unstable) if self._unstable else " ".join(self._finalized)
            return _last_sentence(text)


# ---------------------------------------------------------------------------
# Background fixer (clean interface; no LLM bundled)
# ---------------------------------------------------------------------------

class BackgroundFixer:
    """Correction layer for the unstable tail.

    No LLM is bundled. Implement ``fix_unstable_transcript`` with an LLM later;
    it must only dedupe / resolve drift / repair continuity -- never invent,
    paraphrase, or translate. The default is a deterministic no-op.
    """

    def __init__(self, interval=0.5, context_seconds=7.0):
        self.interval = interval
        self.context_seconds = context_seconds

    def fix_unstable_transcript(self, context, unstable_text):
        return unstable_text


# ---------------------------------------------------------------------------
# Output renderer (redraws the unstable tail; append-only when not a TTY)
# ---------------------------------------------------------------------------

class TranscriptRenderer:
    def __init__(self, stream=None, timestamps=True):
        self._stream = stream or sys.stdout
        self._timestamps = timestamps
        self._redraw = bool(getattr(self._stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._printed_finalized = ""
        self._transient = ""
        self._prev_len = 0

    def _fmt(self, seconds):
        s = int(seconds)
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def render(self, finalized, unstable, elapsed):
        with self._lock:
            if self._redraw:
                self._render_redraw(finalized, unstable, elapsed)
            else:
                self._render_append(finalized, unstable, elapsed)
            self._stream.flush()

    def _render_append(self, finalized, unstable, elapsed):
        out = self._stream
        shown = self._printed_finalized
        if self._transient and finalized.endswith(self._transient):
            # the previously-shown transient has been promoted into finalized text
            head = finalized[: len(finalized) - len(self._transient)]
            new_final = head[len(shown):].strip()
            self._transient = ""
        else:
            new_final = finalized[len(shown):].strip()
        self._printed_finalized = finalized
        if new_final:
            prefix = f"[{self._fmt(elapsed)}] " if self._timestamps else ""
            out.write(prefix + new_final + "\n")
        if unstable and unstable != self._transient:
            out.write("   > " + unstable + "\n")
            self._transient = unstable
        elif not unstable:
            self._transient = ""

    def _render_redraw(self, finalized, unstable, elapsed):
        out = self._stream

        # Commit a transient line that has now moved into finalized text.
        if self._transient and finalized.endswith(self._transient):
            out.write("\n")
            self._printed_finalized = finalized
            self._transient = ""
            self._prev_len = 0

        new_final = finalized[len(self._printed_finalized):].strip()
        if new_final:
            prefix = f"[{self._fmt(elapsed)}] " if self._timestamps else ""
            out.write(prefix + new_final + "\n")
            self._printed_finalized = finalized
            self._prev_len = 0

        if unstable:
            text = "  ... " + unstable
            pad = max(0, self._prev_len - len(text))
            out.write("\r" + text + " " * pad)
            self._prev_len = len(text)
            self._transient = unstable
        elif self._transient:
            out.write("\r" + " " * self._prev_len + "\r")
            self._prev_len = 0
            self._transient = ""


# ---------------------------------------------------------------------------
# Latest-job-wins queue (bounded; newest snapshot replaces stale ones)
# ---------------------------------------------------------------------------

class LatestQueue:
    def __init__(self, maxsize=1):
        self.maxsize = maxsize
        self._cond = threading.Condition()
        self._items = deque()

    def put(self, item):
        with self._cond:
            if len(self._items) >= self.maxsize:
                self._items.popleft()  # drop oldest -> latest wins
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout=None):
        with self._cond:
            while not self._items:
                if timeout is not None:
                    self._cond.wait(timeout)
                    if not self._items:
                        return None
                else:
                    self._cond.wait()
            return self._items.popleft()


# ---------------------------------------------------------------------------
# Snapshot scheduling: latest-job-wins (at most 1 pending, stale ones dropped)
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    audio: object      # numpy float32 mono 16k array
    audio_end: float   # audio-time position (seconds) of this window's end
    id: int = 0


class LatestWinsScheduler:
    """Holds at most one pending snapshot; newer snapshots replace older ones.

    The `skipped` counter increments every time an older pending snapshot is
    discarded in favor of a newer one. This guarantees the pending backlog can
    never grow beyond 1.
    """

    def __init__(self, max_pending=1):
        self.max_pending = max_pending
        self._lock = threading.Lock()
        self._pending = None
        self.skipped = 0

    def submit(self, snapshot):
        with self._lock:
            if self._pending is not None:
                self.skipped += 1  # drop the stale pending snapshot
            self._pending = snapshot

    def take(self):
        with self._lock:
            snapshot = self._pending
            self._pending = None
            return snapshot

    @property
    def pending_count(self):
        with self._lock:
            return 1 if self._pending is not None else 0


# ---------------------------------------------------------------------------
# Latency monitoring
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """Tracks live audio position vs. latest transcribed audio position."""

    def __init__(self, warning=1.0, backlog=2.0):
        self.warning = warning
        self.backlog = backlog
        self._lock = threading.RLock()
        self.live_position = 0.0
        self.transcribed_to = 0.0
        self._whisper_durations = deque(maxlen=100)
        self._completions = deque(maxlen=100)
        self.skipped = 0
        self.pending = 0
        self.duplicates = 0

    def update(self, live=None, transcribed=None, whisper_dur=None, skipped=None, pending=None, duplicates=None):
        with self._lock:
            if live is not None:
                self.live_position = live
            if transcribed is not None:
                self.transcribed_to = max(self.transcribed_to, transcribed)
            if whisper_dur is not None:
                self._whisper_durations.append(whisper_dur)
                self._completions.append(time.monotonic())
            if skipped is not None:
                self.skipped = skipped
            if pending is not None:
                self.pending = pending
            if duplicates is not None:
                self.duplicates = duplicates

    def latency(self):
        with self._lock:
            return self.live_position - self.transcribed_to

    def avg_whisper(self):
        with self._lock:
            if not self._whisper_durations:
                return 0.0
            return sum(self._whisper_durations) / len(self._whisper_durations)

    def avg_update_interval(self):
        with self._lock:
            if len(self._completions) < 2:
                return 0.0
            spans = [b - a for a, b in zip(self._completions, list(self._completions)[1:])]
            return sum(spans) / len(spans)

    def state(self):
        lat = self.latency()
        if lat < self.warning:
            return "REALTIME"
        if lat < self.backlog:
            return "WARNING"
        return "CATCHING_UP"

    def status_line(self):
        with self._lock:
            lat = self.live_position - self.transcribed_to
            return (
                f"[ASR] live_audio={self.live_position:.1f}s transcribed_to={self.transcribed_to:.1f}s "
                f"latency={lat:.2f}s whisper={self.avg_whisper():.2f}s "
                f"interval={self.avg_update_interval():.2f}s "
                f"pending={self.pending} skipped={self.skipped} dup={self.duplicates} state={self.state()}"
            )
