"""Cooperative process shutdown, including repeated Ctrl+C during cleanup."""
import os
import signal
import sys
import threading
from contextlib import contextmanager

requested = threading.Event()


@contextmanager
def signals():
    previous = {}
    requested.clear()
    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        sig = getattr(signal, name, None)
        if sig is not None:
            previous[sig] = signal.signal(sig, lambda *_: requested.set())
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def watch_parent():
    """Private inherited pipe: EOF or STOP means the owning client is gone."""
    def read():
        try:
            # Avoid a buffered stdin lock held at interpreter shutdown.
            os.read(sys.stdin.fileno(), 64)
        finally:
            requested.set()
    threading.Thread(target=read, name='parent-control', daemon=True).start()
