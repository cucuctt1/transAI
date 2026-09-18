import collections
import math
import threading

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly


class AudioCapture:
    """Captures Windows system audio (WASAPI loopback) into a rolling mono 16 kHz buffer."""

    TARGET_RATE = 16000

    def __init__(self, buffer_seconds=5.0, block_seconds=0.2, device_index=None):
        self.buffer_seconds = buffer_seconds
        self.block_seconds = block_seconds
        self.device_index = device_index
        self.target_rate = self.TARGET_RATE
        self.native_rate = None
        self.channels = None
        self._pa = None
        self._stream = None
        self._consumer = None
        self._buf = collections.deque(
            maxlen=math.ceil(buffer_seconds / block_seconds) + 2
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.device_name = ""
        self._captured_frames = 0

    def set_consumer(self, callback):
        """Stream each resampled 16 kHz mono block to `callback` (e.g. a socket)."""
        self._consumer = callback

    def _open_device(self):
        self._pa = pyaudio.PyAudio()
        infos = list(self._pa.get_loopback_device_info_generator())
        if self.device_index is not None and 0 <= self.device_index < len(infos):
            info = infos[self.device_index]
        else:
            info = self._pa.get_default_wasapi_loopback()
        if info is None:
            print("[Audio] Could not detect a default output device.")
            for i, d in enumerate(infos):
                print(f"  [{i}] {d['name']}")
            sel = input("Select device index: ").strip()
            info = infos[int(sel)]
        self.device_name = info["name"]
        self.native_rate = int(info["defaultSampleRate"])
        self.channels = int(info["maxInputChannels"])
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.native_rate,
            input=True,
            input_device_index=info["index"],
            frames_per_buffer=int(self.native_rate * self.block_seconds),
        )

    def _resample(self, audio):
        if self.native_rate == self.target_rate:
            return audio.astype(np.float32, copy=False)
        g = math.gcd(self.target_rate, self.native_rate)
        up = self.target_rate // g
        down = self.native_rate // g
        return resample_poly(audio, up, down).astype(np.float32)

    def start(self):
        self._open_device()
        self._thread = threading.Thread(
            target=self._run, name="audio-capture", daemon=True
        )
        self._thread.start()

    def _run(self):
        block_frames = int(self.native_rate * self.block_seconds)
        try:
            while not self._stop.is_set():
                data = self._stream.read(block_frames, exception_on_overflow=False)
                block = np.frombuffer(data, dtype=np.float32)
                if self.channels > 1:
                    block = block.reshape(-1, self.channels).mean(axis=1)
                with self._lock:
                    self._buf.append(block.astype(np.float32, copy=False))
                    self._captured_frames += len(block)
                if self._consumer is not None:
                    self._consumer(self._resample(block.astype(np.float32)))
        except Exception as exc:
            print(f"[Audio] capture error: {exc}")

    def available_seconds(self):
        with self._lock:
            total = sum(len(b) for b in self._buf)
        return total / self.native_rate if self.native_rate else 0.0

    def live_position(self):
        """Monotonic audio time (seconds) captured so far."""
        with self._lock:
            return self._captured_frames / self.native_rate if self.native_rate else 0.0

    def latest(self, seconds):
        """Return (audio_16k_mono, audio_end_seconds) for the newest `seconds`, or None."""
        if self.native_rate is None:
            return None
        n = int(seconds * self.native_rate)
        with self._lock:
            blocks = list(self._buf)
            end = self._captured_frames / self.native_rate
        if not blocks:
            return None
        audio = np.concatenate(blocks)
        if len(audio) < n:
            return None
        return self._resample(audio[-n:]), end

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass


def list_loopback_devices():
    """Return a list of {"index": int, "name": str} for the WASAPI loopback devices."""
    pa = pyaudio.PyAudio()
    try:
        return [
            {"index": i, "name": info["name"]}
            for i, info in enumerate(pa.get_loopback_device_info_generator())
        ]
    finally:
        pa.terminate()
