import math
import threading

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly
from streaming import AudioRing


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
        self._stop = threading.Event()
        self._thread = None
        self.device_name = ""
        self._ring = AudioRing(buffer_seconds)

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
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        try:
            self._open_device()
        except Exception:
            self._close_device()
            raise
        self._thread = threading.Thread(
            target=self._run, name="audio-capture", daemon=False
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
                # Resample each incoming block once, instead of resampling the
                # whole overlapping window on every inference.
                block = self._resample(block)
                self._ring.feed(block)
                if self._consumer is not None:
                    self._consumer(block)
        except Exception as exc:
            if not self._stop.is_set():
                print(f"[Audio] capture error: {exc}")
        finally:
            # Only the reader releases native resources after its read returns.
            self._close_device()

    def available_seconds(self):
        return self._ring.available_seconds()

    def live_position(self):
        """Monotonic audio time (seconds) captured so far."""
        return self._ring.live_position()

    def latest(self, seconds):
        """Return (audio_16k_mono, audio_end_seconds) for the newest `seconds`, or None."""
        return self._ring.latest(seconds)

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def _close_device(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            except Exception:
                pass
        if self._pa is not None:
            try:
                self._pa.terminate()
                self._pa = None
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
