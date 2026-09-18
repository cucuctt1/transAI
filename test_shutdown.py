"""Shutdown regressions, using real sockets/Qt/processes but no model download."""
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from asr_engine import ASREngine
from server import AudioCaptionServer, LocalServerLauncher, NetworkAudioSource, StreamingClient
from transcript import Config

ROOT = str(Path(__file__).resolve().parent)


def test_network_shutdown_including_unfinished_handshake():
    server = AudioCaptionServer(NetworkAudioSource(), host='127.0.0.1', port=0)
    server.start()
    port = server._sock.getsockname()[1]
    idle = socket.create_connection(('127.0.0.1', port))
    client = StreamingClient('127.0.0.1', port)
    client.start()
    try:
        deadline = time.monotonic() + 3
        while not client.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.connected
        server.stop()
        client.stop()
        assert not server._thread.is_alive()
        assert not server._handlers
        assert not client._thread.is_alive()
        assert not client._stats_thread.is_alive()
        # Idempotent repeat close.
        server.stop()
        client.stop()
    finally:
        idle.close()
        client.stop()
        server.stop()


def test_engine_close_waits_for_native_inference_and_suppresses_late_output():
    import numpy as np
    entered, release = threading.Event(), threading.Event()
    stopped = []
    emitted = []

    class Source:
        def start(self):
            pass
        def stop(self):
            stopped.append(True)
        def available_seconds(self):
            return 1
        def live_position(self):
            return 1
        def latest(self, seconds):
            return np.full(16000, .1, dtype=np.float32), 1

    class Model:
        beam_size = 1
        def translate(self, *args, **kwargs):
            entered.set()
            assert release.wait(4)
            return SimpleNamespace(text='Must not display during shutdown.')

    engine = ASREngine(Config(whisper_step=1), Model(), 'fake', Source(), caption_cb=emitted.append)
    engine.start()
    assert entered.wait(2)
    closer = threading.Thread(target=engine.stop)
    closer.start()
    try:
        assert engine._stop.wait(1)
        assert closer.is_alive() and not stopped
    finally:
        release.set()
        closer.join(3)
    assert not closer.is_alive() and not engine.is_running()
    assert stopped and not emitted


def test_owned_server_receives_cooperative_stop_and_parent_eof():
    code = 'import shutdown; shutdown.watch_parent(); assert shutdown.requested.wait(5)'
    for explicit in (True, False):
        proc = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, stdin=subprocess.PIPE)
        launcher = LocalServerLauncher()
        launcher.proc = proc
        try:
            if explicit:
                launcher.stop()
            else:
                proc.stdin.close()  # same EOF as an unexpectedly exited parent
                proc.wait(timeout=7)
            assert proc.returncode == 0
        finally:
            if proc.poll() is None:
                proc.kill()  # only this test's child, if its test deadline expires
                proc.wait()


def test_main_window_x_exits_qt_instead_of_leaving_hidden_app():
    code = (
        'from PyQt5.QtWidgets import QApplication; '
        'from PyQt5.QtCore import QTimer; '
        'from ui import App; '
        'app=QApplication([]); app.setQuitOnLastWindowClosed(False); '
        'controller=App(mode="server"); controller.main_window.show(); '
        'QTimer.singleShot(0, controller.main_window.close); '
        'QTimer.singleShot(4000, lambda: app.exit(9)); '
        'result=app.exec_(); controller._shutdown_thread.join(); '
        'assert result == 0 and controller._cleaned'
    )
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_audio_device_released_only_after_reader_finishes():
    from audio_capture import AudioCapture
    entered, release = threading.Event(), threading.Event()
    calls = []
    capture = AudioCapture()
    capture.native_rate, capture.channels = 16000, 1

    def read(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return b'\0' * 12800

    capture._stream = SimpleNamespace(read=read, stop_stream=lambda: calls.append('stop'),
                                      close=lambda: calls.append('close'))
    capture._pa = SimpleNamespace(terminate=lambda: calls.append('terminate'))
    capture._thread = threading.Thread(target=capture._run)
    capture._thread.start()
    assert entered.wait(2)
    closer = threading.Thread(target=capture.stop)
    closer.start()
    try:
        assert capture._stop.wait(1)
        assert not calls
    finally:
        release.set()
        closer.join(3)
    assert calls == ['stop', 'close', 'terminate']
    capture.stop()
    assert calls == ['stop', 'close', 'terminate']


def test_repeated_interrupts_request_cleanup_without_raising():
    import signal
    import shutdown
    old = signal.getsignal(signal.SIGINT)
    with shutdown.signals():
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        assert shutdown.requested.is_set()
    assert signal.getsignal(signal.SIGINT) == old
    shutdown.requested.clear()
