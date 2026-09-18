"""Client/server caption streaming over TCP.

Architecture: the CLIENT captures system audio and streams it to the SERVER; the
SERVER runs Whisper and streams translated captions back.

Protocol (single full-duplex TCP connection):
  client -> server  handshake:  {"type":"hello","pairing_code":"..."}\\n
  server -> client  handshake:  {"type":"ok"}\\n
  client -> server  audio frame:  [4-byte BE length][float32 mono 16k bytes]
  server -> client  caption:     {"type":"caption","text":"..."}\\n
"""

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np


def _close_socket(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _recv_line(sock, buf, stop_event=None):
    """Read one newline-terminated line, returning (line, remaining_buffer).

    Tolerates socket timeouts (keeps waiting) instead of treating them as a
    connection failure.
    """
    while b"\n" not in buf:
        if stop_event is not None and stop_event.is_set():
            return None, buf
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return None, buf
        if not chunk:
            return None, buf
        buf += chunk
    line, rest = buf.split(b"\n", 1)
    return line, rest


def _read_frame(sock, buf, stop_event=None):
    """Read one length-prefixed binary frame, returning (frame, remaining_buffer)."""
    while len(buf) < 4:
        if stop_event is not None and stop_event.is_set():
            return None, buf
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return None, buf
        if not chunk:
            return None, buf
        buf += chunk
    (n,) = struct.unpack(">I", buf[:4])
    while len(buf) < 4 + n:
        if stop_event is not None and stop_event.is_set():
            return None, buf
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return None, buf
        if not chunk:
            return None, buf
        buf += chunk
    return buf[4:4 + n], buf[4 + n:]


class NetworkAudioSource:
    """Server-side rolling buffer fed by streamed client audio (16 kHz mono float32).

    Implements the same interface as AudioCapture (available_seconds / latest /
    live_position / start / stop) so ASREngine can consume it unchanged.
    """

    SAMPLERATE = 16000

    def __init__(self, buffer_seconds=7.0, block_seconds=0.2):
        from streaming import AudioRing
        self._ring = AudioRing(buffer_seconds)
        self.buffer_seconds = buffer_seconds
        self.samplerate = self.SAMPLERATE

    def feed(self, samples):
        self._ring.feed(samples)

    def available_seconds(self):
        return self._ring.available_seconds()

    def live_position(self):
        return self._ring.live_position()

    def latest(self, seconds):
        return self._ring.latest(seconds)

    def start(self):
        pass

    def stop(self):
        pass


class AudioCaptionServer:
    """Accepts clients; the first client's audio feeds `audio_source`; captions are
    broadcast to every client."""

    def __init__(self, audio_source, host="0.0.0.0", port=8765, pairing_code="", log_cb=None):
        self.audio_source = audio_source
        self.host = host
        self.port = port
        self.pairing_code = pairing_code or ""
        self.log_cb = log_cb or (lambda line: None)
        self._sock = None
        self._clients = set()
        self._audio_client = None
        self._connections = set()
        self._handlers = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def client_count(self):
        with self._lock:
            return len(self._clients)

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        self.log_cb(f"[SERVER] listening on {self.host}:{self.port}")

    def broadcast(self, caption):
        with self._lock:
            clients = list(self._clients)
        for sock in clients:
            try:
                _send_json(sock, {"type": "caption", "text": caption})
            except OSError:
                with self._lock:
                    self._clients.discard(sock)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.1)  # transient accept error (e.g. aborted probe) -> keep serving
                continue
            with self._lock:
                self._connections.add(conn)
                worker = threading.Thread(target=self._handle_tracked, args=(conn,), daemon=False)
                self._handlers.add(worker)
            worker.start()

    def _handle_tracked(self, conn):
        try:
            self._handle(conn)
        except OSError:
            pass  # socket shutdown wakes handshake/audio reads
        finally:
            _close_socket(conn)
            with self._lock:
                self._connections.discard(conn)
                self._handlers.discard(threading.current_thread())

    def _handle(self, conn):
        conn.settimeout(1.0)
        buf = b""
        line, buf = _recv_line(conn, buf, self._stop)
        if line is None:
            conn.close()
            return
        try:
            msg = json.loads(line.decode("utf-8", "ignore"))
        except ValueError:
            conn.close()
            return
        if msg.get("type") != "hello":
            conn.close()
            return
        if self.pairing_code and msg.get("pairing_code") != self.pairing_code:
            _send_json(conn, {"type": "error", "reason": "bad pairing code"})
            self.log_cb("[SERVER] rejected client: bad pairing code")
            conn.close()
            return
        _send_json(conn, {"type": "ok"})
        with self._lock:
            self._clients.add(conn)
            is_audio = self._audio_client is None
            if is_audio:
                self._audio_client = conn
        self.log_cb(f"[SERVER] client connected ({self.client_count} total, audio={'yes' if is_audio else 'no'})")

        try:
            if is_audio:
                self._read_audio(conn, buf)
            else:
                self._keepalive(conn, buf)
        finally:
            with self._lock:
                self._clients.discard(conn)
                if self._audio_client is conn:
                    self._audio_client = None
            conn.close()

    def _read_audio(self, conn, buf):
        conn.settimeout(1.0)
        while not self._stop.is_set():
            frame, buf = _read_frame(conn, buf, self._stop)
            if frame is None:
                break
            self.audio_source.feed(np.frombuffer(frame, dtype=np.float32))

    def _keepalive(self, conn, buf):
        conn.settimeout(1.0)
        while not self._stop.is_set():
            try:
                if conn.recv(1024) == b"":
                    break
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self):
        self._stop.set()
        _close_socket(self._sock)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        with self._lock:
            clients = list(self._connections)
            handlers = list(self._handlers)
        for c in clients:
            _close_socket(c)
        for worker in handlers:
            worker.join()
        with self._lock:
            self._clients.clear()


class StreamingClient:
    """Captures audio via its `send_audio` hook and streams it to the server, while
    receiving captions from the server."""

    def __init__(self, host, port, pairing_code="", caption_cb=None, status_cb=None, log_cb=None):
        self.host = host
        self.port = port
        self.pairing_code = pairing_code or ""
        self.caption_cb = caption_cb or (lambda text: None)
        self.status_cb = status_cb or (lambda connected: None)
        self.log_cb = log_cb or (lambda line: None)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._send_lock = threading.Lock()
        self._sock = None
        self._thread = None
        self.connected = False
        self._stats_thread = None
        self.frames_sent = 0
        self.samples_sent = 0
        self.captions_received = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._stats_thread = threading.Thread(target=self._stats_loop, name='client-stats', daemon=False)
        self._stats_thread.start()

    def _stats_loop(self):
        while not self._stop.wait(2.0):
            self.log_cb(
                f"[CLIENT] connected={self.connected} audio_sent={self.samples_sent / 16000:.1f}s "
                f"frames={self.frames_sent} captions={self.captions_received}"
            )

    def stop(self):
        self._stop.set()
        self._ready.clear()
        _close_socket(self._sock)
        for worker in (self._thread, self._stats_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join()

    def send_audio(self, block):
        """Stream a 16 kHz mono float32 block to the server (called by the capturer)."""
        if not self._ready.is_set():
            return
        with self._send_lock:
            if self._sock is None:
                return
            try:
                data = np.ascontiguousarray(block, dtype=np.float32).tobytes()
                self._sock.sendall(struct.pack(">I", len(data)) + data)
                self.frames_sent += 1
                self.samples_sent += len(data) // 4
            except OSError:
                pass

    def _run(self):
        while not self._stop.is_set():
            self._ready.clear()
            self.log_cb(f"[CLIENT] connecting to {self.host}:{self.port}...")
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    self._sock = sock
                    sock.settimeout(1.0)
                    _send_json(sock, {"type": "hello", "pairing_code": self.pairing_code})
                    self._ready.set()
                    self.log_cb(f"[CLIENT] connected to {self.host}:{self.port}")
                    buf = b""
                    while not self._stop.is_set():
                        line, buf = _recv_line(sock, buf, self._stop)
                        if line is None:
                            break
                        try:
                            msg = json.loads(line.decode("utf-8", "ignore"))
                        except ValueError:
                            continue
                        if msg.get("type") == "caption":
                            self.connected = True
                            self.status_cb(True)
                            self.captions_received += 1
                            self.caption_cb(msg.get("text", ""))
                        elif msg.get("type") == "ok":
                            self.connected = True
                            self.status_cb(True)
                        elif msg.get("type") == "error":
                            self.log_cb(f"[CLIENT] server error: {msg.get('reason')}")
                            self.connected = False
                            self.status_cb(False)
                            self._stop.wait(2.0)
            except OSError as exc:
                self.log_cb(f"[CLIENT] connection failed: {exc}")
            finally:
                self._sock = None
                self._ready.clear()
                self.connected = False
                self.status_cb(False)
            for _ in range(50):
                if self._stop.is_set():
                    return
                self._stop.wait(0.1)


class LocalServerLauncher:
    """Spawns a headless caption-server subprocess for "local" client mode."""

    def __init__(self, port=8765, log_cb=None, extra_args=None):
        self.port = port
        self.extra_args = list(extra_args or [])
        # First Moonshine launch downloads two models and imports Transformers.
        # Keep checking cancellation/process exit while allowing that cold start.
        selected_backend = "faster-whisper"
        for index, arg in enumerate(self.extra_args):
            if arg.startswith("--backend="):
                selected_backend = arg.split("=", 1)[1]
            elif arg == "--backend" and index + 1 < len(self.extra_args):
                selected_backend = self.extra_args[index + 1]
        self.startup_timeout = 600 if selected_backend == "moonshine" else 60
        self.log_cb = log_cb or (lambda line: None)
        self.proc = None
        self.script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

    def is_up(self):
        try:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=0.5)
            s.close()
            return True
        except OSError:
            return False

    def ensure(self):
        if self.proc is not None and self.proc.poll() is None and self.is_up():
            return True
        if self.is_up():
            # Never inherit mode/model options from an unrelated old server.
            # Do not stop it: another client may still be using it.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                self.port = listener.getsockname()[1]
            self.log_cb(f"[LOCAL] existing server left running; using private port {self.port}")
        self.log_cb(f"[LOCAL] starting headless server on port {self.port}...")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, self.script, *self.extra_args, "--server-headless", "--parent-control", "--port", str(self.port)],
                cwd=os.path.dirname(self.script),
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
        except OSError as exc:
            self.log_cb(f"[LOCAL] failed to start server: {exc}")
            return False
        for _ in range(self.startup_timeout * 2):
            import shutdown
            if shutdown.requested.is_set():
                self.stop()
                return False
            if self.is_up():
                self.log_cb("[LOCAL] server is up")
                return True
            if self.proc.poll() is not None:
                self.log_cb(f"[LOCAL] server exited early (code {self.proc.returncode})")
                return False
            time.sleep(0.5)
        self.stop()
        return False

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'STOP\n')
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass
            finally:
                if self.proc.stdin is not None:
                    try:
                        self.proc.stdin.close()
                    except OSError:
                        pass
            self.log_cb('[LOCAL] waiting for server inference and cleanup...')
            self.proc.wait()
        elif self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc = None
