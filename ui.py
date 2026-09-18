"""PyQt5 two-stage UI: settings window + borderless caption overlay.

Keeps ASR/network out of the UI thread: captions arrive via a thread-safe
latest-wins slot and are drained by a QTimer in the overlay.
"""

import random
import string
import sys
import threading

from PyQt5.QtCore import QObject, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import Settings
from transcript import Config


def parse_addr(addr):
    host, _, port = addr.partition(":")
    return host.strip() or "127.0.0.1", int(port.strip() or 8765)


def gen_pairing_code():
    return "-".join(
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        for _ in range(2)
    )


class CaptionController:
    """Thread-safe latest-caption slot (latest-wins)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None

    def submit(self, text):
        with self._lock:
            self._latest = (text or "").strip()

    def take(self):
        with self._lock:
            t = self._latest
            self._latest = None
            return t

    def clear(self):
        self.submit("")


class CaptionOverlay(QWidget):
    request_show_main = pyqtSignal()
    request_settings = pyqtSignal()
    request_clear = pyqtSignal()
    request_pause = pyqtSignal()
    request_close = pyqtSignal()

    def __init__(self, controller, settings):
        super().__init__()
        self.controller = controller
        self.settings = settings
        self._caption = ""
        self._lines = [""]
        self._paused = False
        self._dragging = False
        self._drag_offset = QPoint()
        self._custom_pos = None

        self._apply_flags()
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self._font = QFont(settings.font_family, settings.font_size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30)

        self._place()
        self._relayout()

    # ---- settings ---------------------------------------------------------

    def _apply_flags(self):
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.settings.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def apply_settings(self, settings):
        self.settings = settings
        self._font = QFont(settings.font_family, settings.font_size)
        self._apply_flags()
        self._place()
        self._relayout()

    def _single_line_height(self):
        fm = QFontMetrics(self._font)
        return fm.lineSpacing() + 2 * self.settings.padding

    def _place(self):
        screen = QApplication.primaryScreen().availableGeometry()
        w = self.settings.caption_width
        ref_h = self._single_line_height()
        m = self.settings.margin
        pos = self.settings.position
        if pos == "custom" and self._custom_pos is not None:
            x, y = self._custom_pos.x(), self._custom_pos.y()
        else:
            if pos.startswith("top"):
                y = screen.top() + m
            elif pos.startswith("bottom"):
                y = screen.bottom() - m - ref_h
            else:
                y = screen.top() + (screen.height() - ref_h) // 2
            if pos.endswith("left"):
                x = screen.left() + m
            elif pos.endswith("right"):
                x = screen.right() - m - w
            else:
                x = screen.left() + (screen.width() - w) // 2
        self.move(x, y)

    # ---- caption polling (latest-wins + dedup) ----------------------------

    def _poll(self):
        text = self.controller.take()
        if text is None or self._paused:
            return
        if text == self._caption:
            return
        self._caption = text
        self._relayout()

    def toggle_pause(self):
        self._paused = not self._paused

    def clear_caption(self):
        self._caption = ""
        self._relayout()

    # ---- layout / painting -------------------------------------------------

    def _relayout(self):
        fm = QFontMetrics(self._font)
        if not self._caption:
            self._lines = [""]
            self.setFixedSize(self.settings.caption_width, self._single_line_height())
            if self.settings.hide_until_caption:
                self.hide()
            else:
                self.show()
                self.update()
            return
        self.show()
        avail = self.settings.caption_width - 2 * self.settings.padding
        self._lines = self._wrap(self._caption, fm, avail)
        text_h = len(self._lines) * fm.lineSpacing()
        self.setFixedSize(self.settings.caption_width, text_h + 2 * self.settings.padding)
        self.update()

    @staticmethod
    def _wrap(text, fm, avail):
        lines = []
        for paragraph in text.split("\n"):
            cur = ""
            for w in paragraph.split():
                cand = (cur + " " + w).strip()
                if fm.horizontalAdvance(cand) <= avail or not cur:
                    cur = cand
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
        return lines or [""]

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(self.settings.background_color)
        bg.setAlphaF(max(0.0, min(1.0, self.settings.background_opacity / 100.0)))
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)
        p.setPen(QColor(self.settings.text_color))
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        pad = self.settings.padding
        y = pad + fm.ascent()
        align = self.settings.text_alignment
        for line in self._lines:
            lw = fm.horizontalAdvance(line)
            if align == "center":
                x = (self.width() - lw) // 2
            elif align == "right":
                x = self.width() - pad - lw
            else:
                x = pad
            p.drawText(x, y, line)
            y += fm.lineSpacing()
        p.end()

    # ---- drag + right-click + ESC ------------------------------------------

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()
        elif e.button() == Qt.RightButton:
            self._show_menu(e.globalPos())

    def mouseMoveEvent(self, e):
        if self._dragging and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPos() - self._drag_offset)
            self._custom_pos = self.pos()
            self.settings.position = "custom"

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.settings.esc_closes:
            self.request_close.emit()

    def _show_menu(self, global_pos):
        menu = QMenu(self)
        menu.addAction("Show Main Window", self.request_show_main.emit)
        menu.addAction("Settings", self.request_settings.emit)
        menu.addAction("Clear Caption", self.request_clear.emit)
        menu.addAction("Pause Caption" if not self._paused else "Resume Caption", self.request_pause.emit)
        menu.addSeparator()
        menu.addAction("Close Transcript", self.request_close.emit)
        menu.exec_(global_pos)


class MainWindow(QMainWindow):
    run_clicked = pyqtSignal()
    stop_server_clicked = pyqtSignal()

    def __init__(self, settings, mode="local"):
        super().__init__()
        self.settings = settings
        self.mode = mode  # local | server | client
        self.setWindowTitle("Realtime Translator")
        self.resize(420, 560)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # caption settings
        cap = QGroupBox("Caption Settings")
        form = QFormLayout(cap)
        self.width_spin = QSpinBox(); self.width_spin.setRange(200, 2000); self.width_spin.setValue(settings.caption_width)
        self.font_spin = QSpinBox(); self.font_spin.setRange(8, 96); self.font_spin.setValue(settings.font_size)
        self.opacity_spin = QSpinBox(); self.opacity_spin.setRange(10, 100); self.opacity_spin.setValue(settings.background_opacity)
        self.padding_spin = QSpinBox(); self.padding_spin.setRange(0, 100); self.padding_spin.setValue(settings.padding)
        self.position_combo = QComboBox()
        for label in ["Top Left", "Top Center", "Top Right", "Center",
                      "Bottom Left", "Bottom Center", "Bottom Right", "Custom"]:
            self.position_combo.addItem(label)
        self.position_combo.setCurrentText(self._position_label(settings.position))
        self.margin_spin = QSpinBox(); self.margin_spin.setRange(0, 400); self.margin_spin.setValue(settings.margin)
        self.always_top_check = QRadioButton("Always on top"); self.always_top_check.setChecked(settings.always_on_top)
        form.addRow("Width:", self.width_spin)
        form.addRow("Font size:", self.font_spin)
        form.addRow("Opacity %:", self.opacity_spin)
        form.addRow("Padding:", self.padding_spin)
        form.addRow("Position:", self.position_combo)
        form.addRow("Margin:", self.margin_spin)
        form.addRow(self.always_top_check)
        root.addWidget(cap)

        # connection
        conn = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn)
        self.local_radio = QRadioButton("Local")
        self.lan_radio = QRadioButton("LAN Server")
        self.remote_radio = QRadioButton("Colab / Remote")
        self.local_radio.setChecked(settings.connection_mode == "local")
        self.lan_radio.setChecked(settings.connection_mode == "lan")
        self.remote_radio.setChecked(settings.connection_mode == "remote")
        conn_layout.addWidget(self.local_radio)
        conn_layout.addWidget(self.lan_radio)
        conn_layout.addWidget(self.remote_radio)
        addr_row = QHBoxLayout()
        addr_row.addWidget(QLabel("Server:"))
        self.server_edit = QLineEdit(settings.server_address)
        addr_row.addWidget(self.server_edit)
        conn_layout.addLayout(addr_row)
        code_row = QHBoxLayout()
        code_row.addWidget(QLabel("Pairing Code:"))
        self.code_edit = QLineEdit(settings.pairing_code)
        code_row.addWidget(self.code_edit)
        conn_layout.addLayout(code_row)
        root.addWidget(conn)

        # server panel (server mode only)
        self.server_panel = QGroupBox("Server")
        self.server_status = QLabel("Status: Stopped")
        panel_layout = QFormLayout(self.server_panel)
        self.lan_addr_label = QLabel("-")
        self.clients_label = QLabel("0 connected")
        self.gpu_label = QLabel("-")
        self.latency_label = QLabel("-")
        panel_layout.addRow("Status:", self.server_status)
        panel_layout.addRow("LAN Address:", self.lan_addr_label)
        panel_layout.addRow("Pairing Code:", QLabel(settings.pairing_code or "-"))
        panel_layout.addRow("Clients:", self.clients_label)
        panel_layout.addRow("ASR:", QLabel("CUDA"))
        panel_layout.addRow("GPU:", self.gpu_label)
        panel_layout.addRow("Latency:", self.latency_label)
        self.server_panel.setVisible(mode == "server")
        root.addWidget(self.server_panel)

        # audio input selection (client captures the audio; server only processes)
        self.audio_combo = None
        if mode != "server":
            audio_grp = QGroupBox("Audio Input")
            audio_form = QFormLayout(audio_grp)
            self.audio_combo = QComboBox()
            self._audio_devices = []
            self.refresh_audio_button = QPushButton("Refresh")
            self.refresh_audio_button.clicked.connect(self._refresh_audio_devices)
            audio_row = QHBoxLayout()
            audio_row.addWidget(self.audio_combo)
            audio_row.addWidget(self.refresh_audio_button)
            audio_form.addRow("Loopback device:", audio_row)
            root.addWidget(audio_grp)
            self._refresh_audio_devices(select=settings.audio_device)

        self.status_label = QLabel("Status: Ready")
        root.addWidget(self.status_label)

        self.run_button = QPushButton("RUN")
        self.run_button.clicked.connect(self.run_clicked.emit)
        root.addWidget(self.run_button)

        if mode == "server":
            self.stop_button = QPushButton("Stop Server")
            self.stop_button.clicked.connect(self.stop_server_clicked.emit)
            root.addWidget(self.stop_button)

    @staticmethod
    def _position_label(key):
        return {"top-left": "Top Left", "top-center": "Top Center", "top-right": "Top Right",
                "center": "Center", "bottom-left": "Bottom Left", "bottom-center": "Bottom Center",
                "bottom-right": "Bottom Right", "custom": "Custom"}.get(key, "Bottom Center")

    @staticmethod
    def _position_key(label):
        return {"Top Left": "top-left", "Top Center": "top-center", "Top Right": "top-right",
                "Center": "center", "Bottom Left": "bottom-left", "Bottom Center": "bottom-center",
                "Bottom Right": "bottom-right", "Custom": "custom"}.get(label, "bottom-center")

    def collect_settings(self, settings):
        settings.caption_width = self.width_spin.value()
        settings.font_size = self.font_spin.value()
        settings.background_opacity = self.opacity_spin.value()
        settings.padding = self.padding_spin.value()
        settings.position = self._position_key(self.position_combo.currentText())
        settings.margin = self.margin_spin.value()
        settings.always_on_top = self.always_top_check.isChecked()
        if self.local_radio.isChecked():
            settings.connection_mode = "local"
        elif self.lan_radio.isChecked():
            settings.connection_mode = "lan"
        else:
            settings.connection_mode = "remote"
        settings.server_address = self.server_edit.text().strip()
        settings.pairing_code = self.code_edit.text().strip()
        if self.audio_combo is not None and self.audio_combo.currentText():
            settings.audio_device = self.audio_combo.currentText()

    def set_status(self, text):
        self.status_label.setText("Status: " + text)

    def set_server_status(self, running, lan_addr="", clients=0, gpu="-", latency=""):
        self.server_status.setText("Status: " + ("Running" if running else "Stopped"))
        self.lan_addr_label.setText(lan_addr or "-")
        self.clients_label.setText(f"{clients} connected")
        self.gpu_label.setText(gpu or "-")
        self.latency_label.setText(latency or "-")

    def _refresh_audio_devices(self, select=None):
        if self.audio_combo is None:
            return
        try:
            from audio_capture import list_loopback_devices
            self._audio_devices = list_loopback_devices()
        except Exception:
            self._audio_devices = []
        self.audio_combo.clear()
        for d in self._audio_devices:
            self.audio_combo.addItem(d["name"])
        if select:
            idx = self.audio_combo.findText(select)
            if idx >= 0:
                self.audio_combo.setCurrentIndex(idx)

    def selected_audio_index(self):
        if self.audio_combo is None or self.audio_combo.currentIndex() < 0:
            return None
        return self._audio_devices[self.audio_combo.currentIndex()]["index"]


class ConnectionController(QObject):
    status_changed = pyqtSignal(str)

    def __init__(self, caption_controller):
        super().__init__()
        self.caption_controller = caption_controller
        self.asr = None
        self.client = None
        self.server = None
        self.capture = None

    def start_client(self, host, port, code, audio_device_index, log_cb):
        from audio_capture import AudioCapture  # lazy import (pyaudiowpatch, no torch)
        from server import StreamingClient
        self.client = StreamingClient(host, port, code,
                                      caption_cb=self.caption_controller.submit,
                                      log_cb=log_cb)
        self.client.start()
        # The client captures system audio and streams it to the server.
        self.capture = AudioCapture(buffer_seconds=7.0, device_index=audio_device_index)
        self.capture.set_consumer(self.client.send_audio)
        self.capture.start()
        log_cb(f"[CLIENT] capturing: {self.capture.device_name}")
        return self.client

    def start_server(self, cfg, translator, gpu_name, code, log_cb):
        from asr_engine import ASREngine  # lazy import (torch already loaded by caller)
        from server import AudioCaptionServer, NetworkAudioSource, get_lan_ip
        self.audio_source = NetworkAudioSource(buffer_seconds=cfg.audio_buffer)
        self.server = AudioCaptionServer(self.audio_source, pairing_code=code, log_cb=log_cb)
        self.server.start()
        self.asr = ASREngine(cfg, translator, gpu_name, self.audio_source,
                             caption_cb=self.server.broadcast, log_cb=log_cb)
        self.asr.start()
        return self.server, get_lan_ip()

    def stop(self):
        for obj in (self.asr, self.client, self.server):
            if obj is not None:
                try:
                    obj.stop()
                except Exception:
                    pass
        if getattr(self, "capture", None) is not None:
            try:
                self.capture.stop()
            except Exception:
                pass
            self.capture = None
        self.asr = self.client = self.server = None


class App(QObject):
    def __init__(self, mode="local", translator=None, gpu_name=None):
        super().__init__()
        self.mode = mode  # local | server | client
        self.translator = translator
        self.gpu_name = gpu_name
        self.settings = Settings.load()
        if mode == "server":
            self.settings.connection_mode = "local"
        self.caption_controller = CaptionController()
        self.main_window = MainWindow(self.settings, mode=mode)
        self.overlay = CaptionOverlay(self.caption_controller, self.settings)
        self.connection = ConnectionController(self.caption_controller)
        self._server_running = False
        self._pairing_code = ""
        self.launcher = None

        self.main_window.run_clicked.connect(self.on_run)
        self.main_window.stop_server_clicked.connect(self.on_stop_server)
        self.overlay.request_close.connect(self.on_close_transcript)
        self.overlay.request_show_main.connect(self.show_main)
        self.overlay.request_settings.connect(self.show_main)
        self.overlay.request_clear.connect(self.overlay.clear_caption)
        self.overlay.request_pause.connect(self.overlay.toggle_pause)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_server_status)
        self._status_timer.start(1000)

        if mode == "server" and gpu_name:
            self.main_window.set_server_status(False, gpu=gpu_name)

        import atexit
        atexit.register(self._shutdown)

    def _shutdown(self):
        self.connection.stop()
        if self.launcher is not None:
            self.launcher.stop()

    # ---- state machine -----------------------------------------------------

    def show_main(self):
        self.overlay.hide()
        self.main_window.show()

    def on_run(self):
        self.main_window.collect_settings(self.settings)
        self.settings.save()
        self.overlay.apply_settings(self.settings)

        if self.settings.connection_mode == "remote":
            self.main_window.set_status("Remote/Colab not yet implemented")
            return

        self.connection.stop()
        log_cb = lambda line: print(line, flush=True)

        try:
            if self.mode == "server":
                self._pairing_code = self.settings.pairing_code  # empty = no auth
                self.connection.start_server(self._cfg(), self.translator, self.gpu_name,
                                             self._pairing_code, log_cb)
                self._server_running = True
                self.main_window.set_server_status(True, clients=0, gpu=self.gpu_name or "loading...")
            elif self.settings.connection_mode == "lan":
                host, port = parse_addr(self.settings.server_address)
                self.connection.start_client(host, port, self.settings.pairing_code,
                                             self.main_window.selected_audio_index(), log_cb)
                self.main_window.set_status("Connecting...")
                self.main_window.hide()
                self.overlay.show()
            else:  # local: auto-spawn a headless server, then connect (thin client)
                from server import LocalServerLauncher
                if self.launcher is None:
                    self.launcher = LocalServerLauncher(8765, log_cb=log_cb)
                if not self.launcher.ensure():
                    self.main_window.set_status("Could not start local server")
                    return
                self.connection.start_client("127.0.0.1", 8765, self.settings.pairing_code,
                                             self.main_window.selected_audio_index(), log_cb)
                self.main_window.hide()
                self.overlay.show()
        except Exception as exc:
            print(f"[UI] failed to start connection: {exc}", file=sys.stderr, flush=True)
            self.main_window.set_status(f"Error: {exc}")

    def on_stop_server(self):
        self.connection.stop()
        self._server_running = False
        self.main_window.set_server_status(False)

    def on_close_transcript(self):
        self.overlay.hide()
        self.main_window.show()

    def _refresh_server_status(self):
        if self.mode == "server" and self._server_running:
            server = self.connection.server
            clients = server.client_count if server else 0
            gpu = self.gpu_name or ""
            if self.connection.asr is not None:
                gpu = self.connection.asr.gpu_name or gpu
            self.main_window.set_server_status(True, clients=clients, gpu=gpu)

    @staticmethod
    def _cfg():
        return Config()  # default 5s window / 1s step / 3s rollback


def run_app(mode, translator=None, gpu_name=None):
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    controller = App(mode, translator=translator, gpu_name=gpu_name)
    controller.show_main()
    app.exec_()


def run_client():
    run_app("client")


def run_server(translator=None, gpu_name=None):
    run_app("server", translator, gpu_name)
