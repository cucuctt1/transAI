"""Tests for the UI's display-independent logic (no GUI display required)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Settings
from ui import CaptionController, gen_pairing_code, parse_addr
from transcript import TranscriptState


def test_caption_controller_latest_wins():
    c = CaptionController()
    c.submit("A")
    c.submit("B")
    c.submit("C")
    assert c.take() == "C", "latest caption must win"
    assert c.take() is None, "slot must clear after take"


def test_settings_roundtrip():
    path = os.path.join(tempfile.gettempdir(), "tt_settings_test.json")
    s = Settings(caption_width=640, font_size=32, background_opacity=60,
                 position="top-center", always_on_top=False, connection_mode="lan",
                 server_address="10.0.0.2:9000")
    s.save(path)
    s2 = Settings.load(path)
    assert s2.caption_width == 640
    assert s2.font_size == 32
    assert s2.background_opacity == 60
    assert s2.position == "top-center"
    assert s2.always_on_top is False
    assert s2.connection_mode == "lan"
    assert s2.server_address == "10.0.0.2:9000"
    os.remove(path)


def test_parse_addr():
    assert parse_addr("192.168.1.50:8765") == ("192.168.1.50", 8765)
    assert parse_addr("192.168.1.50") == ("192.168.1.50", 8765)


def test_latest_caption():
    st = TranscriptState(rollback_window=10.0, words_per_second=3.0)
    st.apply_hypothesis("Hello everyone.")
    assert st.latest_caption() == "Hello everyone"
    st.apply_hypothesis("Hello everyone. Welcome back.")
    assert st.latest_caption() == "Welcome back"


if __name__ == "__main__":
    test_caption_controller_latest_wins()
    test_settings_roundtrip()
    test_parse_addr()
    test_latest_caption()
    print("All UI logic tests passed.")
