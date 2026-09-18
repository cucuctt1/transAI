"""Settings persistence (small JSON file)."""

import json
import os
from dataclasses import asdict, dataclass, fields

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


@dataclass
class Settings:
    # caption
    caption_width: int = 700
    font_size: int = 28
    font_family: str = "Segoe UI"
    padding: int = 20
    background_opacity: int = 75  # 0..100
    background_color: str = "#101010"
    text_color: str = "#ffffff"
    text_alignment: str = "center"  # left | center | right
    position: str = "bottom-center"  # top-left|top-center|top-right|center|bottom-left|bottom-center|bottom-right|custom
    margin: int = 20

    # behavior
    fade_duration: int = 150
    hide_until_caption: bool = True
    always_on_top: bool = True
    esc_closes: bool = True

    # connection
    connection_mode: str = "local"  # local | lan | remote
    server_address: str = "127.0.0.1:8765"
    pairing_code: str = ""

    # audio input (server side)
    audio_device: str = ""  # loopback device name; empty = default

    @classmethod
    def load(cls, path=_CONFIG_PATH):
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            pass
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save(self, path=_CONFIG_PATH):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
        except OSError:
            pass
