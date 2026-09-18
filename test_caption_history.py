"""Run Qt offscreen in a fresh process, separate from native model imports."""
import os
import subprocess
import sys


def test_history_transition_duplicate_and_clear():
    script = '''
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase
from pathlib import Path
from ui import CaptionController, CaptionOverlay
from config import Settings
app = QApplication([])
font_path = Path('C:/Windows/Fonts/segoeui.ttf')
if font_path.exists():
    QFontDatabase.addApplicationFont(str(font_path))
c = CaptionController()
w = CaptionOverlay(c, Settings())
c.submit('First phrase.')
w._poll()
assert w._history == ''
c.submit('Second phrase.')
w._poll()
assert w._history == 'First phrase.'
started = w._history_started
c.submit('Second phrase.')
w._poll()
assert w._history_started == started
app.processEvents()
rendered = w.grab().toImage()
assert not rendered.isNull()
if QFontDatabase().families():
    assert any(rendered.pixelColor(x,y).red() > 150
               for x in range(0,rendered.width(),2) for y in range(0,rendered.height(),2))
w._history_started -= 5
w._poll()
assert w._history == '' and w._caption == 'Second phrase.'
c.submit('Third phrase.')
w._poll()
assert w._history == 'Second phrase.'
c.submit('')
w._poll()
assert w._caption == '' and w._history == '' and not w.isVisible()
w._timer.stop()
w.deleteLater()
app.processEvents()
'''
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
