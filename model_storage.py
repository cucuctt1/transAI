"""Project-local model storage. Configure before importing Hugging Face."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'


def configure():
    for name, path in {
        'HF_HOME': MODELS / 'huggingface',
        'HF_HUB_CACHE': MODELS / 'huggingface' / 'hub',
        'HF_XET_CACHE': MODELS / 'huggingface' / 'xet',
    }.items():
        os.environ[name] = str(path)
    return MODELS
