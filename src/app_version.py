"""Shared application version and the official release repository."""

import sys
from pathlib import Path


_resource_root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
APP_VERSION = (_resource_root / 'VERSION').read_text(encoding='utf-8-sig').strip()

GITHUB_OWNER = 'B0ogie888'
GITHUB_REPO = 'Meshropractor'
