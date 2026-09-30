"""Paths and settings. Everything lives under one local directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

APP_NAME = "shotsearch"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".heic", ".heif"})
OCR_LANGUAGES = ["fr-FR", "en-US"]


def home() -> Path:
    """Data directory: database and model cache. Override with SHOTSEARCH_HOME."""
    override = os.environ.get("SHOTSEARCH_HOME")
    base = (
        Path(override).expanduser()
        if override
        else Path.home() / "Library" / "Application Support" / APP_NAME
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path() -> Path:
    return home() / f"{APP_NAME}.db"


def models_dir() -> Path:
    """Model cache. Override with SHOTSEARCH_MODELS to share it between data dirs."""
    override = os.environ.get("SHOTSEARCH_MODELS")
    return Path(override).expanduser() if override else home() / "models"


def default_models_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME / "models"


def default_screenshot_dir() -> Path:
    """Folder macOS saves screenshots to, falling back to ~/Desktop."""
    try:
        out = subprocess.run(
            ["defaults", "read", "com.apple.screencapture", "location"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        location = out.stdout.strip()
        if out.returncode == 0 and location:
            path = Path(location).expanduser()
            if path.is_dir():
                return path
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / "Desktop"
