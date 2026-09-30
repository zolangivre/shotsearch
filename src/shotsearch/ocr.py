"""Image loading and text extraction with Apple's Vision framework."""

from __future__ import annotations

from pathlib import Path

from ocrmac.ocrmac import text_from_image
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from shotsearch.config import OCR_LANGUAGES

register_heif_opener()


class UnreadableImageError(Exception):
    """The file exists but cannot be decoded as an image."""


def load_image(path: Path) -> Image.Image:
    """Open an image fully decoded, upright and in RGB."""
    try:
        with Image.open(path) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnreadableImageError("not a readable image") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise UnreadableImageError(f"corrupt image ({exc})") from exc


def extract_text(image: Image.Image) -> str:
    """Recognised text, one line per Vision observation, top to bottom."""
    observations = text_from_image(
        image,
        recognition_level="accurate",
        language_preference=OCR_LANGUAGES,
        detail=True,
    )
    # Vision uses a bottom-left origin: sort by descending top edge, then x.
    observations.sort(key=lambda o: (-round(o[2][1] + o[2][3], 2), o[2][0]))
    return "\n".join(text for text, _conf, _bbox in observations if text.strip())
