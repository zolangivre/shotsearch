"""Generate small test images in tests/fixtures/.

uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener

register_heif_opener()

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in MONO_CANDIDATES if mono else FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def text_image(
    lines: list[str], *, size=(1200, 520), bg="white", fg="black", mono=False, pt=34
) -> Image.Image:
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    f = font(pt, mono)
    y = 50
    for line in lines:
        draw.text((50, y), line, fill=fg, font=f)
        y += int(pt * 1.6)
    return img


def build_failed() -> Image.Image:
    return text_image(
        [
            "$ npm run build",
            "> vite build",
            "src/App.tsx(12,7): error TS2322: Type 'string'",
            "is not assignable to type 'number'.",
            "Build failed with 1 error.",
        ],
        bg="#1e1e1e",
        fg="#e6e6e6",
        mono=True,
    )


def facture() -> Image.Image:
    return text_image(
        [
            "Facture n° 2026-042",
            "Client : Atelier Génial",
            "Montant total : 1 250,00 €",
            "Échéance : 15 octobre 2026",
            "Merci de régler par virement bancaire.",
        ]
    )


def sunset() -> Image.Image:
    """No text at all: only the image embedding can find this one."""
    w, h = 1200, 700
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    horizon = int(h * 0.62)
    for y in range(horizon):
        t = y / horizon
        draw.line([(0, y), (w, y)], fill=(int(60 + 195 * t), int(40 + 90 * t), int(120 - 60 * t)))
    draw.ellipse([w // 2 - 110, horizon - 120, w // 2 + 110, horizon + 100], fill=(255, 190, 70))
    for y in range(horizon, h):
        t = (y - horizon) / (h - horizon)
        draw.line([(0, y), (w, y)], fill=(int(30 + 40 * (1 - t)), int(50 + 30 * (1 - t)), 110))
    return img


def meeting_notes() -> Image.Image:
    return text_image(
        [
            "#team-product",
            "Alice: the meeting is moved to Thursday 3pm",
            "Bob: ok, I'll book the big room",
            "Alice: don't forget the quarterly roadmap slides",
        ],
        bg="#f8f8fa",
        pt=30,
    )


def recette() -> Image.Image:
    return text_image(
        [
            "Recette : tarte aux pommes",
            "4 pommes, 1 pâte brisée, 50 g de sucre",
            "Préchauffer le four à 180 °C",
            "Cuire 35 minutes",
        ],
        bg="#fff8ec",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_failed().save(OUT / "build_failed.png")
    facture().save(OUT / "facture.png")
    sunset().save(OUT / "sunset.png")
    meeting_notes().save(OUT / "meeting_notes.jpg", quality=90)
    recette().save(OUT / "recette.heic", quality=90)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:22} {p.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
