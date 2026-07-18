"""
Image utility functions used during preprocessing.

Responsibility:
    - Load images
    - Normalize images
    - Save temporary PNGs

Nothing else.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import io
import pillow_heif
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

pillow_heif.register_heif_opener()

TEMP_DIR = Path("data/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> Image.Image:
    """
    Load image from disk.

    Always converts to RGB because
    vision models expect RGB images.
    """

    image = Image.open(path)

    image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def normalize_image(
    image: Image.Image,
    max_dimension: int = 896,
) -> Image.Image:
    """
    Resize image while preserving aspect ratio.

    Large images waste inference time.
    Small images are left unchanged.
    """

    width, height = image.size

    longest = max(width, height)

    if longest <= max_dimension:
        return image

    scale = max_dimension / longest

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def save_temp_png(image: Image.Image) -> Path:
    """
    Save image as temporary PNG.

    Returns
    -------
    Path
        Path to temporary PNG.
    """

    filename = f"{uuid.uuid4().hex}.png"

    output = TEMP_DIR / filename

    image.save(output, format="PNG")

    return output


def remove_temp_file(path: Path) -> None:
    """
    Delete temporary PNG.
    """

    if path.exists():
        path.unlink()

def image_from_pixmap(pixmap) -> Image.Image:
    """
    Build a PIL Image directly from a PyMuPDF pixmap — no disk round-trip.
    """
    mode = "RGB" if pixmap.n == 3 else "RGBA" if pixmap.n == 4 else "L"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    return image.convert("RGB")


def text_to_image(
    text: str,
    width: int = 1100,
    margin: int = 40,
) -> Image.Image:
    """
    Render plain text into an RGB image.

    This allows Vision Language Models to reason over
    textual documents while preserving layout.
    """

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(dummy)

    lines = []

    for paragraph in text.splitlines():

        current = ""

        for word in paragraph.split():

            test = current + " " + word if current else word

            if draw.textlength(test, font=font) <= width - margin * 2:

                current = test

            else:

                lines.append(current)

                current = word

        lines.append(current)

    line_height = 32

    height = max(300, margin * 2 + line_height * len(lines))

    image = Image.new("RGB", (width, height), "white")

    draw = ImageDraw.Draw(image)

    y = margin

    for line in lines:

        draw.text((margin, y), line, fill="black", font=font)

        y += line_height

    return image


def load_svg_as_image(path: Path) -> Image.Image:
    """
    Rasterize an SVG to a PIL Image. Pure-Python (svglib + reportlab),
    no native Cairo/GTK dependency needed on Windows.
    """
    drawing = svg2rlg(str(path))
    png_bytes = renderPM.drawToString(drawing, fmt="PNG")
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")