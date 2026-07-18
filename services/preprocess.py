from __future__ import annotations


from PIL.ImagePalette import raw
import fitz
from docx import Document
import csv
from models.file_job import FileJob
from odf.opendocument import load as load_odf
from odf.text import P
from striprtf.striprtf import rtf_to_text
from utils.image_utils import (
    load_image,
    normalize_image,
    save_temp_png,
    text_to_image,
    image_from_pixmap,
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rtf",
  
}


MAX_EXTRACTED_CHARS = 4000  # ≈1000-1300 tokens, leaves headroom for the ~400-token system prompt within an 8192 context


def _cap_text(text: str) -> str:
    if len(text) > MAX_EXTRACTED_CHARS:
        return text[:MAX_EXTRACTED_CHARS] + "\n...[truncated]"
    return text

def preprocess(job: FileJob) -> None:
    """
    Preprocess a supported file into a normalized PNG
    consumable by the Vision Language Model.

    Mutates FileJob in-place.
    """

    try:

        suffix = job.extension.lower()

        if suffix == ".svg":
            _process_svg(job)

        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            _process_image(job)

        elif suffix == ".pdf":
            _process_pdf(job)

        elif suffix in {".doc", ".docx"}:
            _process_docx(job)

        elif suffix == ".csv":
            _process_csv(job)

        elif suffix == ".odt":
            _process_odt(job)

        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            _process_text(job)

        else:
            job.add_error(
                f"Unsupported extension: {suffix}"
            )
            return

        if job.is_text_document:

            if not job.extracted_text:

                job.add_error(
                    "No text extracted."
                )

        else:

            validate_preprocessed(job)

    except Exception as exc:

        job.add_error(str(exc))

def _process_image(job: FileJob) -> None:
    """
    Normalize an image and store a temporary PNG for the VLM.
    """

    image = load_image(job.path)

    image = normalize_image(image)

    job.rendered_image_path = save_temp_png(image)

def _process_svg(job: FileJob) -> None:
    """
    SVG rasterized via MuPDF's built-in SVG support — same pixmap
    pipeline as PDF, no separate rasterizer dependency needed.
    """
    document = fitz.open(job.path)
    try:
        page = document.load_page(0)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0),
            alpha=False,
        )
        image = image_from_pixmap(pixmap)
        image = normalize_image(image)
        job.rendered_image_path = save_temp_png(image)
    finally:
        document.close()


def _process_pdf(job: FileJob) -> None:
    """
    Render the first page of a PDF to an image, entirely in memory.
    """
    document = fitz.open(job.path)
    try:
        job.page_count = document.page_count

        if document.page_count == 0:
            raise ValueError("Empty PDF.")

        page = document.load_page(0)

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0),
            alpha=False,
        )

        image = image_from_pixmap(pixmap)
        image = normalize_image(image)

        job.rendered_image_path = save_temp_png(image)

    finally:
        document.close()


def _process_odt(job: FileJob) -> None:
    try:
        doc = load_odf(job.path)
        paragraphs = doc.getElementsByType(P)
        text = "\n".join(
            "".join(node.data for node in p.childNodes if node.nodeType == node.TEXT_NODE)
            for p in paragraphs
        )
    except Exception as exc:
        job.add_error(f"Failed to read ODT: {exc}")
        return

    job.extracted_text = _cap_text(text.strip())
    job.is_text_document = True

    
def _process_docx(job: FileJob) -> None:
    """
    DOCX -> Extract text -> Render as image.

    We intentionally render extracted text into an image so the
    same Vision model processes every supported format.
    """

    document = Document(str(job.path))

    paragraphs = [
        p.text.strip()
        for p in document.paragraphs
        if p.text.strip()
    ]

    text = "\n".join(paragraphs)

    if not text:
        text = "[Empty DOCX]"

    job.is_text_document = True

    job.extracted_text = _cap_text(text)

    job.rendered_image_path = None

def _process_text(job: FileJob) -> None:
    """
    TXT / MD / RTF

    RTF is converted to plain text first — sending raw RTF markup
    wastes tokens on control codes the model doesn't need.
    """

    raw_text = job.path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if job.extension.lower() == ".rtf":
        text = rtf_to_text(raw_text)
    else:
        text = raw_text

    if not text.strip():
        text = "[Empty Document]"

    job.is_text_document = True

    job.extracted_text = _cap_text(text)

    job.rendered_image_path = None

    
def _process_csv(job: FileJob) -> None:
    """
    CSV -> plain text.

    No rendering required.
    """

    rows: list[str] = []

    with open(
        job.path,
        newline="",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        reader = csv.reader(file)

        for i, row in enumerate(reader):

            if i >= 40:
                rows.append("...")
                break

            rows.append(" | ".join(row))

    text = "\n".join(rows)

    if not text.strip():
        text = "[Empty CSV]"

    job.is_text_document = True

    job.extracted_text = _cap_text(text)

    job.rendered_image_path = None


def validate_preprocessed(job: FileJob) -> bool:
    """
    Verify preprocessing produced a valid rendered image.

    Returns
    -------
    bool
        True if the rendered image exists and is ready for inference.
    """

    if job.has_errors:
        return False

    if job.rendered_image_path is None:
        job.add_error("Rendered image was not generated.")
        return False

    if not job.rendered_image_path.exists():
        job.add_error("Rendered image file does not exist.")
        return False

    return True