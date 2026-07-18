from __future__ import annotations

from pathlib import Path

from models.file_job import FileJob

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".rtf",
    ".odt",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".heic",
    ".heif",
    ".tiff",
    ".tif",
}


def discover_files(directory: Path) -> list[FileJob]:
    """
    Recursively discover all supported files.

    Parameters
    ----------
    directory:
        Root directory.

    Returns
    -------
    list[FileJob]
    """

    jobs: list[FileJob] = []

    if not directory.exists():
        return jobs

    for file in directory.rglob("*"):

        if not file.is_file():
            continue

        extension = file.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        jobs.append(
            FileJob(
                path=file,
                extension=extension,
                size=file.stat().st_size,
            )
        )

    jobs.sort(key=lambda job: job.filename.lower())

    return jobs