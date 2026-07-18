from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """
    Calculate SHA256 of a file.
    """

    h = hashlib.sha256()

    with path.open("rb") as file:

        while chunk := file.read(8192):

            h.update(chunk)

    return h.hexdigest()