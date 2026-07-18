from __future__ import annotations
from pathlib import Path


def safe_move(source: Path, destination: Path) -> Path:
    """
    Rename `source` to `destination`. If destination already exists,
    auto-uniquify with _v2, _v3, ... instead of raising.

    Used for every file move in the pipeline (renamed/ and failed/)
    so that reprocessing the same file, or two files landing on the
    same generated name, never crashes the run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        counter = 2
        candidate = destination
        while candidate.exists():
            candidate = destination.with_stem(f"{destination.stem}_v{counter}")
            counter += 1
        destination = candidate

    source.rename(destination)
    return destination