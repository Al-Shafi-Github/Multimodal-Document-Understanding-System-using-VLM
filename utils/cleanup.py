from pathlib import Path


def cleanup_temp_file(path: Path | None) -> None:

    if path is None:
        return

    try:

        if path.exists():
            path.unlink()

    except Exception:
        pass