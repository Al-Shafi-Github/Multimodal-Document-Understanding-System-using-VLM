from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class FileJob:


    # Discovery
    path: Path
    extension: str
    size: int

  
    # Preprocessing

    rendered_image_path: Path |None = None

    page_count: int = 1

    is_text_document: bool = False

    extracted_text: str | None = None

    
  
    # AI Response

    category: str | None = None
    organization: str | None = None
    document_title: str | None = None
    document_date: str | None = None
    document_number: str | None = None

    confidence: float | None = None

  
    # Output

    suggested_filename: str | None = None


    # Diagnostics

    processing_time_ms: float = 0.0

    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)