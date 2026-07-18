from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Optional
from unicodedata import category

from models.file_understanding import (
    FileUnderstanding,
    SENSITIVE_CATEGORIES,
)

MAX_LENGTH = 100
BLACKLIST = {"file", "document", "image", "scan", "copy", "new", "final"}

MAX_FIELD_WORDS = 8
# FIELD_STOPWORDS = {"a", "an", "the", "of", "for", "in", "on", "to", "with"}
NO_CATEGORY_PREFIX = {"screenshot", "photograph", "other", "unknown"}

# Which facts actually matter per category. "subject" = primary_subject,

CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "research_paper": ["subject", "date"],
    "book": ["subject", "organization", "date"],
    "chart": ["subject", "organization", "date"],
    "graph": ["subject", "organization", "date"],
    "resume": ["organization", "date"],
    "business_card": ["organization", "date"],
}
DEFAULT_TEMPLATE = ["subject", "organization", "date", "identifier"]

ORG_MAX_WORDS_OVERRIDE: dict[str, int] = {
    "resume": 8,
    "business_card": 8,
}


def _limit_words(value: Optional[str], max_words: int = MAX_FIELD_WORDS) -> Optional[str]:
    if not value:
        return None
    value = re.sub(r"\s+-\s+", " ", value) 
    words = value.split()
    return " ".join(words[:max_words]) if words else None

def _slugify(value: Optional[str]) -> Optional[str]:
    """lowercase, spaces->underscores, strip punctuation except _/-"""
    if not value:
        return None
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)   # strip punctuation
    value = re.sub(r"\s+", "_", value)        # spaces -> underscores
    value = re.sub(r"_+", "_", value).strip("_")
    return value or None


def _dedupe_words(parts: list[str]) -> list[str]:
    """Drop a later field entirely if its words are already fully covered
    by earlier fields, drop repeated tokens WITHIN a field too (e.g. an
    organization string like "Google Brain, Google Research" repeating
    "Google"), and drop repeated tokens across fields."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        words = [w for w in part.split("_") if w]
        kept: list[str] = []
        local_seen: set[str] = set()
        for w in words:
            if w in seen or w in local_seen:
                continue
            kept.append(w)
            local_seen.add(w)
        if not kept:
            continue  # this field added nothing new -> drop it
        seen.update(kept)
        out.append("_".join(kept))
    return out


def _identifier_token(result: FileUnderstanding) -> Optional[str]:
    identifier = getattr(result, "document_number", None)
    if not identifier:
        return None

    digits_only = re.sub(r"[\s-]", "", identifier)
    if len(digits_only) > 20 and digits_only.isdigit():
        return None  # almost certainly a barcode/MRZ misread

    if result.category in SENSITIVE_CATEGORIES:
        digest = hashlib.sha256(identifier.encode()).hexdigest()[:8]
        return f"ref{digest}"
    return _slugify(identifier)


def build_filename(result: FileUnderstanding, original_path: Path) -> str:
    category = result.category.value  # already a safe closed-set token

    org_max_words = ORG_MAX_WORDS_OVERRIDE.get(category, 4)

    field_values: dict[str, Optional[str]] = {
        "subject": _slugify(_limit_words(result.primary_subject, max_words=8)),
        "organization": _slugify(_limit_words(result.organization, max_words=org_max_words)),
        "date": result.document_date.replace("-", "") if result.document_date else None,  # 2026-05 -> 202605, matches reference format and avoids ambiguity with the hyphen field-separator
        "identifier": _identifier_token(result),
    }

    template = CATEGORY_TEMPLATES.get(category, DEFAULT_TEMPLATE)

    prefix = [] if category in NO_CATEGORY_PREFIX else [category]
    fields = prefix + [
        field_values[key] for key in template if field_values.get(key)
    ]
    fields = _dedupe_words(fields)

    # Drop blacklisted filler words unless a field would otherwise be empty.
    # fields are underscore-joined internally at this point (hyphens
    # are only introduced later, when fields are joined together below), so
    # we must split on "_" here, not "-", or the blacklist check silently
    # never fires on multi-word fields.
    cleaned_fields = []
    for f in fields:
        words = f.split("_")
        kept = [w for w in words if w not in BLACKLIST]
        cleaned_fields.append("_".join(kept) if kept else f)

    name = "-".join(cleaned_fields) if cleaned_fields else category
    ext = original_path.suffix.lower().lstrip(".")
    full = f"{name}.{ext}"

    if len(full) > MAX_LENGTH:
        keep = MAX_LENGTH - len(ext) - 1
        full = f"{name[:keep]}.{ext}"

    return full


FILENAME_PATTERN = re.compile(r"^[a-z0-9_-]+\.[a-z0-9]+$")


def validate_filename(name: str) -> bool:
    """Last guardrail before touching the filesystem. build_filename() is
    deterministic template code, so this mainly exists as a regression
    check and an explicit contract for the backend team."""
    return bool(FILENAME_PATTERN.match(name)) and len(name) <= MAX_LENGTH