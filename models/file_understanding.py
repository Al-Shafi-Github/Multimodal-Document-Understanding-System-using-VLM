from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class Category(str, Enum):

    INVOICE = "invoice"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"
    DRIVING_LICENSE = "driving_license"
    PASSPORT = "passport"
    MEETING_NOTES = "meeting_notes"
    SCREENSHOT = "screenshot"
    GITHUB_README = "github_readme"
    CSV = "csv"
    MARKDOWN = "markdown"
    RESEARCH_PAPER = "research_paper"
    BOOK = "book"
    RESUME = "resume"
    TAX_FORM = "tax_form"
    INSURANCE = "insurance"
    UTILITY_BILL = "utility_bill"
    MEDICAL_REPORT = "medical_report"
    EMAIL_SCREENSHOT = "email_screenshot"
    GRAPH = "graph"
    CHART = "chart"
    BUSINESS_CARD = "business_card"
    CONTRACT = "contract"
    PHOTOGRAPH = "photograph"
    LEGAL_FORM = "legal_form"
    OTHER = "other"
    UNKNOWN = "unknown"


SENSITIVE_CATEGORIES = {
    Category.PASSPORT,
    Category.DRIVING_LICENSE,
    Category.BANK_STATEMENT,
    Category.MEDICAL_REPORT,
    Category.INSURANCE,
    Category.UTILITY_BILL,
}


_DATE_PATTERN = re.compile(
    r"^\d{4}(-\d{2})?(-\d{2})?$"
)


class FileUnderstanding(BaseModel):

    model_config = ConfigDict(extra="forbid")


    # Required
  
    category: Category

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


    # Optional extracted fields

    primary_subject: Optional[str] = Field(
        default=None,
        max_length=60,
    )

    organization: Optional[str] = Field(
        default=None,
        max_length=60,
    )

    # document_title: Optional[str] = Field(
    #     default=None,
    #     max_length=100,
    # )

    document_date: Optional[str] = Field(
        default=None,
    )

    document_number: Optional[str] = Field(
        default=None,
        max_length=40,
    )

  
    # Validators

    @field_validator("document_date")
    @classmethod
    def validate_date(cls, value):

        if value is None:
            return value

        if not _DATE_PATTERN.match(value):
            raise ValueError(
                "Date must be YYYY, YYYY-MM or YYYY-MM-DD"
            )

        return value

    @field_validator(
        "primary_subject",
        "organization",
        "document_title",
        "document_number",
    )
    @classmethod
    def sanitize(cls, value):

        if value is None:
            return value

        value = re.sub(
            r"[\r\n\t]+",
            " ",
            value,
        )

        value = "".join(
            c for c in value
            if c.isprintable()
        )

        value = value.strip()

        return value or None