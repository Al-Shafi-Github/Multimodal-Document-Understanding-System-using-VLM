from __future__ import annotations

from pathlib import Path

from models.file_understanding import FileUnderstanding


def evaluate_prediction(
    prediction: FileUnderstanding,
    original: Path,
) -> dict:
    """
    Compare prediction with the expected label.

    Expected layout:

    golden_dataset/
        invoice/
        receipt/
        passport/

    """

    expected = original.parent.name.lower()

    predicted = prediction.category.value.lower()

    return {
        "expected": expected,
        "predicted": predicted,
        "correct": expected == predicted,
        "confidence": prediction.confidence,
    }