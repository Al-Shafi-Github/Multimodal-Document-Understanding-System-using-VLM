from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import settings
from models.file_understanding import FileUnderstanding


client = OpenAI(
    base_url=f"{settings.AI_HOST}/v1",
    api_key=settings.AI_API_KEY,
)


def _load_prompt(name: str) -> str:
    return Path("prompts", name).read_text(encoding="utf-8")


def _image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


DOCUMENT_SCHEMA = FileUnderstanding.model_json_schema()

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "document_result",
        "strict": True,
        "schema": DOCUMENT_SCHEMA,
    },
}


def _confidence_from_logprobs(response) -> Optional[float]:
    """
    Real confidence from the model's own token probabilities, instead of
    trusting a self-reported number in the JSON body (which a 3B model
    tends to anchor to whatever the few-shot examples showed).

    Returns None if the LM Studio/llama.cpp build in use doesn't return
    logprobs for this endpoint — callers should fall back to the
    self-reported schema field in that case.
    """
    choice = response.choices[0]
    logprobs_obj = getattr(choice, "logprobs", None)
    content = getattr(logprobs_obj, "content", None) if logprobs_obj else None

    if not content:
        return None

    token_logprobs = [t.logprob for t in content if t.logprob is not None]
    if not token_logprobs:
        return None

    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)  # log-prob -> probability


def _apply_real_confidence(result: FileUnderstanding, response) -> FileUnderstanding:
    computed = _confidence_from_logprobs(response)
    if computed is None:
        return result  # fall back to the self-reported field, unchanged
    return result.model_copy(update={"confidence": round(computed, 2)})


def analyze_image(image: Path) -> FileUnderstanding:
    """
    Vision pipeline. Used for: jpg, png, pdf, scans, passport, receipts,
    screenshots, charts.
    """

    encoded = _image_to_base64(image)

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        temperature=0,
        max_tokens=250,
        logprobs=True,
        response_format=JSON_SCHEMA,
        messages=[
            {
                "role": "system",
                "content": _load_prompt("system_prompt.txt"),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the attached document and return the JSON object per schema.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("Model response contains no content")

    result = FileUnderstanding.model_validate_json(raw)
    return _apply_real_confidence(result, response)


def analyze_text(text: str) -> FileUnderstanding:
    """
    Text pipeline. Used for: txt, md, rtf, csv, docx.
    """

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        temperature=0,
        max_tokens=250,
        logprobs=True,
        response_format=JSON_SCHEMA,
        messages=[
            {
                "role": "system",
                "content": _load_prompt("system_prompt.txt"),
            },
            {
                "role": "user",
                "content": f"Analyze the following document and return the JSON object per schema.\n\n{text}",
            },
        ],
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("Model response contains no content")

    result = FileUnderstanding.model_validate_json(raw)
    return _apply_real_confidence(result, response)