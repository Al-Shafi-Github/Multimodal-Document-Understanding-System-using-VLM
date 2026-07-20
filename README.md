# Multimodal-Document-Understanding-System-using-VLM

A privacy-first, fully offline pipeline that classifies and renames local files (documents, receipts, images) using a local vision-language model. No cloud APIs are used at any point during inference.

---

## 1. Overview

This project processes a batch of mixed files — PDFs, Office documents, images, spreadsheets — and produces clean, structured, hierarchical filenames based on the actual visual and textual content of each file, without relying on the original filename or extension. It is built as a Python batch script that drives a locally hosted vision-language model through an OpenAI-compatible API, not as a web service.

Example transformation:

```
Screenshot 2026-07-13 130349.png  ->  bank_statement_brac_2023-11.png
images.jpg                        ->  driving_license_commercial_drivers_pennsylvania_2017-08-06.jpg
1706.03762v7.pdf                  ->  research_paper_attention_is_all_you_need_2017.pdf
```

---

## 2. Architecture

### 2.1 Pipeline stages

```
Discovery  ->  Preprocessing  ->  AI Inference  ->  Confidence Gate  ->  Filename Build  ->  Safe Move
```

1. **Discovery** (`services/discovery.py`) — walks `data/incoming/` and collects every file with a supported extension into a `FileJob`.
2. **Preprocessing** (`services/preprocess.py`) — normalizes each file into whatever the model needs:
   - Images (JPG, PNG, GIF, BMP, WEBP, TIFF, HEIC/HEIF) are resized and re-encoded to PNG.
   - PDFs and SVGs are rasterized to an image entirely in memory using PyMuPDF (`fitz`) — no intermediate disk round-trip.
   - DOCX and ODT have their text extracted directly (via `python-docx` and `odfpy` respectively) rather than being treated as plain text, since both are zip/XML containers, not flat text.
   - RTF is converted to plain text via `striprtf` to strip control markup before it reaches the model.
   - CSV is read and capped to the first 40 rows.
   - All extracted text is capped at a fixed character budget (see Section 4) to stay within the model's context window.
3. **AI Inference** (`services/inference.py`) — sends either the rendered image or the extracted text to the local model via an OpenAI-compatible chat completion request, with `response_format` set to a strict JSON schema derived directly from the `FileUnderstanding` Pydantic model.
4. **Confidence Gate** (`app.py`) — files below the configured confidence threshold are quarantined to `data/failed/` rather than renamed. A file that is flagged as `category=other` with very low confidence is logged separately as a probable prompt-injection attempt (see Section 6).
5. **Filename Build** (`services/naming.py`) — deterministic, non-LLM string composition from the extracted fields. This is a deliberate architectural choice, explained below.
6. **Safe Move** (`utils/safe_move.py`) — every file move (success or failure) goes through one shared function that auto-suffixes (`_v2`, `_v3`, ...) on any destination collision, so re-running the pipeline on the same input, or two files independently producing the same generated name, never crashes the run.

### 2.2 Why the model never writes the filename directly

The AI model's only job is fact extraction: category, subject, organization, date, document number, confidence. Turning those facts into a filename is handled entirely by deterministic Python string logic in `services/naming.py`, never by a second model call or free-text generation.

This matters for reliability: an LLM, especially a 3B model, cannot be guaranteed to apply formatting rules (lowercase, character limits, deduplication, filesystem-safe characters) with 100% consistency on every single run. Deterministic code can. The model is used for the part it is actually good at — visual and semantic understanding — and code is used for the part that requires guarantees.

### 2.3 Category-aware naming templates

Different document categories carry different meaningful fields. A research paper has no meaningful "organization" field once the title is known; a chart has no meaningful "document number"; a driving license's most identifying information is the issuing region and expiry date, not a free-text subject line. Rather than forcing one flat `category_subject_organization_date_identifier` template onto every file type, `naming.py` defines a per-category field template:

```python
CATEGORY_TEMPLATES = {
    "research_paper": ["subject", "date"],
    "book":           ["subject", "organization", "date"],
    "chart":          ["subject", "organization", "date"],
    "graph":          ["subject", "organization", "date"],
}
DEFAULT_TEMPLATE = ["subject", "organization", "date", "identifier"]
```

Low-information capture-method categories (`screenshot`, `photograph`, `other`, `unknown`) omit the category name from the filename itself, since it adds no information beyond what the destination folder/classification already conveys.

### 2.4 Model and engine choice

- **Model:** `qwen2.5-vl-3b-instruct`, Q4_K_M quantization (~3.3 GB on disk).
- **Engine:** LM Studio, exposing an OpenAI-compatible REST API (`/v1/chat/completions`) on `localhost`.
- **Why this model:** it is one of very few sub-4B-parameter models with genuine vision-language capability (not just OCR bolted onto a text model), small enough to load and run entirely on CPU within the 8 GB memory ceiling, and fully offline once downloaded.
- **Why this engine:** LM Studio provides an OpenAI-compatible endpoint out of the box, which means the Python backend uses the standard `openai` client library rather than a bespoke protocol — this keeps the integration code simple and makes it trivial to swap engines (e.g. to Ollama) by changing only the base URL.

---

## 3. Setup Instructions

### 3.1 Prerequisites

- Python 3.11+
- LM Studio (or another OpenAI-compatible local inference server)
- The `qwen2.5-vl-3b-instruct` model (GGUF, Q4_K_M or similar quantization) downloaded into LM Studio

### 3.2 Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Core dependencies used by this project:

| Package | Purpose |
|---|---|
| `openai` | Client for the local OpenAI-compatible inference API |
| `pydantic` / `pydantic-settings` | Schema definition, strict JSON validation, configuration |
| `loguru` | Structured logging to console and file |
| `PyMuPDF` (`fitz`) | PDF and SVG rasterization, entirely in-memory |
| `Pillow` | Image loading, normalization, resizing |
| `pillow-heif` | HEIC/HEIF decoding (registered as a Pillow plugin) |
| `python-docx` | DOCX text extraction |
| `odfpy` | ODT (OpenDocument Text) extraction |
| `striprtf` | RTF-to-plain-text conversion |

### 3.3 Start the local inference engine

1. Open LM Studio and load `qwen2.5-vl-3b-instruct`.
2. Start the local server (default: `http://127.0.0.1:1234`).
3. Recommended server settings for CPU-only, memory-constrained hardware:
   - **Max Concurrency: 1** — this pipeline processes files strictly sequentially; higher concurrency reserves resources for requests that never arrive.
   - **CPU Thread Pool Size:** match your physical core count.
   - **Flash Attention:** on, if supported by your build.
   - **K/V cache quantization:** test before enabling — see Section 5.

### 3.4 Configure the application

Copy `.env.example` to `.env` and adjust as needed:

```
AI_ENGINE=lmstudio
AI_HOST=http://127.0.0.1:1234
MODEL_NAME=qwen2.5-vl-3b-instruct
CONFIDENCE_THRESHOLD=0.50
```

### 3.5 Run

Place files to be processed into `data/incoming/`, then:

```bash
python app.py
```

Results:
- Successfully classified files are moved to `data/renamed/`.
- Files that fail preprocessing, fail inference, or fall below the confidence threshold are moved to `data/failed/` — the original file is never deleted or left in a broken state.
- A run summary (files found, processed, renamed, failed, average confidence) is printed at the end and written to `logs/app.log`.

---

## 4. Supported Formats

| Type | Extensions |
|---|---|
| Documents | PDF, DOC/DOCX, TXT, MD, CSV, RTF, ODT |
| Images | JPG, JPEG, PNG, GIF, WEBP, BMP, SVG, HEIC, HEIF, TIFF, TIF |

All formats route through one of two paths: the vision path (rendered to a normalized PNG) or the text path (extracted and capped to a fixed character budget). Every text-extraction function shares one truncation helper so the context-window limit is enforced consistently, rather than risking a request that exceeds the model's context size on a large CSV or text file.

---

## 5. Optimization Strategies

### 5.1 Hardware context

This pipeline was developed and benchmarked on CPU-only, consumer-grade hardware: a Ryzen 7 7700 with a single 16 GB RAM stick (single-channel, 6000 MHz CL36), no discrete GPU. The assignment's 8 GB constraint is phrased as "RAM/VRAM," which implicitly assumes GPU-accelerated inference is available. On this hardware, all inference runs on CPU, and single-channel memory provides roughly half the effective bandwidth of a dual-channel configuration — a meaningful, measurable constraint on CPU prompt-evaluation throughput, which is memory-bandwidth-bound. This is documented here rather than hidden, since it materially affects the achievable per-file latency relative to a GPU-equipped target.

### 5.2 Token budget reduction

Two independent levers were used to reduce prompt size, since prompt evaluation (not generation) dominates per-file latency on CPU:

- **Image resolution:** the maximum image dimension sent to the model was reduced from 1536px to 896px. For a patch-based vision-language model, image token count scales roughly with resolution, so this had a substantial effect on total prompt tokens with no measurable loss in classification accuracy on the golden dataset.
- **System prompt length:** the system prompt was reduced from an early draft of roughly 1900 tokens (verbose category list, six few-shot examples, redundant "do not explain" boilerplate that duplicates what the strict JSON schema already guarantees) to a leaner instruction set with three few-shot examples.

### 5.3 Text extraction caps

All text-extraction paths (CSV, TXT/MD/RTF, DOCX, ODT) share a single character cap before content is sent to the model, preventing a large input file (e.g. a multi-thousand-row CSV) from exceeding the model's context window and causing a hard request failure.

### 5.4 Engine-level tuning

- Max Concurrency set to 1, matching the pipeline's strictly sequential request pattern.
- K/V cache quantization was evaluated and found to measurably reduce classification accuracy on this model/quantization combination (small, ambiguous visual categories were more likely to be misclassified) — the small memory-bandwidth benefit did not justify the accuracy cost, so it is left disabled.

### 5.5 Confidence calibration

Model-reported confidence values were found to anchor strongly to the values shown in the system prompt's few-shot examples (an early version of the prompt used near-identical confidence values across all examples, and the model reproduced that pattern rather than reasoning per-file). This was corrected by using visibly different confidence values across the few-shot examples and by giving the model an explicit instruction to vary confidence based on document legibility. Where the inference server exposes token-level log-probabilities, these can additionally be used to compute a confidence score independent of the model's self-reported value.

### 5.6 Measured results

Latency and accuracy figures from the golden dataset run are recorded in `logs/app.log` from the most recent full run. On this hardware, achieved per-file latency does not meet the 3-second target; the gap is attributable to the CPU-only/single-channel-RAM constraint described in Section 5.1, not to unoptimized code paths, given that the two largest available levers (image resolution and prompt length) have both been applied.

---

## 6. Security and Robustness

### 6.1 Prompt injection resistance

The system prompt explicitly instructs the model to treat all file content — including barcodes, QR codes, and encoded text (morse, hex, base64) — as data to describe, never as instructions to follow. A file whose content attempts to redirect the model's own behavior, category, or confidence score is treated as suspicious and classified as `category=other` with a low confidence score, which routes it to `data/failed/` via the confidence gate rather than being silently renamed. This is logged with a distinct `SECURITY:` warning message, separate from an ordinary low-confidence classification, so it is identifiable in the run log.

The security instruction is scoped specifically to text that addresses the model directly (e.g. "ignore previous instructions"), not to ordinary imperative writing addressed to a human reader (legal forms, manuals, terms of service) — an earlier, broader version of this rule caused legitimate instructional documents to be misclassified as suspicious, which was corrected during testing.

### 6.2 Filesystem safety (path traversal)

Two independent guarantees prevent file content from ever influencing the destination path in an unsafe way:

- `category` is a closed Python `Enum`, enforced by the model's strict JSON schema at the token-generation level — the model cannot output a category value outside the fixed list, so it cannot inject a path segment into the destination folder.
- Every extracted text field is passed through a slugification step that strips all characters except word characters, whitespace, and hyphens before it can become part of a filename — no `/`, `\`, `..`, quotes, or shell/SQL metacharacters can survive into the final filename.

This pipeline has no database and does not construct SQL queries; the filesystem-path-injection defenses above are the applicable equivalent for a file-renaming system.

### 6.3 Error handling

The pipeline is designed to fail safely at every stage rather than crash:

- A corrupted, unreadable, or unsupported file is caught during preprocessing and routed to `data/failed/` with the error recorded in the log.
- A malformed or non-schema-conforming model response raises a validation error that is caught and logged; the file is quarantined rather than the run terminating.
- A connection failure to the local inference engine (including a transient failure caused by the engine reloading the model under memory pressure) is caught and logged per file; the run continues to the next file.
- Every file move — success or failure path — goes through a single shared function that automatically de-duplicates destination filenames rather than raising a filesystem error, so re-processing the same input file, or two files independently producing the same generated name, cannot crash the pipeline.

A full run consistently completes and prints a run summary even when individual files fail for any of the above reasons, which is the core requirement of this phase of the assignment: catch and handle errors gracefully rather than allow one bad file to terminate the entire batch.

---

## 7. Known Limitations

These are constraints of the selected model size and quantization level, or of the available hardware, rather than defects in the pipeline code:

- **Per-file latency does not reach the 3-second target on CPU-only hardware.** See Section 5.1 and 5.6.
- **Fine visual distinctions between similar categories** (for example, a chart-heavy page versus a research paper containing an embedded chart) are not always resolved correctly. A 3B-parameter model at 4-bit quantization has a genuine ceiling on this kind of fine-grained visual discrimination that prompt engineering can reduce but not eliminate.
- **Confidence scores, even when calibrated, are the model's own self-assessment** and are not a rigorously calibrated probability — they should be read as a relative signal (this file was less legible than that one), not an exact statistical measure.
- **Minor run-to-run variance is possible on ambiguous, low-signal files** (for example, a page of body text with no title, letterhead, or identifying number) due to floating-point non-determinism in multi-threaded CPU inference. This is most visible on files that are genuinely difficult to classify even for a human skimming the same page.

---

## 8. Project Structure

```
datacrunch-ai-renamer/
├── app.py                     # Entry point — orchestrates the full pipeline
├── config.py                  # Settings (paths, model name, thresholds) via pydantic-settings           # Custom exception hierarchy for typed error handling
├── requirements.txt
├── .env.example
├── data/
│   ├── incoming/               # Drop files here to be processed
│   ├── renamed/                # Successfully classified output
│   ├── failed/                 # Quarantined files (low confidence, errors, security flags)
│   ├── golden_dataset/         # Labeled test set for accuracy evaluation
│   └── temp/                   # Scratch space for rendered images
├── models/
│   ├── file_job.py              # Per-file state as it moves through the pipeline
│   ├── file_understanding.py    # The strict output schema (Category enum, fields)
│   └── run_statistics.py
├── prompts/
│   └── system_prompt.txt        # The model's instructions, categories, and few-shot examples
├── services/
│   ├── discovery.py             # Finds supported files in data/incoming/
│   ├── preprocess.py            # Normalizes every format into image-or-text
│   ├── inference.py              # Calls the local model, validates the response
│   ├── naming.py                 # Deterministic filename construction
│   └── evaluator.py              # Golden-dataset accuracy scoring
├── utils/
│   ├── image_utils.py            # Image loading, resizing, PDF/SVG rasterization
│   ├── safe_move.py               # Collision-safe file moves used throughout app.py
│   ├── hashing.py
│   └── cleanup.py
└── logs/
    └── app.log
```
