import io
import json
from typing import List, Dict, Any

from openai import OpenAI

client = OpenAI()


def _read_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required to read PDF files.")

    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts).strip()


def _read_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required to read DOCX files.")

    doc = Document(io.BytesIO(file_bytes))
    texts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(texts).strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Read uploaded literature file and return extracted text.
    Supported:
    - pdf
    - txt
    - md
    - docx
    """
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if file_name.endswith(".pdf"):
        return _read_pdf(file_bytes)

    if file_name.endswith(".docx"):
        return _read_docx(file_bytes)

    if file_name.endswith(".txt") or file_name.endswith(".md"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _truncate_text(text: str, max_chars: int = 35000) -> str:
    """
    Keep prompt size manageable.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def summarize_literature_text(text: str, paper_name: str) -> Dict[str, Any]:
    """
    Use LLM to summarize a paper into a standardized JSON structure.
    """
    text = _truncate_text(text)

    prompt = f"""
Read the following paper content and return ONLY valid JSON.

Goal:
Produce a standardized literature summary so that outputs from different papers
share the same structure.

Rules:
- Output valid JSON only
- Do not use markdown
- If a field is unknown, use null
- Do not invent values not supported by the text
- Keep summary concise and evidence-based

Return exactly this schema:

{{
  "paper_name": "{paper_name}",
  "title": null,
  "research_question": null,
  "population": null,
  "study_design": null,
  "intervention_or_exposure": null,
  "comparison": null,
  "outcome": null,
  "main_finding": null,
  "effect_type": null,
  "effect_value": null,
  "ci_lower": null,
  "ci_upper": null,
  "p_value": null,
  "key_terms": [],
  "plain_summary": null
}}

Paper text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are an academic literature extraction assistant. "
                    "Return valid JSON only. Be faithful to the provided paper text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content
    return json.loads(content)


def summarize_uploaded_literature_files(uploaded_files) -> List[Dict[str, Any]]:
    """
    Process multiple uploaded literature files and return standardized summaries.
    """
    results = []

    for f in uploaded_files:
        try:
            text = extract_text_from_uploaded_file(f)
            if not text.strip():
                results.append({
                    "paper_name": f.name,
                    "error": "No readable text extracted from file."
                })
                continue

            summary = summarize_literature_text(text, f.name)
            results.append(summary)

        except Exception as e:
            results.append({
                "paper_name": f.name,
                "error": str(e)
            })

    return results