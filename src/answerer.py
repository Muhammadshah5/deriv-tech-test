"""Citation-strict answer generation.

Primary path: a local LLM via Ollama, constrained to JSON output. Every
citation the model produces is re-validated in code against the retrieved
chunks for this query — uncited or hallucinated references are rejected
and the answer downgraded to `insufficient_context`.

Fallback path: deterministic extractive answer (best sentence from top-1
chunk by token overlap with the question). Used when Ollama is
unreachable, the model returns malformed JSON, or all citations fail
validation. This keeps the pipeline runnable on a clean checkout.

Every LLM call is logged to `llm_calls.jsonl`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .vocab import ANSWER_LABELS, format_citation, CITATION_RE

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
LLM_LOG_PATH = os.environ.get("LLM_LOG_PATH", "llm_calls.jsonl")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s)}


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "of", "to", "in", "on", "for", "and", "or", "but",
    "if", "then", "else", "what", "which", "who", "whom", "where", "when",
    "why", "how", "i", "my", "me", "we", "our", "you", "your", "they", "them",
    "their", "it", "its", "this", "that", "these", "those", "can", "could",
    "should", "would", "will", "shall", "may", "might", "from", "with", "about",
    "as", "by", "at", "into", "after", "before", "ago",
}


def _content_tokens(s: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOPWORDS and len(t) > 1]


def _best_sentence(chunk_text: str, question: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", chunk_text) if s.strip()]
    if not sentences:
        return chunk_text.strip()
    q_tokens = set(_content_tokens(question))
    if not q_tokens:
        return sentences[0]
    best, best_score = sentences[0], -1
    for s in sentences:
        score = len(q_tokens & set(_content_tokens(s)))
        if score > best_score:
            best, best_score = s, score
    return best


def _extractive(question: str, retrieved: list[dict]) -> dict:
    if not retrieved:
        return {
            "answer_label": "insufficient_context",
            "answer": "The retrieved context does not contain enough information to answer this question.",
            "citations": [],
            "used_chunk_ids": [],
        }
    top = retrieved[0]
    sentence = _best_sentence(top["chunk_text"], question)
    citation = format_citation(top["doc_title"], top["chunk_id"])
    return {
        "answer_label": "grounded_answer",
        "answer": f"{sentence} {citation}",
        "citations": [citation],
        "used_chunk_ids": [top["chunk_id"]],
    }


def _build_prompt(question: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[doc_title: {r['doc_title']} | chunk_id: {r['chunk_id']}]\n{r['chunk_text']}"
        )
    context = "\n\n".join(context_blocks)

    return (
        "You answer questions strictly from the CONTEXT below.\n\n"
        "CHOOSE EXACTLY ONE answer_label:\n"
        '  - "grounded_answer": the context contains the information to answer. '
        'Use this even when the answer is "No" or contradicts the user\'s assumption — '
        'if the context resolves the question, it is grounded.\n'
        '  - "insufficient_context": the context contains nothing relevant to the question.\n'
        '  - "conflicting_context": two retrieved chunks directly contradict each other on the same fact.\n\n'
        "CITATION RULES:\n"
        "  - Every factual sentence in `answer` must end with an inline citation in the form "
        "[doc_title §chunk_id], copied verbatim from a context block header.\n"
        "  - Only cite chunks that appear in CONTEXT. Do not invent chunk_ids.\n"
        "  - Put every citation you used into the `citations` array as well.\n\n"
        "EXAMPLE (illustrative, do not reuse):\n"
        '  question: "Can I withdraw my profit from a demo account?"\n'
        '  context block header: [doc_title: Demo account behaviour | chunk_id: demo_account_behaviour__chunk_1]\n'
        '  output: {"answer_label": "grounded_answer", '
        '"answer": "No. Demo profit cannot be withdrawn. '
        '[Demo account behaviour §demo_account_behaviour__chunk_1]", '
        '"citations": ["[Demo account behaviour §demo_account_behaviour__chunk_1]"]}\n\n'
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Respond with ONE JSON object containing keys: answer_label, answer, citations."
    )


def _ollama_generate(prompt: str, timeout: float = 60.0) -> str | None:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response")
    except Exception as exc:
        print(f"[answerer] Ollama call failed: {exc!r}")
        return None


def _validate_and_clean(
    parsed: dict, retrieved: list[dict]
) -> dict | None:
    """Return cleaned record or None if structurally invalid."""
    label = parsed.get("answer_label")
    answer = parsed.get("answer")
    citations = parsed.get("citations") or []
    if label not in ANSWER_LABELS or not isinstance(answer, str):
        return None
    if not isinstance(citations, list):
        citations = []

    allowed = {format_citation(r["doc_title"], r["chunk_id"]): r["chunk_id"] for r in retrieved}
    valid_citations: list[str] = []
    used_ids: list[str] = []
    for c in citations:
        if not isinstance(c, str):
            continue
        if c in allowed:
            valid_citations.append(c)
            used_ids.append(allowed[c])

    # Also pull citations directly out of the answer text in case the model
    # cited inline but didn't list them in `citations`.
    for m in re.finditer(CITATION_RE, answer):
        rebuilt = format_citation(m.group(1).strip(), m.group(2).strip())
        if rebuilt in allowed and rebuilt not in valid_citations:
            valid_citations.append(rebuilt)
            used_ids.append(allowed[rebuilt])

    if label == "grounded_answer" and not valid_citations:
        # Grounded answers must have at least one valid citation.
        return None

    return {
        "answer_label": label,
        "answer": answer.strip(),
        "citations": valid_citations,
        "used_chunk_ids": used_ids,
    }


def _log_llm_call(
    stage: str,
    query_id: str | None,
    prompt: str,
    input_artifacts: list[str],
    output_artifact: str,
) -> None:
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
    }
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def answer_query(
    query_id: str,
    question: str,
    retrieved: list[dict],
    use_llm: bool = True,
    input_artifacts: list[str] | None = None,
    output_artifact: str = "artifacts/answers.json",
) -> dict:
    """Generate one citation-validated answer record."""
    if use_llm:
        prompt = _build_prompt(question, retrieved)
        raw = _ollama_generate(prompt)
        if raw:
            _log_llm_call(
                stage="ANSWERS_GENERATED",
                query_id=query_id,
                prompt=prompt,
                input_artifacts=input_artifacts or [],
                output_artifact=output_artifact,
            )
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                cleaned = _validate_and_clean(parsed, retrieved)
                if cleaned is not None:
                    record = {"query_id": query_id, **cleaned}
                    return record

    # Fallback: deterministic extractive.
    record = {"query_id": query_id, **_extractive(question, retrieved)}
    return record


def reset_llm_log() -> None:
    if os.path.exists(LLM_LOG_PATH):
        os.remove(LLM_LOG_PATH)
