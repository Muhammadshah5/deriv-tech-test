"""Validate that the generated artifacts meet the spec.

Run after `python main.py`. Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import re
import sys

from src.vocab import ANSWER_LABELS, RETRIEVAL_STATUSES, CITATION_RE

ARTIFACTS = {
    "chunks": "artifacts/chunks.json",
    "retrieval": "artifacts/retrieval.json",
    "answers": "artifacts/answers.json",
    "eval": "artifacts/eval.json",
}
OPTIONAL_ARTIFACTS = {
    "grounding_check": "artifacts/grounding_check.json",
    "chunking_comparison": "artifacts/chunking_comparison.json",
}


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"  FAIL: {msg}")


def _ok(msg: str) -> None:
    print(f"  OK:   {msg}")


def _load_json(path: str, errors: list[str]):
    if not os.path.exists(path):
        _fail(f"missing artifact: {path}", errors)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path}: {exc}", errors)
        return None


def main() -> int:
    errors: list[str] = []

    print("== validating artifacts ==")
    chunks = _load_json(ARTIFACTS["chunks"], errors)
    retrieval = _load_json(ARTIFACTS["retrieval"], errors)
    answers = _load_json(ARTIFACTS["answers"], errors)
    eval_doc = _load_json(ARTIFACTS["eval"], errors)
    queries = _load_json("queries.json", errors)

    if any(x is None for x in (chunks, retrieval, answers, eval_doc, queries)):
        print(f"\nFAILED with {len(errors)} error(s).")
        return 1

    for name, path in ARTIFACTS.items():
        if os.path.exists(path):
            _ok(f"artifact present: {path}")

    # All queries processed
    q_ids = {q["query_id"] for q in queries}
    r_ids = {r["query_id"] for r in retrieval}
    a_ids = {a["query_id"] for a in answers}
    if q_ids != r_ids:
        _fail(f"retrieval missing queries: {q_ids - r_ids}", errors)
    else:
        _ok("retrieval covers all queries")
    if q_ids != a_ids:
        _fail(f"answers missing queries: {q_ids - a_ids}", errors)
    else:
        _ok("answers cover all queries")

    # Each query has >= 3 retrieved chunks with numeric scores
    for r in retrieval:
        if len(r.get("top_k", [])) < 3:
            _fail(f"{r['query_id']}: fewer than 3 retrieved chunks", errors)
            continue
        for c in r["top_k"]:
            if not isinstance(c.get("score"), (int, float)):
                _fail(f"{r['query_id']}: non-numeric score for {c.get('chunk_id')}", errors)
    _ok("retrieval has >=3 chunks/query with numeric scores")

    # Build allowed-citation set per query
    allowed_by_qid: dict[str, set[str]] = {}
    for r in retrieval:
        allowed_by_qid[r["query_id"]] = {
            f"[{c['doc_title']} §{c['chunk_id']}]" for c in r["top_k"]
        }

    # Answer label vocab + citations
    for a in answers:
        label = a.get("answer_label")
        if label not in ANSWER_LABELS:
            _fail(f"{a['query_id']}: invalid answer_label '{label}'", errors)
            continue
        citations = a.get("citations", [])
        if label == "grounded_answer" and not citations:
            _fail(f"{a['query_id']}: grounded_answer has no citations", errors)
        for c in citations:
            if not re.fullmatch(CITATION_RE, c):
                _fail(f"{a['query_id']}: malformed citation '{c}'", errors)
                continue
            if c not in allowed_by_qid.get(a["query_id"], set()):
                _fail(f"{a['query_id']}: citation refers to non-retrieved chunk: {c}", errors)
    _ok("answer labels + citations validated")

    # Eval statuses + aggregate summary
    per_query = eval_doc.get("per_query") if isinstance(eval_doc, dict) else None
    summary = eval_doc.get("summary") if isinstance(eval_doc, dict) else None
    if not isinstance(per_query, list) or not per_query:
        _fail("eval.json missing per_query records", errors)
    else:
        for e in per_query:
            if e.get("retrieval_status") not in RETRIEVAL_STATUSES:
                _fail(f"{e.get('query_id')}: invalid retrieval_status", errors)
        _ok("evaluation statuses use controlled vocabulary")
    if not isinstance(summary, dict):
        _fail("eval.json missing aggregate summary", errors)
    else:
        for key in ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"):
            if key not in summary:
                _fail(f"eval summary missing key: {key}", errors)
        _ok("evaluation summary present with required keys")

    for name, path in OPTIONAL_ARTIFACTS.items():
        if os.path.exists(path):
            _ok(f"optional artifact present: {path}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s).")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
