"""Grounding check: confirm every citation is real and supported.

For each answer:

  1. Every citation must reference a chunk that appears in the retrieval
     results for that query.
  2. The cited chunk's text must share at least one meaningful content
     token with the answer (simple lexical overlap heuristic).
"""
from __future__ import annotations

import re

from .vocab import CITATION_RE

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "if", "then", "with", "as", "at", "by", "it", "its",
}


def _content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s) if t.lower() not in _STOPWORDS and len(t) > 2}


def check_grounding(answers: list[dict], retrieval: list[dict]) -> list[dict]:
    retrieval_by_qid = {r["query_id"]: r for r in retrieval}

    out: list[dict] = []
    for ans in answers:
        qid = ans["query_id"]
        retrieved = retrieval_by_qid.get(qid, {}).get("top_k", [])
        chunk_lookup = {(c["doc_title"], c["chunk_id"]): c for c in retrieved}

        citation_results = []
        all_valid = True
        for citation in ans.get("citations", []):
            m = re.fullmatch(CITATION_RE, citation)
            if not m:
                citation_results.append(
                    {"citation": citation, "exists_in_retrieval": False,
                     "supports_answer": False, "reason": "malformed citation"}
                )
                all_valid = False
                continue
            doc_title, chunk_id = m.group(1).strip(), m.group(2).strip()
            chunk = chunk_lookup.get((doc_title, chunk_id))
            if not chunk:
                citation_results.append(
                    {"citation": citation, "exists_in_retrieval": False,
                     "supports_answer": False, "reason": "chunk not in retrieved set"}
                )
                all_valid = False
                continue
            overlap = _content_tokens(ans["answer"]) & _content_tokens(chunk["chunk_text"])
            supports = len(overlap) >= 1
            citation_results.append(
                {
                    "citation": citation,
                    "exists_in_retrieval": True,
                    "supports_answer": supports,
                    "overlap_tokens": sorted(overlap)[:10],
                }
            )
            if not supports:
                all_valid = False

        out.append(
            {
                "query_id": qid,
                "answer_label": ans.get("answer_label"),
                "num_citations": len(ans.get("citations", [])),
                "all_citations_valid": all_valid and bool(ans.get("citations")) ==
                                       (ans.get("answer_label") == "grounded_answer"),
                "citations": citation_results,
            }
        )
    return out
