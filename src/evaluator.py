"""Deterministic retrieval evaluation.

For each query we look at the top-3 retrieved doc titles and compare to
`expected_doc_titles` from the ground truth:

    hit          - every expected title is present in top-3
    partial_hit  - some but not all expected titles appear
    miss         - none of the expected titles appear

The aggregate summary reports the overall hit rate.
"""
from __future__ import annotations


def evaluate(retrieval: list[dict], queries: list[dict]) -> tuple[list[dict], dict]:
    queries_by_id = {q["query_id"]: q for q in queries}

    records: list[dict] = []
    hits = 0
    partials = 0
    misses = 0

    for r in retrieval:
        q = queries_by_id.get(r["query_id"])
        if not q:
            continue
        expected = q.get("expected_doc_titles") or []
        top3_titles = [c["doc_title"] for c in r["top_k"][:3]]

        expected_set = set(expected)
        top3_set = set(top3_titles)
        intersect = expected_set & top3_set

        if expected_set and intersect == expected_set:
            status = "hit"
            hits += 1
            explanation = f"All expected titles found in top 3 (rank(s): " \
                f"{[top3_titles.index(t) + 1 for t in expected if t in top3_titles]})"
        elif intersect:
            status = "partial_hit"
            partials += 1
            explanation = f"{len(intersect)}/{len(expected_set)} expected titles in top 3"
        else:
            status = "miss"
            misses += 1
            explanation = "No expected titles in top 3"

        records.append(
            {
                "query_id": r["query_id"],
                "expected_doc_titles": expected,
                "retrieved_doc_titles_top3": top3_titles,
                "retrieval_status": status,
                "matched_expected_title": bool(intersect),
                "explanation": explanation,
            }
        )

    total = len(records)
    summary = {
        "top3_hit_rate": (hits / total) if total else 0.0,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partials,
        "misses": misses,
    }
    return records, summary
