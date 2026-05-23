ANSWER_LABELS = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}

CITATION_RE = r"\[([^\[\]]+?) §([a-zA-Z0-9_\-]+)\]"


def format_citation(doc_title: str, chunk_id: str) -> str:
    return f"[{doc_title} §{chunk_id}]"
