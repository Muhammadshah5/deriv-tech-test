"""End-to-end pipeline runner.

Usage:
    python main.py                       # paragraph chunking (default)
    python main.py --strategy fixed      # fixed-size chunking
    python main.py --no-llm              # force extractive fallback path

Outputs:
    artifacts/chunks.json
    artifacts/retrieval.json
    artifacts/answers.json
    artifacts/eval.json
    artifacts/grounding_check.json
    llm_calls.jsonl  (only when an LLM is used)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src import answerer, evaluator, grounding
from src.chunker import STRATEGIES, load_documents
from src.pipeline import PipelineState
from src.retriever import HybridRetriever

KB_DIR = "kb"
QUERIES_PATH = "queries.json"
ARTIFACTS_DIR = "artifacts"


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run(strategy: str = "paragraph", use_llm: bool = True,
        artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    state = PipelineState()

    # 1. Load documents
    docs = load_documents(KB_DIR)
    if not docs:
        raise RuntimeError(f"No .txt documents found in {KB_DIR}/")
    state.advance_to("DOCUMENTS_LOADED")

    # 2. Chunk
    chunk_fn = STRATEGIES[strategy]
    chunks = [c.to_dict() for c in chunk_fn(docs)]
    chunks_path = os.path.join(artifacts_dir, "chunks.json")
    _write_json(chunks_path, chunks)
    state.advance_to("DOCUMENTS_CHUNKED")

    # 3. Build index
    retriever = HybridRetriever(chunks)
    state.advance_to("INDEX_BUILT")

    # 4. Retrieve
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    retrieval = []
    for q in queries:
        top_k = retriever.search(q["question"], top_k=3)
        retrieval.append({"query_id": q["query_id"], "question": q["question"], "top_k": top_k})
    retrieval_path = os.path.join(artifacts_dir, "retrieval.json")
    _write_json(retrieval_path, retrieval)
    state.advance_to("RETRIEVAL_COMPLETE")

    # 5. Generate answers (pipeline guarantees retrieval is complete first)
    state.require("RETRIEVAL_COMPLETE")
    answerer.reset_llm_log()
    answers_path = os.path.join(artifacts_dir, "answers.json")
    answers = []
    for q, r in zip(queries, retrieval):
        rec = answerer.answer_query(
            query_id=q["query_id"],
            question=q["question"],
            retrieved=r["top_k"],
            use_llm=use_llm,
            input_artifacts=[chunks_path, retrieval_path],
            output_artifact=answers_path,
        )
        answers.append(rec)
    _write_json(answers_path, answers)
    state.advance_to("ANSWERS_GENERATED")

    # 6. Evaluate retrieval
    eval_records, eval_summary = evaluator.evaluate(retrieval, queries)
    eval_path = os.path.join(artifacts_dir, "eval.json")
    _write_json(eval_path, {"summary": eval_summary, "per_query": eval_records})
    state.advance_to("EVALUATION_COMPLETE")

    # 7. Grounding check
    grounding_records = grounding.check_grounding(answers, retrieval)
    grounding_path = os.path.join(artifacts_dir, "grounding_check.json")
    _write_json(grounding_path, grounding_records)
    state.advance_to("VALIDATION_COMPLETE")

    state.advance_to("RESULTS_FINALISED")
    return {
        "strategy": strategy,
        "summary": eval_summary,
        "artifacts": {
            "chunks": chunks_path,
            "retrieval": retrieval_path,
            "answers": answers_path,
            "eval": eval_path,
            "grounding_check": grounding_path,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="paragraph", choices=list(STRATEGIES.keys()))
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM, use deterministic extractive fallback only.")
    parser.add_argument("--compare-chunking", action="store_true",
                        help="Run both chunking strategies and write artifacts/chunking_comparison.json.")
    args = parser.parse_args(argv)

    if args.compare_chunking:
        results = {}
        for strat in STRATEGIES:
            sub_dir = os.path.join(ARTIFACTS_DIR, f"_{strat}")
            res = run(strategy=strat, use_llm=not args.no_llm, artifacts_dir=sub_dir)
            results[strat] = res["summary"]

        best = max(results.items(), key=lambda kv: kv[1]["top3_hit_rate"])
        comparison = {
            "strategies": results,
            "best_strategy": best[0],
            "tradeoff_notes": (
                "Paragraph chunking preserves semantic units (each fact is a sentence). "
                "Fixed-size chunking may split semantic units but offers more uniform "
                "embedding-space coverage. For small, well-structured KBs paragraph "
                "chunking is usually superior."
            ),
        }
        _write_json(os.path.join(ARTIFACTS_DIR, "chunking_comparison.json"), comparison)

        # Also leave the default artifacts in place using the winning strategy.
        run(strategy=best[0], use_llm=not args.no_llm)
        print(f"[main] chunking comparison: {results}")
        print(f"[main] best strategy: {best[0]}")
        return 0

    res = run(strategy=args.strategy, use_llm=not args.no_llm)
    print(f"[main] done — strategy={res['strategy']} summary={res['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
