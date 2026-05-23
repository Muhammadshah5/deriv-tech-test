"""Minimal FastAPI service.

POST /answer
    body: {"question": "string"}
    returns: {"answer_label", "answer", "citations"}

Index is built once at startup from the same kb/ used by the offline
pipeline, so the API and the batch run share retrieval logic.
"""
from __future__ import annotations

import json
from fastapi import FastAPI
from pydantic import BaseModel

from src import answerer
from src.chunker import chunk_paragraphs, load_documents
from src.retriever import HybridRetriever

app = FastAPI(title="deriv-tech-test mini-RAG")

_chunks = [c.to_dict() for c in chunk_paragraphs(load_documents("kb"))]
_retriever = HybridRetriever(_chunks)


class AnswerRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer_label: str
    answer: str
    citations: list[str]


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    retrieved = _retriever.search(req.question, top_k=3)
    record = answerer.answer_query(
        query_id="api",
        question=req.question,
        retrieved=retrieved,
        use_llm=True,
        input_artifacts=["kb/"],
        output_artifact="(api)",
    )
    return AnswerResponse(
        answer_label=record["answer_label"],
        answer=record["answer"],
        citations=record["citations"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(_chunks)}
