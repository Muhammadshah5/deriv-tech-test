"""Deterministic document loading and chunking.

Two strategies are supported so we can compare retrieval performance:

  - paragraph: split the document body on blank lines (semantically natural)
  - fixed:     split the document body into fixed-size character windows

Both strategies parse `Title:` and `Section:` headers from the file and
preserve `start_char` / `end_char` offsets into the original body text.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return asdict(self)


_TITLE_RE = re.compile(r"^Title:\s*(.+?)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^Section:\s*(.+?)\s*$", re.MULTILINE)


def _parse_headers(raw: str) -> tuple[str, str, str]:
    """Return (title, section, body). Body is text after the headers."""
    title_m = _TITLE_RE.search(raw)
    section_m = _SECTION_RE.search(raw)
    title = title_m.group(1).strip() if title_m else "Untitled"
    section = section_m.group(1).strip() if section_m else "General"

    # Body starts after the last of (Title, Section) lines.
    header_end = 0
    for m in (title_m, section_m):
        if m and m.end() > header_end:
            header_end = m.end()
    body = raw[header_end:].lstrip("\n")
    return title, section, body


def load_documents(kb_dir: str) -> list[dict]:
    """Load every .txt file under kb_dir into {filename, title, section, body}."""
    docs = []
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(kb_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        title, section, body = _parse_headers(raw)
        docs.append({"filename": fname, "title": title, "section": section, "body": body})
    return docs


def chunk_paragraphs(docs: list[dict]) -> list[Chunk]:
    """Split each document body on blank lines; one chunk per non-empty paragraph."""
    chunks: list[Chunk] = []
    for doc in docs:
        body = doc["body"]
        cursor = 0
        idx = 0
        for para in re.split(r"\n\s*\n", body):
            if not para.strip():
                cursor += len(para) + 2
                continue
            start = body.find(para, cursor)
            if start < 0:
                start = cursor
            end = start + len(para)
            cursor = end
            idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{_slug(doc['title'])}__chunk_{idx}",
                    doc_title=doc["title"],
                    section=doc["section"],
                    text=para.strip(),
                    start_char=start,
                    end_char=end,
                )
            )
    return chunks

def chunk_fixed(docs: list[dict], size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Fixed-size character windows with overlap."""
    chunks: list[Chunk] = []
    for doc in docs:
        body = doc["body"]
        n = len(body)
        idx = 0
        start = 0
        step = max(1, size - overlap)
        while start < n:
            end = min(n, start + size)
            window = body[start:end].strip()
            if window:
                idx += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{_slug(doc['title'])}__fx_{idx}",
                        doc_title=doc["title"],
                        section=doc["section"],
                        text=window,
                        start_char=start,
                        end_char=end,
                    )
                )
            if end == n:
                break
            start += step
    return chunks


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return s or "doc"


STRATEGIES = {
    "paragraph": chunk_paragraphs,
    "fixed": chunk_fixed,
}
