"""Pipeline state machine.

Enforces the required stage order:

    INIT -> DOCUMENTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT ->
    RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> EVALUATION_COMPLETE ->
    VALIDATION_COMPLETE -> RESULTS_FINALISED

Calling a stage out of order raises immediately. Answers cannot be
generated before retrieval completes.
"""
from __future__ import annotations

STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


class PipelineState:
    def __init__(self) -> None:
        self._idx = 0  # currently at INIT

    @property
    def current(self) -> str:
        return STAGES[self._idx]

    def advance_to(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        target = STAGES.index(stage)
        if target != self._idx + 1:
            raise RuntimeError(
                f"Illegal stage transition: {self.current} -> {stage}. "
                f"Expected {STAGES[self._idx + 1] if self._idx + 1 < len(STAGES) else 'end'}."
            )
        self._idx = target
        print(f"[pipeline] -> {stage}")

    def require(self, stage: str) -> None:
        if STAGES.index(stage) > self._idx:
            raise RuntimeError(
                f"Cannot proceed: pipeline is at {self.current}, requires {stage} first."
            )
