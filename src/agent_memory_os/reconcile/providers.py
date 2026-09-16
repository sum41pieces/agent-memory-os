"""Deterministic, I/O-free sources of reconciliation candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agent_memory_os.reconcile.models import (
    MemoryCandidate,
    ReconciliationInputError,
    _validate_required_text,
)


class MemoryCandidateProvider(Protocol):
    """A source of normalized memory candidates for one project."""

    def candidates_for(self, project_id: str) -> Sequence[MemoryCandidate]:
        """Return the candidates belonging to ``project_id``."""

        ...


@dataclass(frozen=True)
class InMemoryCandidateProvider:
    """An immutable candidate provider with deterministic ordering."""

    project_id: str
    _candidates: tuple[MemoryCandidate, ...]

    def __init__(
        self,
        project_id: str,
        candidates: Sequence[MemoryCandidate],
    ) -> None:
        _validate_required_text(project_id, "project_id")
        copied = tuple(candidates)
        if not all(
            isinstance(candidate, MemoryCandidate) for candidate in copied
        ):
            raise ReconciliationInputError(
                "candidates must contain only MemoryCandidate instances"
            )
        candidate_ids = [candidate.candidate_id for candidate in copied]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ReconciliationInputError(
                "duplicate candidate IDs are not allowed"
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(
            self,
            "_candidates",
            tuple(sorted(copied, key=lambda item: item.candidate_id)),
        )

    def candidates_for(self, project_id: str) -> tuple[MemoryCandidate, ...]:
        return self._candidates if project_id == self.project_id else ()


__all__ = ["InMemoryCandidateProvider", "MemoryCandidateProvider"]
