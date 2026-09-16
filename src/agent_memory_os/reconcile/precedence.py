"""Immutable source ranks and reconciliation policy validation."""

from __future__ import annotations

import math

from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    MemoryCandidate,
    ReconciliationInputError,
    ReconciliationPolicy,
    SourceType,
)


DEFAULT_MAX_FUTURE_CLOCK_SKEW_SECONDS = 300

DEFAULT_SOURCE_PRECEDENCE: tuple[tuple[SourceType, int], ...] = (
    (SourceType.USER_EXPLICIT, 7),
    (SourceType.CURRENT_EVIDENCE, 6),
    (SourceType.PROJECT_DOC, 5),
    (SourceType.PROJECT_CARD, 4),
    (SourceType.TEMPORAL_RECORD, 3),
    (SourceType.SESSION_LOG, 2),
    (SourceType.HISTORICAL_MEMORY, 1),
)


def is_valid_source_of_truth(
    candidate: MemoryCandidate,
    policy: ReconciliationPolicy,
) -> bool:
    """Return whether a candidate has the exact explicit designation."""

    return (
        policy.allow_explicit_source_of_truth_override
        and candidate.source_type is SourceType.USER_EXPLICIT
        and candidate.explicit_user_instruction.status
        is EvidenceStatus.KNOWN
        and candidate.explicit_user_instruction.value is True
        and candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.SOURCE_OF_TRUTH
    )


def is_direct_current_evidence(candidate: MemoryCandidate) -> bool:
    """Return whether a candidate explicitly declares direct current evidence."""

    return (
        candidate.source_type is SourceType.CURRENT_EVIDENCE
        and candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.CURRENT_FACT
    )


def is_strict_historical_memory(candidate: MemoryCandidate) -> bool:
    """Return whether a candidate explicitly declares historical memory."""

    return (
        candidate.source_type is SourceType.HISTORICAL_MEMORY
        and candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.HISTORICAL
    )


def _validate_complete_unique_ranks(
    source_precedence: tuple[tuple[SourceType, int], ...],
) -> None:
    source_types: list[SourceType] = []
    ranks: list[int] = []

    for source_type, rank in source_precedence:
        if not isinstance(source_type, SourceType):
            raise ReconciliationInputError(
                "source_precedence entries must use SourceType enum members"
            )
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ReconciliationInputError(
                "source_precedence ranks must be integers, not booleans"
            )
        source_types.append(source_type)
        ranks.append(rank)

    if len(set(source_types)) != len(source_types):
        raise ReconciliationInputError(
            "source_precedence must not contain duplicate source types"
        )
    if len(set(ranks)) != len(ranks):
        raise ReconciliationInputError(
            "source_precedence must not contain duplicate ranks"
        )

    missing = set(SourceType).difference(source_types)
    if missing:
        names = ", ".join(sorted(source_type.name for source_type in missing))
        raise ReconciliationInputError(
            f"source_precedence is missing SourceType members: {names}"
        )


def validate_policy(policy: ReconciliationPolicy) -> ReconciliationPolicy:
    """Validate a policy without selecting or sorting reconciliation winners."""

    if not isinstance(policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")

    threshold = policy.active_confidence_threshold
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
        or not math.isfinite(threshold)
    ):
        raise ReconciliationInputError(
            "active_confidence_threshold must be finite and within [0.0, 1.0]"
        )

    skew = policy.max_future_clock_skew_seconds
    if isinstance(skew, bool) or not isinstance(skew, int) or skew < 0:
        raise ReconciliationInputError(
            "max_future_clock_skew_seconds must be a non-negative integer"
        )

    for field_name in (
        "allow_explicit_source_of_truth_override",
        "allow_current_evidence_over_historical",
        "require_explicit_hint_for_user_override",
        "conflict_on_equal_precedence_disagreement",
    ):
        if not isinstance(getattr(policy, field_name), bool):
            raise ReconciliationInputError(f"{field_name} must be a boolean")

    _validate_complete_unique_ranks(policy.source_precedence)
    return policy
