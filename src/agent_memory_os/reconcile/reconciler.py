"""Atomic construction of the immutable Phase 2 relation graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import secrets
from types import MappingProxyType

from agent_memory_os.evidence.models import (
    EvidenceSnapshot,
    EvidenceStatus,
    EvidenceValue,
    utc_now_iso,
)
from agent_memory_os.reconcile.models import (
    BaseFact,
    FactDecision,
    MemoryCandidate,
    ReconciledFact,
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationStatus,
    ReconciliationWarning,
    ReconciliationPolicy,
    ReconciliationResult,
    RelationGraph,
    RelationRecord,
    RelationType,
    ResolutionMethod,
    SourceType,
    UnresolvedCandidate,
    WarningCode,
    _BASE_FACT_CONSTRUCTION_TOKEN,
    _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN,
    _RECONCILED_FACT_CONSTRUCTION_TOKEN,
    _RELATION_GRAPH_CONSTRUCTION_TOKEN,
    _ReconciliationCohort,
    _ReconciliationStage,
    _claim_canonical_stage_builders,
    _strict_evidence_signature,
    _validate_reconciliation_cohort,
    _validate_stage_authorization,
)


(
    _base_fact_stage_builder,
    _materialized_fact_stage_builder,
    _materialized_relation_stage_builder,
    _materialized_graph_stage_builder,
) = _claim_canonical_stage_builders(__name__)
del _claim_canonical_stage_builders

from agent_memory_os.reconcile.rules import (
    RelationRequest,
    _candidate_id_namespace,
    _derive_unresolved_candidates,
    _make_predicate_outcome,
    _policy_fingerprint,
    _resolve_predicate_payload,
    _validate_complete_decision_cohort,
    _validate_decision_candidate_context,
    canonical_json_bytes,
    classify_group,
    group_candidates,
    make_relation_id,
    parse_known_timestamp,
    parse_reconciliation_clock,
    resolve_predicate,
)
from agent_memory_os.reconcile.precedence import validate_policy


def aggregate_human_review(
    facts: Sequence[ReconciledFact],
    unresolved: Sequence[UnresolvedCandidate],
    warnings: Sequence[ReconciliationWarning],
) -> bool:
    """Apply the exact result-level human-review disjunction."""

    return (
        any(fact.requires_human_review for fact in facts)
        or bool(unresolved)
        or any(warning.requires_human_review for warning in warnings)
    )


def _warning_key(warning: ReconciliationWarning) -> tuple[object, ...]:
    return (
        warning.code.value,
        warning.candidate_ids,
        warning.message,
        warning.evidence_refs,
        warning.requires_human_review,
    )


def _complete_warning_union(
    decisions: tuple[FactDecision, ...],
    warnings: Sequence[ReconciliationWarning],
) -> tuple[ReconciliationWarning, ...]:
    supplied = _guarded_materialization_sequence(warnings, "warnings")
    if any(type(warning) is not ReconciliationWarning for warning in supplied):
        raise ReconciliationInputError(
            "warnings must contain only exact ReconciliationWarning records"
        )
    indexed = {
        _warning_key(warning): warning
        for decision in decisions
        for warning in decision.warnings
    }
    indexed.update({_warning_key(warning): warning for warning in supplied})
    return tuple(indexed[key] for key in sorted(indexed))


def assemble_result(
    snapshot: EvidenceSnapshot,
    decisions: Sequence[FactDecision],
    relation_requests: Sequence[RelationRequest],
    unresolved: Sequence[UnresolvedCandidate],
    warnings: Sequence[ReconciliationWarning],
    reconciled_at: datetime,
):
    """Materialize and publish one validated result from canonical stages."""

    if type(snapshot) is not EvidenceSnapshot:
        raise ReconciliationInputError(
            "snapshot must be an exact EvidenceSnapshot"
        )
    if (
        type(snapshot.project_id) is not EvidenceValue
        or snapshot.project_id.status is not EvidenceStatus.KNOWN
        or type(snapshot.project_id.value) is not str
        or not snapshot.project_id.value.strip()
        or "\x00" in snapshot.project_id.value
    ):
        raise ReconciliationInputError(
            "snapshot project_id must be a KNOWN non-empty string"
        )
    if (
        type(reconciled_at) is not datetime
        or reconciled_at.tzinfo is None
        or reconciled_at.utcoffset() is None
    ):
        raise ReconciliationInputError(
            "reconciled_at must be a timezone-aware datetime"
        )
    try:
        normalized_at = reconciled_at.astimezone(timezone.utc).isoformat()
    except (OverflowError, ValueError) as error:
        raise ReconciliationInputError(
            "reconciled_at cannot be normalized to UTC"
        ) from error

    decision_snapshot = _guarded_materialization_sequence(
        decisions,
        "decisions",
    )
    if not decision_snapshot or any(
        type(decision) is not FactDecision for decision in decision_snapshot
    ):
        raise ReconciliationInputError(
            "decisions must contain exact FactDecision records"
        )
    unresolved_snapshot = _guarded_materialization_sequence(
        unresolved,
        "unresolved",
    )
    if any(
        type(record) is not UnresolvedCandidate
        for record in unresolved_snapshot
    ):
        raise ReconciliationInputError(
            "unresolved must contain only exact UnresolvedCandidate records"
        )

    checked_warnings = _complete_warning_union(
        decision_snapshot,
        warnings,
    )
    graph = materialize_relation_graph(decision_snapshot, relation_requests)

    from agent_memory_os.reconcile.serialization import make_snapshot_id

    result = ReconciliationResult.create(
        project_id=snapshot.project_id.value,
        snapshot_id=make_snapshot_id(snapshot),
        reconciled_at=normalized_at,
        graph=graph,
        source_decisions=decision_snapshot,
        unresolved=unresolved_snapshot,
        warnings=checked_warnings,
    )
    if result.human_review_required != aggregate_human_review(
        result.all_facts(),
        result.unresolved,
        result.warnings,
    ):
        raise ReconciliationInvariantError(
            "result human-review aggregation is inconsistent"
        )
    return result


def _validate_snapshot_identity_and_time(snapshot: object) -> EvidenceSnapshot:
    """Validate the Phase 1 identity fields consumed by reconciliation."""

    if type(snapshot) is not EvidenceSnapshot:
        raise ReconciliationInputError(
            "snapshot must be an exact EvidenceSnapshot"
        )
    for field_name in ("schema_version", "project_id", "captured_at"):
        field_value = getattr(snapshot, field_name)
        if type(field_value) is not EvidenceValue:
            raise ReconciliationInputError(
                f"snapshot {field_name} must be an exact EvidenceValue"
            )
        if field_value.status is not EvidenceStatus.KNOWN:
            raise ReconciliationInputError(
                f"snapshot {field_name} must be KNOWN"
            )
    if (
        type(snapshot.schema_version.value) is not str
        or snapshot.schema_version.value != "1.0.0"
    ):
        raise ReconciliationInputError(
            "snapshot schema_version must be the supported Phase 1 version 1.0.0"
        )
    project_id = snapshot.project_id.value
    if (
        type(project_id) is not str
        or not project_id.strip()
        or "\x00" in project_id
    ):
        raise ReconciliationInputError(
            "snapshot project_id must be a KNOWN non-empty string"
        )
    parse_known_timestamp(snapshot.captured_at, "snapshot captured_at")
    return snapshot


def _copy_and_sort_candidates(
    candidates: object,
) -> tuple[MemoryCandidate, ...]:
    """Take one stable caller-independent snapshot and reject duplicate IDs."""

    if not isinstance(candidates, Sequence) or isinstance(
        candidates,
        (str, bytes, bytearray),
    ):
        raise ReconciliationInputError(
            "candidates must be a sequence of MemoryCandidate records"
        )
    try:
        copied = tuple(candidates)
    except Exception as error:
        raise ReconciliationInputError(
            "candidates must be a stable finite sequence"
        ) from error
    if any(type(candidate) is not MemoryCandidate for candidate in copied):
        raise ReconciliationInputError(
            "candidates must contain only exact MemoryCandidate records"
        )
    counts: dict[str, int] = {}
    for candidate in copied:
        counts[candidate.candidate_id] = counts.get(candidate.candidate_id, 0) + 1
    duplicates = sorted(
        candidate_id
        for candidate_id, count in counts.items()
        if count > 1
    )
    if duplicates:
        raise ReconciliationInputError(
            f"duplicate candidate_id: {duplicates[0]}"
        )
    return tuple(sorted(copied, key=lambda candidate: candidate.candidate_id))


def _empty_result(
    snapshot: EvidenceSnapshot,
    snapshot_id: str,
    policy: ReconciliationPolicy,
    reconciled_at: datetime,
) -> ReconciliationResult:
    """Create the only valid zero-candidate graph and result shape."""

    cohort = _ReconciliationCohort._from_payload(
        _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN,
        project_id=snapshot.project_id.value,
        snapshot_id=snapshot_id,
        policy_fingerprint=_policy_fingerprint(policy),
        reconciled_at=reconciled_at.isoformat(),
        candidate_id_namespace=_candidate_id_namespace(
            snapshot.project_id.value,
            (),
        ),
        predicate_membership=(),
        invocation_token=secrets.token_hex(32),
    )
    graph = _build_empty_relation_graph(cohort)
    return ReconciliationResult.create(
        project_id=snapshot.project_id.value,
        snapshot_id=snapshot_id,
        reconciled_at=reconciled_at.isoformat(),
        graph=graph,
        source_decisions=(),
        unresolved=(),
        warnings=(),
    )


def reconcile(
    snapshot: EvidenceSnapshot,
    candidates: Sequence[MemoryCandidate],
    policy: ReconciliationPolicy,
    *,
    clock: Callable[[], str] = utc_now_iso,
) -> ReconciliationResult:
    """Run the complete deterministic Phase 2 reconciliation pipeline."""

    checked_policy = validate_policy(policy)
    if not callable(clock):
        raise ReconciliationInputError("clock must be callable")
    checked_snapshot = _validate_snapshot_identity_and_time(snapshot)

    from agent_memory_os.reconcile.serialization import make_snapshot_id

    snapshot_id = make_snapshot_id(checked_snapshot)
    checked_candidates = _copy_and_sort_candidates(candidates)
    reconciled_at = parse_reconciliation_clock(clock())
    if not checked_candidates:
        return _empty_result(
            checked_snapshot,
            snapshot_id,
            checked_policy,
            reconciled_at,
        )

    groups = group_candidates(
        checked_snapshot.project_id.value,
        checked_candidates,
        snapshot_id=snapshot_id,
    )
    classified = tuple(
        classify_group(group, reconciled_at, checked_policy)
        for group in groups
    )
    by_predicate: dict[tuple[str, str], list[FactDecision]] = {}
    for decision in classified:
        by_predicate.setdefault(
            (decision.subject, decision.predicate),
            [],
        ).append(decision)

    final_decisions: list[FactDecision] = []
    requests: list[RelationRequest] = []
    warning_index: dict[tuple[object, ...], ReconciliationWarning] = {}
    for key in sorted(by_predicate):
        outcome = resolve_predicate(
            tuple(
                sorted(
                    by_predicate[key],
                    key=lambda decision: decision.fact_id,
                )
            ),
            checked_policy,
        )
        final_decisions.extend(outcome.decisions)
        requests.extend(outcome.relation_requests)
        for decision in outcome.decisions:
            for warning in decision.warnings:
                warning_index[_warning_key(warning)] = warning

    decision_snapshot = tuple(
        sorted(final_decisions, key=lambda decision: decision.fact_id)
    )
    unresolved = _derive_unresolved_candidates(decision_snapshot)
    warnings = tuple(warning_index[key] for key in sorted(warning_index))
    return assemble_result(
        checked_snapshot,
        decision_snapshot,
        tuple(requests),
        unresolved,
        warnings,
        reconciled_at,
    )


_RESOLUTION_METHOD_PRIORITY = {
    method: index
    for index, method in enumerate(
        (
            ResolutionMethod.SAME_VALUE_REACTIVATION,
            ResolutionMethod.EXPLICIT_DEPRECATION,
            ResolutionMethod.TEMPORAL_PENDING,
            ResolutionMethod.INSUFFICIENT_EVIDENCE,
            ResolutionMethod.PENDING_SEMANTICS,
            ResolutionMethod.UNRESOLVED_CONFLICT,
            ResolutionMethod.EXPLICIT_SUPERSEDES,
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
            ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
            ResolutionMethod.SAME_VALUE_MERGE,
            ResolutionMethod.DIRECT_CURRENT,
        )
    )
}


@dataclass
class _RequestAccumulator:
    primary: RelationRequest
    candidate_ids: set[str]
    evidence_refs: set[str]

    def add(self, request: RelationRequest) -> None:
        if _request_priority(request) < _request_priority(self.primary):
            self.primary = request
        self.candidate_ids.update(request.candidate_ids)
        self.evidence_refs.update(request.evidence_refs)


@dataclass
class _FactIndexes:
    relation_ids: set[str]
    supersedes: set[str]
    superseded_by: set[str]
    conflicts: set[str]
    conflict_candidates: set[str]


def _request_priority(request: RelationRequest) -> tuple[int, str]:
    return (_RESOLUTION_METHOD_PRIORITY[request.method], request.reason)


def _snapshot_request(request: object) -> RelationRequest:
    if type(request) is not RelationRequest:
        raise ReconciliationInputError(
            "requests must contain only exact RelationRequest records"
        )
    try:
        relation_type = request.relation_type
        if type(relation_type) is not RelationType:
            raise ReconciliationInputError(
                "relation_type must be a RelationType"
            )
        return RelationRequest(
            relation_type=relation_type,
            from_fact_id=request.from_fact_id,
            to_fact_id=request.to_fact_id,
            reason=request.reason,
            candidate_ids=tuple(request.candidate_ids),
            evidence_refs=tuple(request.evidence_refs),
            method=request.method,
        )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        if isinstance(error, ReconciliationInputError):
            raise
        raise ReconciliationInputError(
            "requests must be complete canonical RelationRequest records"
        ) from error


def _copy_clock(reconciled_at: datetime) -> str:
    try:
        if (
            type(reconciled_at) is not datetime
            or reconciled_at.utcoffset() is None
            or reconciled_at.utcoffset().total_seconds() != 0
        ):
            raise ValueError
        copied = datetime(
            reconciled_at.year,
            reconciled_at.month,
            reconciled_at.day,
            reconciled_at.hour,
            reconciled_at.minute,
            reconciled_at.second,
            reconciled_at.microsecond,
            tzinfo=timezone.utc,
            fold=reconciled_at.fold,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ReconciliationInputError(
            "FactDecision reconciliation clock is invalid"
        ) from error
    return copied.isoformat()


def _canonical_base_fact_payload(fact: object) -> bytes:
    return canonical_json_bytes(
        {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "selected_value": _strict_evidence_signature(
                fact.selected_value,
                "selected_value",
                "json",
            ).hex(),
            "status": fact.status.value,
            "confidence": _strict_evidence_signature(
                fact.confidence,
                "confidence",
                "float",
            ).hex(),
            "reason": fact.reason,
            "evidence_refs": list(fact.evidence_refs),
            "candidate_ids": list(fact.candidate_ids),
            "superseded_candidate_ids": list(
                fact.superseded_candidate_ids
            ),
            "valid_from": _strict_evidence_signature(
                fact.valid_from,
                "valid_from",
                "string",
            ).hex(),
            "resolved_at": _strict_evidence_signature(
                fact.resolved_at,
                "resolved_at",
                "string",
            ).hex(),
            "resolution_method": fact.resolution_method.value,
            "requires_human_review": fact.requires_human_review,
            "activation_witness_candidate_ids": list(
                fact.activation_witness_candidate_ids
            ),
        }
    )


def _base_fact_stage_payload(fact: BaseFact) -> bytes:
    if type(fact) is not BaseFact:
        raise ReconciliationInputError(
            "base fact stage requires an exact BaseFact"
        )
    return _canonical_base_fact_payload(fact)


def _final_fact_base_payload(fact: ReconciledFact) -> bytes:
    if type(fact) is not ReconciledFact:
        raise ReconciliationInputError(
            "final fact base payload requires an exact ReconciledFact"
        )
    return _canonical_base_fact_payload(fact)


def _fact_stage_payload(fact: ReconciledFact) -> bytes:
    return canonical_json_bytes(
        {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "selected_value": _strict_evidence_signature(
                fact.selected_value,
                "selected_value",
                "json",
            ).hex(),
            "status": fact.status.value,
            "confidence": _strict_evidence_signature(
                fact.confidence,
                "confidence",
                "float",
            ).hex(),
            "reason": fact.reason,
            "evidence_refs": list(fact.evidence_refs),
            "candidate_ids": list(fact.candidate_ids),
            "superseded_candidate_ids": list(
                fact.superseded_candidate_ids
            ),
            "conflict_candidate_ids": list(fact.conflict_candidate_ids),
            "valid_from": _strict_evidence_signature(
                fact.valid_from,
                "valid_from",
                "string",
            ).hex(),
            "resolved_at": _strict_evidence_signature(
                fact.resolved_at,
                "resolved_at",
                "string",
            ).hex(),
            "resolution_method": fact.resolution_method.value,
            "requires_human_review": fact.requires_human_review,
            "activation_witness_candidate_ids": list(
                fact.activation_witness_candidate_ids
            ),
            "relation_ids": list(fact.relation_ids),
            "supersedes_fact_ids": list(fact.supersedes_fact_ids),
            "superseded_by_fact_ids": list(fact.superseded_by_fact_ids),
            "conflict_fact_ids": list(fact.conflict_fact_ids),
        }
    )


def _relation_stage_payload(relation: RelationRecord) -> bytes:
    return canonical_json_bytes(
        {
            "relation_id": relation.relation_id,
            "relation_type": relation.relation_type.value,
            "from_fact_id": relation.from_fact_id,
            "to_fact_id": relation.to_fact_id,
            "relation_reason": relation.relation_reason,
            "candidate_ids": list(relation.candidate_ids),
            "evidence_refs": list(relation.evidence_refs),
        }
    )


def _graph_stage_payload(graph: RelationGraph) -> bytes:
    lookup_entries = _graph_lookup_stage_entries(graph)
    return canonical_json_bytes(
        {
            "facts": [fact._stage_auth.fingerprint for fact in graph.facts],
            "relations": [
                relation._stage_auth.fingerprint
                for relation in graph.relations
            ],
            "fact_lookup": [
                [fact_id, stage_fingerprint]
                for fact_id, _, stage_fingerprint in lookup_entries
            ],
        }
    )


def _graph_lookup_stage_entries(
    graph: RelationGraph,
) -> tuple[tuple[str, ReconciledFact, str], ...]:
    try:
        lookup = object.__getattribute__(graph, "_facts_by_id")
        if type(lookup) is not MappingProxyType:
            raise ReconciliationInputError(
                "graph fact lookup must be an exact frozen mapping"
            )
        items = tuple(lookup.items())
        entries: list[tuple[str, ReconciledFact, str]] = []
        for item in items:
            if type(item) is not tuple or len(item) != 2:
                raise ReconciliationInputError(
                    "graph fact lookup items must be exact pairs"
                )
            fact_id, fact = item
            if (
                type(fact_id) is not str
                or not fact_id.strip()
                or "\x00" in fact_id
                or type(fact) is not ReconciledFact
            ):
                raise ReconciliationInputError(
                    "graph fact lookup must contain exact fact members"
                )
            authorization = object.__getattribute__(fact, "_stage_auth")
            stage_fingerprint = object.__getattribute__(
                authorization,
                "fingerprint",
            )
            if (
                type(stage_fingerprint) is not str
                or not stage_fingerprint.strip()
                or "\x00" in stage_fingerprint
            ):
                raise ReconciliationInputError(
                    "graph fact lookup stage fingerprint is invalid"
                )
            entries.append((fact_id, fact, stage_fingerprint))
        return tuple(sorted(entries, key=lambda entry: entry[0]))
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ReconciliationInputError(
            "graph fact lookup could not be inspected safely"
        ) from error


def _base_fact_from_authenticated_decision(
    decision: FactDecision,
) -> BaseFact:
    """Derive one index-free base fact from an authenticated decision."""

    canonical_base = _validate_decision_candidate_context(decision)
    reconciled_at = _copy_clock(
        canonical_base._candidate_context.reconciled_at
    )
    witnesses = (
        decision.activation_witness_candidate_ids
        if decision.status is ReconciliationStatus.ACTIVE
        else ()
    )
    return BaseFact._from_fields(
        _BASE_FACT_CONSTRUCTION_TOKEN,
        fact_id=decision.fact_id,
        subject=decision.subject,
        predicate=decision.predicate,
        selected_value=decision.selected_value,
        status=decision.status,
        confidence=decision.confidence,
        reason=decision.reason,
        evidence_refs=decision.evidence_refs,
        candidate_ids=decision.candidate_ids,
        superseded_candidate_ids=decision.superseded_candidate_ids,
        valid_from=decision.valid_from,
        resolved_at=EvidenceValue.known(
            reconciled_at,
            source="reconciliation:clock",
        ),
        resolution_method=decision.resolution_method,
        requires_human_review=decision.requires_human_review,
        activation_witness_candidate_ids=witnesses,
    )


def _merge_requests(
    requests: tuple[RelationRequest, ...],
) -> tuple[RelationRequest, ...]:
    accumulators: dict[tuple[str, str, str], _RequestAccumulator] = {}
    for request in requests:
        key = (
            request.relation_type.value,
            request.from_fact_id,
            request.to_fact_id,
        )
        if key not in accumulators:
            accumulators[key] = _RequestAccumulator(request, set(), set())
        accumulators[key].add(request)

    merged: list[RelationRequest] = []
    for key in sorted(accumulators):
        accumulator = accumulators[key]
        primary = accumulator.primary
        factory = (
            RelationRequest.supersedes
            if primary.relation_type is RelationType.SUPERSEDES
            else RelationRequest.conflicts
        )
        merged.append(
            factory(
                primary.from_fact_id,
                primary.to_fact_id,
                primary.reason,
                tuple(sorted(accumulator.candidate_ids)),
                tuple(sorted(accumulator.evidence_refs)),
                method=primary.method,
            )
        )
    return tuple(merged)


def _validate_conflict_symmetry(requests: tuple[RelationRequest, ...]) -> None:
    by_key = {
        (request.from_fact_id, request.to_fact_id): request
        for request in requests
        if request.relation_type is RelationType.CONFLICTS
    }
    for (source, target), request in sorted(by_key.items()):
        inverse = by_key.get((target, source))
        if inverse is None:
            raise ReconciliationInputError(
                "CONFLICTS requests must be supplied as a symmetric pair"
            )
        if (
            inverse.reason != request.reason
            or inverse.candidate_ids != request.candidate_ids
            or inverse.evidence_refs != request.evidence_refs
        ):
            raise ReconciliationInputError(
                "symmetric CONFLICTS requests must have the same reason and provenance"
            )


def _derive_indexes(
    facts_by_id: Mapping[str, BaseFact],
    relations: tuple[RelationRecord, ...],
) -> dict[str, _FactIndexes]:
    indexes = {
        fact_id: _FactIndexes(set(), set(), set(), set(), set())
        for fact_id in facts_by_id
    }
    for relation in relations:
        source = indexes[relation.from_fact_id]
        target = indexes[relation.to_fact_id]
        source.relation_ids.add(relation.relation_id)
        target.relation_ids.add(relation.relation_id)
        if relation.relation_type is RelationType.SUPERSEDES:
            source.supersedes.add(relation.to_fact_id)
            target.superseded_by.add(relation.from_fact_id)
            continue
        source.conflicts.add(relation.to_fact_id)
        target.conflicts.add(relation.from_fact_id)
        source.conflict_candidates.update(
            set(relation.candidate_ids)
            - set(facts_by_id[relation.from_fact_id].candidate_ids)
        )
        target.conflict_candidates.update(
            set(relation.candidate_ids)
            - set(facts_by_id[relation.to_fact_id].candidate_ids)
        )
    return indexes


def _relation_parent_fingerprint(
    parent_auth_by_fact_id: Mapping[str, object],
    from_fact_id: str,
    to_fact_id: str,
) -> str:
    try:
        parent_fingerprints = tuple(
            sorted(
                object.__getattribute__(
                    parent_auth_by_fact_id[fact_id],
                    "fingerprint",
                )
                for fact_id in (from_fact_id, to_fact_id)
            )
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise ReconciliationInputError(
            "relation parent source decisions are incomplete"
        ) from error
    return (
        "relation-parents:v1:"
        + hashlib.sha256(canonical_json_bytes(parent_fingerprints)).hexdigest()
    )


def _authenticate_coordinated_payload(
    decisions: tuple[FactDecision, ...],
    requests: tuple[RelationRequest, ...],
) -> tuple[tuple[FactDecision, ...], tuple[RelationRequest, ...]]:
    """Verify and return the exact coordinator outcomes for all predicates."""

    _validate_complete_decision_cohort(decisions)
    grouped: dict[tuple[str, str], list[tuple[str, FactDecision]]] = {}
    fact_predicates: dict[str, tuple[str, str]] = {}
    for decision in decisions:
        canonical = _validate_decision_candidate_context(decision)
        key = (canonical.subject, canonical.predicate)
        if canonical.fact_id in fact_predicates:
            raise ReconciliationInputError(
                "coordinated decisions must have unique canonical fact IDs"
            )
        fact_predicates[canonical.fact_id] = key
        grouped.setdefault(key, []).append((canonical.fact_id, decision))

    requests_by_predicate: dict[
        tuple[str, str], list[RelationRequest]
    ] = {key: [] for key in grouped}
    for request in requests:
        source_key = fact_predicates.get(request.from_fact_id)
        target_key = fact_predicates.get(request.to_fact_id)
        if source_key is None or source_key != target_key:
            raise ReconciliationInputError(
                "coordinated relation requests must stay within one predicate"
            )
        requests_by_predicate[source_key].append(request)

    authenticated_decisions: list[FactDecision] = []
    authenticated_requests: list[RelationRequest] = []
    for key in sorted(grouped):
        supplied_decisions = tuple(
            decision for _, decision in sorted(grouped[key])
        )
        supplied_requests = tuple(
            sorted(
                requests_by_predicate[key],
                key=lambda request: (
                    request.relation_type.value,
                    request.from_fact_id,
                    request.to_fact_id,
                ),
            )
        )
        try:
            outcome = _make_predicate_outcome(
                supplied_decisions,
                supplied_requests,
            )
        except ReconciliationInputError as error:
            raise ReconciliationInputError(
                "decisions and requests must match canonical coordinator payload"
            ) from error
        authenticated_decisions.extend(supplied_decisions)
        authenticated_requests.extend(outcome.relation_requests)

    return (
        tuple(sorted(authenticated_decisions, key=lambda item: item.fact_id)),
        tuple(
            sorted(
                authenticated_requests,
                key=lambda item: (
                    item.relation_type.value,
                    item.from_fact_id,
                    item.to_fact_id,
                ),
            )
        ),
    )


def _make_canonical_relation_graph_builder(
    issue_base_fact_stage: object,
    issue_fact_stage: object,
    issue_relation_stage: object,
    issue_graph_stage: object,
) -> tuple[object, object]:
    def build_canonical_relation_graph(
        decisions: tuple[FactDecision, ...],
        request_snapshot: tuple[RelationRequest, ...],
    ) -> RelationGraph:
        canonical_decisions, canonical_requests = (
            _authenticate_coordinated_payload(decisions, request_snapshot)
        )
        checked_cohort = _validate_complete_decision_cohort(
            canonical_decisions
        )
        graph_build_token = secrets.token_hex(32)
        base_facts = tuple(
            _base_fact_from_authenticated_decision(decision)
            for decision in canonical_decisions
        )
        decisions_by_fact_id = {
            decision.fact_id: decision for decision in canonical_decisions
        }
        for base_fact in base_facts:
            decision = decisions_by_fact_id[base_fact.fact_id]
            decision_auth = object.__getattribute__(decision, "_stage_auth")
            decision_stage = object.__getattribute__(decision_auth, "stage")
            object.__setattr__(base_fact, "_cohort", checked_cohort)
            object.__setattr__(base_fact, "_source_decision", decision)
            issue_base_fact_stage(
                base_fact,
                cohort=checked_cohort,
                previous_stage=decision_stage,
                parent_fingerprint=object.__getattribute__(
                    decision_auth,
                    "fingerprint",
                ),
                payload=_base_fact_stage_payload(base_fact),
                graph_build_token=graph_build_token,
            )
        parent_auth_by_fact_id = {
            base_fact.fact_id: base_fact._stage_auth
            for base_fact in base_facts
        }
        return assemble(
            base_facts,
            canonical_requests,
            checked_cohort,
            parent_auth_by_fact_id,
            graph_build_token,
        )

    def assemble(
        base_facts: tuple[BaseFact, ...],
        request_snapshot: tuple[RelationRequest, ...],
        checked_cohort: _ReconciliationCohort,
        parent_auth_by_fact_id: Mapping[str, object],
        graph_build_token: str,
    ) -> RelationGraph:
        facts_by_id: dict[str, BaseFact] = {}
        for fact in base_facts:
            if fact.fact_id in facts_by_id:
                raise ReconciliationInputError(
                    "facts must have unique fact_id values"
                )
            facts_by_id[fact.fact_id] = fact

        for request in request_snapshot:
            if request.from_fact_id == request.to_fact_id:
                raise ReconciliationInputError(
                    "relation request cannot be a fact self-edge"
                )
            if (
                request.from_fact_id not in facts_by_id
                or request.to_fact_id not in facts_by_id
            ):
                raise ReconciliationInputError(
                    "every relation request endpoint must reference an input fact"
                )

        merged_requests = _merge_requests(request_snapshot)
        _validate_conflict_symmetry(merged_requests)
        relations = tuple(
            RelationRecord.create(
                request.relation_type,
                request.from_fact_id,
                request.to_fact_id,
                request.reason,
                request.candidate_ids,
                request.evidence_refs,
            )
            for request in merged_requests
        )
        for relation in relations:
            object.__setattr__(relation, "_cohort", checked_cohort)
            issue_relation_stage(
                relation,
                cohort=checked_cohort,
                previous_stage=_ReconciliationStage.BASE_FACT,
                parent_fingerprint=_relation_parent_fingerprint(
                    parent_auth_by_fact_id,
                    relation.from_fact_id,
                    relation.to_fact_id,
                ),
                payload=_relation_stage_payload(relation),
                graph_build_token=graph_build_token,
            )
        indexes = _derive_indexes(facts_by_id, relations)
        materialized = tuple(
            ReconciledFact._from_materialized(
                _RECONCILED_FACT_CONSTRUCTION_TOKEN,
                facts_by_id[fact_id],
                conflict_candidate_ids=tuple(
                    sorted(indexes[fact_id].conflict_candidates)
                ),
                relation_ids=tuple(sorted(indexes[fact_id].relation_ids)),
                supersedes_fact_ids=tuple(
                    sorted(indexes[fact_id].supersedes)
                ),
                superseded_by_fact_ids=tuple(
                    sorted(indexes[fact_id].superseded_by)
                ),
                conflict_fact_ids=tuple(
                    sorted(indexes[fact_id].conflicts)
                ),
            )
            for fact_id in sorted(facts_by_id)
        )
        for fact in materialized:
            object.__setattr__(fact, "_cohort", checked_cohort)
            base_fact = facts_by_id[fact.fact_id]
            object.__setattr__(fact, "_base_fact", base_fact)
            parent_auth = parent_auth_by_fact_id[fact.fact_id]
            issue_fact_stage(
                fact,
                cohort=checked_cohort,
                previous_stage=_ReconciliationStage.BASE_FACT,
                parent_fingerprint=object.__getattribute__(
                    parent_auth,
                    "fingerprint",
                ),
                payload=_fact_stage_payload(fact),
                graph_build_token=graph_build_token,
            )
        graph = RelationGraph._from_materialized(
            _RELATION_GRAPH_CONSTRUCTION_TOKEN,
            facts=materialized,
            relations=relations,
        )
        object.__setattr__(graph, "_cohort", checked_cohort)
        graph_previous_stage = (
            _ReconciliationStage.MATERIALIZED_RELATION
            if relations
            else _ReconciliationStage.MATERIALIZED_FACT
        )
        issue_graph_stage(
            graph,
            cohort=checked_cohort,
            previous_stage=graph_previous_stage,
            parent_fingerprint=(
                "graph-members:v1:"
                + hashlib.sha256(_graph_stage_payload(graph)).hexdigest()
            ),
            payload=_graph_stage_payload(graph),
            graph_build_token=graph_build_token,
        )
        return graph

    def build_empty_relation_graph(
        cohort: _ReconciliationCohort,
    ) -> RelationGraph:
        """Materialize the authenticated empty graph for one valid run."""

        checked_cohort = _validate_reconciliation_cohort(
            cohort,
            require_bound=True,
        )
        if checked_cohort.predicate_membership:
            raise ReconciliationInputError(
                "empty graph requires an empty reconciliation cohort"
            )
        return assemble(
            (),
            (),
            checked_cohort,
            {},
            secrets.token_hex(32),
        )

    return build_canonical_relation_graph, build_empty_relation_graph


(
    _build_canonical_relation_graph,
    _build_empty_relation_graph,
) = _make_canonical_relation_graph_builder(
        _base_fact_stage_builder,
        _materialized_fact_stage_builder,
        _materialized_relation_stage_builder,
        _materialized_graph_stage_builder,
)
del _make_canonical_relation_graph_builder
del _base_fact_stage_builder
del _materialized_fact_stage_builder
del _materialized_relation_stage_builder
del _materialized_graph_stage_builder


def _guarded_materialization_sequence(
    value: object,
    field_name: str,
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ReconciliationInputError(
            f"{field_name} must be a sequence for materialization"
        )
    try:
        return tuple(value)
    except Exception as error:
        raise ReconciliationInputError(
            f"{field_name} must be a stable finite sequence for materialization"
        ) from error


def materialize_relation_graph(
    facts: Sequence[FactDecision],
    requests: Sequence[RelationRequest],
) -> RelationGraph:
    """Materialize only authenticated decisions from one current cohort."""

    fact_inputs = _guarded_materialization_sequence(facts, "facts")
    request_inputs = _guarded_materialization_sequence(requests, "requests")
    if not fact_inputs or any(
        type(fact) is not FactDecision for fact in fact_inputs
    ):
        raise ReconciliationInputError(
            "materialization requires exact authenticated FactDecision records"
        )
    request_snapshot = tuple(
        _snapshot_request(request) for request in request_inputs
    )
    decisions, request_snapshot = _authenticate_coordinated_payload(
        fact_inputs,
        request_snapshot,
    )
    return _build_canonical_relation_graph(decisions, request_snapshot)


def _fail_invariant(message: str) -> None:
    raise ReconciliationInvariantError(message)


def _snapshot_validation_sequence(value: object, field_name: str) -> tuple:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        _fail_invariant(f"{field_name} must be a finite sequence")
    try:
        return tuple(value)
    except Exception as error:
        raise ReconciliationInvariantError(
            f"{field_name} must be a finite sequence"
        ) from error


def _is_exact_required_text(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and "\x00" not in value


def _validate_final_fact_base_stage(
    fact: ReconciledFact,
    cohort: _ReconciliationCohort,
    graph_token: str,
) -> BaseFact:
    if type(fact) is not ReconciledFact:
        raise ReconciliationInputError(
            "final fact stage requires an exact ReconciledFact"
        )
    base_fact = object.__getattribute__(fact, "_base_fact")
    if type(base_fact) is not BaseFact:
        raise ReconciliationInputError(
            "final fact must retain an exact BaseFact parent"
        )
    base_cohort = _validate_reconciliation_cohort(
        object.__getattribute__(base_fact, "_cohort"),
        require_bound=True,
    )
    if base_cohort.fingerprint != cohort.fingerprint:
        raise ReconciliationInputError(
            "BaseFact and final fact cohorts differ"
        )
    source_decision = object.__getattribute__(base_fact, "_source_decision")
    if type(source_decision) is not FactDecision:
        raise ReconciliationInputError(
            "BaseFact must retain an exact source FactDecision"
        )
    _validate_decision_candidate_context(source_decision)
    source_cohort = _validate_reconciliation_cohort(
        object.__getattribute__(source_decision, "_cohort"),
        require_bound=True,
    )
    if source_cohort.fingerprint != cohort.fingerprint:
        raise ReconciliationInputError(
            "BaseFact and source decision cohorts differ"
        )
    canonical_base_fact = _base_fact_from_authenticated_decision(
        source_decision
    )
    if (
        _base_fact_stage_payload(base_fact)
        != _base_fact_stage_payload(canonical_base_fact)
    ):
        raise ReconciliationInputError(
            "BaseFact fields do not match its source decision"
        )
    source_auth = object.__getattribute__(source_decision, "_stage_auth")
    source_stage = object.__getattribute__(source_auth, "stage")
    if source_stage not in (
        _ReconciliationStage.CLASSIFIED,
        _ReconciliationStage.COORDINATED,
    ):
        raise ReconciliationInputError(
            "BaseFact source decision stage is invalid"
        )
    base_auth = _validate_stage_authorization(
        base_fact,
        cohort=base_cohort,
        expected_stage=_ReconciliationStage.BASE_FACT,
        expected_previous_stage=source_stage,
        payload=_base_fact_stage_payload(base_fact),
        graph_build_token=graph_token,
    )
    if (
        object.__getattribute__(base_auth, "parent_fingerprint")
        != object.__getattribute__(source_auth, "fingerprint")
    ):
        raise ReconciliationInputError(
            "BaseFact parent does not match its source decision"
        )
    if _base_fact_stage_payload(base_fact) != _final_fact_base_payload(fact):
        raise ReconciliationInputError(
            "final fact non-derived fields do not match its BaseFact"
        )
    fact_cohort = _validate_reconciliation_cohort(
        object.__getattribute__(fact, "_cohort"),
        require_bound=True,
    )
    if fact_cohort.fingerprint != cohort.fingerprint:
        raise ReconciliationInputError(
            "final fact and graph cohorts differ"
        )
    fact_auth = _validate_stage_authorization(
        fact,
        cohort=fact_cohort,
        expected_stage=_ReconciliationStage.MATERIALIZED_FACT,
        expected_previous_stage=_ReconciliationStage.BASE_FACT,
        payload=_fact_stage_payload(fact),
        graph_build_token=graph_token,
    )
    if (
        object.__getattribute__(fact_auth, "parent_fingerprint")
        != object.__getattribute__(base_auth, "fingerprint")
    ):
        raise ReconciliationInputError(
            "final fact parent does not match its BaseFact"
        )
    return base_fact


def _read_record_fields(
    record: object,
    field_names: tuple[str, ...],
    record_name: str,
) -> dict[str, object]:
    try:
        return {
            field_name: object.__getattribute__(record, field_name)
            for field_name in field_names
        }
    except Exception as error:
        raise ReconciliationInvariantError(
            f"{record_name} must contain complete canonical fields"
        ) from error


def _validate_exact_text_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    if type(value) is not tuple:
        _fail_invariant(f"{field_name} must be an exact tuple")
    if not allow_empty and not value:
        _fail_invariant(f"{field_name} must be non-empty")
    if any(not _is_exact_required_text(item) for item in value):
        _fail_invariant(
            f"{field_name} must contain non-empty NUL-free strings"
        )
    if value != tuple(sorted(set(value))):
        _fail_invariant(f"{field_name} must be sorted and unique")


def _relation_key(relation: RelationRecord) -> tuple[str, str, str]:
    return (
        relation.relation_type.value,
        relation.from_fact_id,
        relation.to_fact_id,
    )


def _validate_relation_record_types(
    relations: tuple[object, ...],
) -> None:
    for relation in relations:
        if type(relation) is not RelationRecord:
            _fail_invariant(
                "relations must contain only exact RelationRecord records"
            )
        values = _read_record_fields(
            relation,
            (
                "relation_id",
                "relation_type",
                "from_fact_id",
                "to_fact_id",
                "relation_reason",
                "candidate_ids",
                "evidence_refs",
            ),
            "RelationRecord",
        )
        if type(values["relation_type"]) is not RelationType:
            _fail_invariant("relation_type must be an exact RelationType")
        for field_name in ("from_fact_id", "to_fact_id"):
            if not _is_exact_required_text(values[field_name]):
                _fail_invariant(
                    f"{field_name} must be an exact non-empty NUL-free string"
                )
        for field_name in ("relation_id", "relation_reason"):
            if not _is_exact_required_text(values[field_name]):
                _fail_invariant(
                    f"{field_name} must be an exact non-empty NUL-free string"
                )
        if not values["candidate_ids"] or not values["evidence_refs"]:
            _fail_invariant("relation must have non-empty provenance")
        _validate_exact_text_tuple(
            values["candidate_ids"],
            "candidate_ids",
            allow_empty=False,
        )
        _validate_exact_text_tuple(
            values["evidence_refs"],
            "evidence_refs",
            allow_empty=False,
        )


def _validate_fact_record_types(facts: tuple[object, ...]) -> None:
    evidence_fields = (
        "selected_value",
        "confidence",
        "valid_from",
        "resolved_at",
    )
    required_text_fields = ("fact_id", "subject", "predicate", "reason")
    required_tuple_fields = ("evidence_refs", "candidate_ids")
    optional_tuple_fields = (
        "superseded_candidate_ids",
        "conflict_candidate_ids",
        "activation_witness_candidate_ids",
        "relation_ids",
        "supersedes_fact_ids",
        "superseded_by_fact_ids",
        "conflict_fact_ids",
    )
    field_names = (
        *required_text_fields,
        *evidence_fields,
        "status",
        "resolution_method",
        "requires_human_review",
        *required_tuple_fields,
        *optional_tuple_fields,
    )
    for fact in facts:
        if type(fact) is not ReconciledFact:
            _fail_invariant(
                "facts must contain only exact ReconciledFact records"
            )
        values = _read_record_fields(
            fact,
            field_names,
            "ReconciledFact",
        )
        for field_name in required_text_fields:
            if not _is_exact_required_text(values[field_name]):
                _fail_invariant(
                    f"fact {field_name} must be an exact non-empty NUL-free string"
                )
        for field_name in evidence_fields:
            if type(values[field_name]) is not EvidenceValue:
                _fail_invariant(
                    f"fact {field_name} must be an exact EvidenceValue"
                )
        evidence_kinds = {
            "selected_value": "json",
            "confidence": "float",
            "valid_from": "string",
            "resolved_at": "string",
        }
        for field_name, known_kind in evidence_kinds.items():
            try:
                _strict_evidence_signature(
                    values[field_name],
                    field_name,
                    known_kind,
                )
            except ReconciliationInputError as error:
                raise ReconciliationInvariantError(
                    f"fact {field_name} must be canonical"
                ) from error
        if type(values["status"]) is not ReconciliationStatus:
            _fail_invariant("fact status must be an exact ReconciliationStatus")
        if type(values["resolution_method"]) is not ResolutionMethod:
            _fail_invariant(
                "fact resolution_method must be an exact ResolutionMethod"
            )
        if type(values["requires_human_review"]) is not bool:
            _fail_invariant("fact requires_human_review must be an exact boolean")
        if not values["candidate_ids"] or not values["evidence_refs"]:
            _fail_invariant(
                f"fact {values['fact_id']} must have non-empty provenance"
            )
        for field_name in required_tuple_fields:
            _validate_exact_text_tuple(
                values[field_name],
                f"fact {field_name}",
                allow_empty=False,
            )
        for field_name in optional_tuple_fields:
            _validate_exact_text_tuple(
                values[field_name],
                f"fact {field_name}",
                allow_empty=True,
            )


def _validate_graph_stage_authentication(
    graph: RelationGraph,
    facts: tuple[ReconciledFact, ...],
    relations: tuple[RelationRecord, ...],
) -> None:
    try:
        cohort = object.__getattribute__(graph, "_cohort")
        checked_cohort = _validate_reconciliation_cohort(
            cohort,
            require_bound=True,
        )
        graph_auth = object.__getattribute__(graph, "_stage_auth")
        graph_token = object.__getattribute__(
            graph_auth,
            "graph_build_token",
        )
        if not _is_exact_required_text(graph_token):
            raise ReconciliationInputError("graph build token is invalid")
        base_auth_by_fact_id: dict[str, object] = {}
        for fact in facts:
            base_fact = _validate_final_fact_base_stage(
                fact,
                checked_cohort,
                graph_token,
            )
            base_auth_by_fact_id[fact.fact_id] = object.__getattribute__(
                base_fact,
                "_stage_auth",
            )
        for relation in relations:
            relation_cohort = _validate_reconciliation_cohort(
                object.__getattribute__(relation, "_cohort"),
                require_bound=True,
            )
            if relation_cohort.fingerprint != checked_cohort.fingerprint:
                raise ReconciliationInputError(
                    "relation does not belong to graph cohort"
                )
            relation_auth = _validate_stage_authorization(
                relation,
                cohort=relation_cohort,
                expected_stage=_ReconciliationStage.MATERIALIZED_RELATION,
                expected_previous_stage=_ReconciliationStage.BASE_FACT,
                payload=_relation_stage_payload(relation),
                graph_build_token=graph_token,
            )
            if object.__getattribute__(
                relation_auth,
                "parent_fingerprint",
            ) != _relation_parent_fingerprint(
                base_auth_by_fact_id,
                relation.from_fact_id,
                relation.to_fact_id,
            ):
                raise ReconciliationInputError(
                    "relation parent does not match endpoint BaseFacts"
                )
        _validate_graph_lookup_authentication(
            graph,
            facts,
            checked_cohort,
            graph_token,
        )
        graph_previous_stage = (
            _ReconciliationStage.MATERIALIZED_RELATION
            if relations
            else _ReconciliationStage.MATERIALIZED_FACT
        )
        _validate_stage_authorization(
            graph,
            cohort=checked_cohort,
            expected_stage=_ReconciliationStage.MATERIALIZED_GRAPH,
            expected_previous_stage=graph_previous_stage,
            payload=_graph_stage_payload(graph),
            graph_build_token=graph_token,
        )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        raise ReconciliationInvariantError(
            "relation graph stage or graph build authentication is invalid"
        ) from error


def _validate_graph_lookup_authentication(
    graph: RelationGraph,
    facts: tuple[ReconciledFact, ...],
    cohort: _ReconciliationCohort,
    graph_token: str,
) -> None:
    entries = _graph_lookup_stage_entries(graph)
    facts_by_id = {fact.fact_id: fact for fact in facts}
    if tuple(entry[0] for entry in entries) != tuple(sorted(facts_by_id)):
        raise ReconciliationInputError(
            "graph fact lookup keys must exactly match graph fact IDs"
        )
    for fact_id, lookup_fact, _ in entries:
        lookup_cohort = _validate_reconciliation_cohort(
            object.__getattribute__(lookup_fact, "_cohort"),
            require_bound=True,
        )
        if (
            lookup_fact.fact_id != fact_id
            or lookup_cohort.fingerprint != cohort.fingerprint
        ):
            raise ReconciliationInputError(
                "graph fact lookup member cohort or ID is invalid"
            )
        _validate_final_fact_base_stage(
            lookup_fact,
            lookup_cohort,
            graph_token,
        )
        if lookup_fact is not facts_by_id[fact_id]:
            raise ReconciliationInputError(
                "graph fact lookup must bind each canonical fact member"
            )


def _validate_endpoint_and_identity(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> None:
    seen_edges: set[tuple[str, str, str]] = set()
    for relation in sorted(relations, key=_relation_key):
        if (
            relation.from_fact_id not in facts_by_id
            or relation.to_fact_id not in facts_by_id
        ):
            _fail_invariant(
                "every relation endpoint exists in the result fact set: "
                f"{relation.from_fact_id} -> {relation.to_fact_id}"
            )
        if relation.from_fact_id == relation.to_fact_id:
            _fail_invariant(
                "relation graph must not contain a self-edge: "
                f"{relation.from_fact_id}"
            )
        try:
            expected_id = make_relation_id(
                relation.relation_type,
                relation.from_fact_id,
                relation.to_fact_id,
            )
        except (AttributeError, ReconciliationInputError) as error:
            raise ReconciliationInvariantError(
                "relation identity fields must be canonical"
            ) from error
        if relation.relation_id != expected_id:
            _fail_invariant(
                "relation_id must match canonical identity: "
                f"{relation.from_fact_id} -> {relation.to_fact_id}"
            )
        edge = _relation_key(relation)
        if edge in seen_edges:
            _fail_invariant(
                "duplicate relation edge: "
                f"{edge[0]} {edge[1]} -> {edge[2]}"
            )
        seen_edges.add(edge)


def _validate_conflict_relations(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> None:
    allowed_reasons = {
        "unresolved competing current facts",
        "contradictory explicit supersedes declarations",
    }
    for relation in relations:
        if relation.relation_type is not RelationType.SUPERSEDES:
            continue
        if any(
            facts_by_id[fact_id].status is ReconciliationStatus.CONFLICTED
            for fact_id in (relation.from_fact_id, relation.to_fact_id)
        ):
            _fail_invariant(
                "CONFLICTED facts cannot participate in SUPERSEDES edges: "
                f"{relation.from_fact_id} -> {relation.to_fact_id}"
            )
    conflicts = {
        (relation.from_fact_id, relation.to_fact_id): relation
        for relation in relations
        if relation.relation_type is RelationType.CONFLICTS
    }
    for (source, target), relation in sorted(conflicts.items()):
        endpoints = (facts_by_id[source], facts_by_id[target])
        if any(
            fact.status is not ReconciliationStatus.CONFLICTED
            or not fact.requires_human_review
            for fact in endpoints
        ):
            _fail_invariant(
                "CONFLICTS edges require exact CONFLICTED endpoints and "
                f"require human review: {source} -> {target}"
            )
        source_fact, target_fact = endpoints
        if (
            source_fact.subject != target_fact.subject
            or source_fact.predicate != target_fact.predicate
        ):
            _fail_invariant(
                "CONFLICTS endpoints must share one exact subject and predicate: "
                f"{source} -> {target}"
            )
        expected_candidate_ids = tuple(
            sorted(set((*source_fact.candidate_ids, *target_fact.candidate_ids)))
        )
        expected_evidence_refs = tuple(
            sorted(set((*source_fact.evidence_refs, *target_fact.evidence_refs)))
        )
        if (
            relation.candidate_ids != expected_candidate_ids
            or relation.evidence_refs != expected_evidence_refs
        ):
            _fail_invariant(
                "CONFLICTS provenance must exactly equal the normalized endpoint "
                f"provenance union: {source} -> {target}"
            )
        if (
            relation.relation_reason not in allowed_reasons
            or source_fact.reason != relation.relation_reason
            or target_fact.reason != relation.relation_reason
        ):
            _fail_invariant(
                "CONFLICTS relation and endpoints must share a canonical conflict "
                f"reason: {source} -> {target}"
            )
        inverse = conflicts.get((target, source))
        if inverse is None:
            _fail_invariant(
                "CONFLICTS edge must have a symmetric inverse: "
                f"{source} -> {target}"
            )
        if (
            relation.relation_reason != inverse.relation_reason
            or relation.candidate_ids != inverse.candidate_ids
            or relation.evidence_refs != inverse.evidence_refs
        ):
            _fail_invariant(
                "symmetric CONFLICTS edges must have the same reason and provenance: "
                f"{source} <-> {target}"
            )
    conflicted_by_predicate: dict[tuple[str, str], list[str]] = {}
    for fact_id, fact in facts_by_id.items():
        if fact.status is ReconciliationStatus.CONFLICTED:
            conflicted_by_predicate.setdefault(
                (fact.subject, fact.predicate),
                [],
            ).append(fact_id)
    for fact_ids in conflicted_by_predicate.values():
        ordered = tuple(sorted(fact_ids))
        member_ids = set(ordered)
        expected_edges = {
            (source, target)
            for source in ordered
            for target in ordered
            if source != target
        }
        actual_edges = {
            (source, target)
            for source, target in conflicts
            if source in member_ids and target in member_ids
        }
        if actual_edges != expected_edges:
            _fail_invariant(
                "same-predicate CONFLICTED facts must form a complete directed clique"
            )


def _validate_supersedes_acyclic(
    fact_ids: tuple[str, ...],
    relations: tuple[RelationRecord, ...],
) -> None:
    supersedes: dict[str, set[str]] = {fact_id: set() for fact_id in fact_ids}
    for relation in relations:
        if relation.relation_type is RelationType.SUPERSEDES:
            supersedes[relation.from_fact_id].add(relation.to_fact_id)

    colors: dict[str, str] = {}
    for root in fact_ids:
        if colors.get(root) == "black":
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                colors[node] = "black"
                continue
            if colors.get(node) == "gray":
                _fail_invariant("SUPERSEDES graph must be acyclic")
            if colors.get(node) == "black":
                continue
            colors[node] = "gray"
            stack.append((node, True))
            for child in reversed(sorted(supersedes[node])):
                if colors.get(child) == "gray":
                    _fail_invariant("SUPERSEDES graph must be acyclic")
                if colors.get(child) != "black":
                    stack.append((child, False))


def _expected_fact_indexes(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> dict[str, _FactIndexes]:
    return _derive_indexes(dict(facts_by_id), relations)


def _validate_derived_indexes(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> None:
    expected = _expected_fact_indexes(facts_by_id, relations)
    fields_and_values = (
        ("relation_ids", lambda value: value.relation_ids),
        ("supersedes_fact_ids", lambda value: value.supersedes),
        ("superseded_by_fact_ids", lambda value: value.superseded_by),
        ("conflict_fact_ids", lambda value: value.conflicts),
        (
            "conflict_candidate_ids",
            lambda value: value.conflict_candidates,
        ),
    )
    for fact_id in sorted(facts_by_id):
        fact = facts_by_id[fact_id]
        for field_name, expected_value in fields_and_values:
            canonical = tuple(sorted(expected_value(expected[fact_id])))
            if getattr(fact, field_name) != canonical:
                _fail_invariant(
                    f"fact {fact_id} {field_name} must match canonical relation edges"
                )


def _validate_fact_statuses(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> None:
    incoming_supersedes: set[str] = set()
    outgoing_supersedes: set[str] = set()
    conflict_sources: set[str] = set()
    for relation in relations:
        if relation.relation_type is RelationType.SUPERSEDES:
            outgoing_supersedes.add(relation.from_fact_id)
            incoming_supersedes.add(relation.to_fact_id)
        else:
            conflict_sources.add(relation.from_fact_id)

    for fact_id in sorted(facts_by_id):
        fact = facts_by_id[fact_id]
        if (
            fact.status is ReconciliationStatus.ACTIVE
            and fact_id in incoming_supersedes
        ):
            _fail_invariant(
                f"ACTIVE fact {fact_id} must have no incoming SUPERSEDES edge"
            )
        if (
            fact.status is ReconciliationStatus.SUPERSEDED
            and fact_id not in incoming_supersedes
        ):
            _fail_invariant(
                f"SUPERSEDED fact {fact_id} must have an incoming SUPERSEDES edge"
            )
        if fact.status is ReconciliationStatus.CONFLICTED:
            if fact_id not in conflict_sources:
                _fail_invariant(
                    f"CONFLICTED fact {fact_id} must have a conflict edge"
                )
            if not fact.requires_human_review:
                _fail_invariant(
                    f"CONFLICTED fact {fact_id} must require human review"
                )
        if (
            fact.status is ReconciliationStatus.DEPRECATED
            and fact_id in outgoing_supersedes
        ):
            _fail_invariant(
                f"DEPRECATED fact {fact_id} cannot be selected as a replacement fact"
            )
        if (
            fact.status is ReconciliationStatus.ACTIVE
            and fact.selected_value.status is not EvidenceStatus.KNOWN
        ):
            _fail_invariant(
                f"ACTIVE fact {fact_id} must have a KNOWN selected_value"
            )


def _validate_provenance(
    facts_by_id: Mapping[str, ReconciledFact],
    relations: tuple[RelationRecord, ...],
) -> None:
    for fact_id in sorted(facts_by_id):
        fact = facts_by_id[fact_id]
        if not fact.candidate_ids or not fact.evidence_refs:
            _fail_invariant(f"fact {fact_id} must have non-empty provenance")
    for relation in sorted(relations, key=_relation_key):
        if not relation.candidate_ids or not relation.evidence_refs:
            _fail_invariant(
                "relation must have non-empty provenance: "
                f"{relation.from_fact_id} -> {relation.to_fact_id}"
            )


def _validate_candidate_lineage(
    facts_by_id: Mapping[str, ReconciledFact],
) -> None:
    """Recheck candidate invariants retained after canonical classification.

    ``_fact_from_decision`` recomputes each decision from its private candidate
    context before that context is discarded.  At graph validation time the
    representable checks are exact membership, witness presence, and internal
    lineage consistency.
    """

    for fact_id in sorted(facts_by_id):
        fact = facts_by_id[fact_id]
        candidates = set(fact.candidate_ids)
        superseded = set(fact.superseded_candidate_ids)
        witnesses = set(fact.activation_witness_candidate_ids)
        if not superseded <= candidates:
            _fail_invariant(
                f"fact {fact_id} candidate_ids must retain all merged candidates"
            )
        if not witnesses <= candidates:
            _fail_invariant(
                f"fact {fact_id} activation witness must appear in candidate_ids"
            )
        if witnesses & superseded:
            _fail_invariant(
                f"fact {fact_id} activation witness cannot be internally superseded"
            )
        if fact.status is ReconciliationStatus.ACTIVE:
            if not witnesses:
                _fail_invariant(
                    f"ACTIVE fact {fact_id} requires an activation witness"
                )
            if fact.confidence.status is not EvidenceStatus.KNOWN:
                _fail_invariant(
                    f"ACTIVE fact {fact_id} activation witness confidence must be KNOWN"
                )
        elif witnesses:
            _fail_invariant(
                f"non-ACTIVE fact {fact_id} cannot retain an activation witness"
            )
        if (
            fact.resolution_method is ResolutionMethod.SAME_VALUE_REACTIVATION
            and not superseded
        ):
            _fail_invariant(
                f"same-value fact {fact_id} must identify an internally "
                "replaced candidate"
            )


def _validate_graph_fields(
    facts: tuple[ReconciledFact, ...],
    relations: tuple[RelationRecord, ...],
) -> None:
    _validate_fact_record_types(facts)
    _validate_relation_record_types(relations)
    facts_by_id: dict[str, ReconciledFact] = {}
    for fact in facts:
        if fact.fact_id in facts_by_id:
            _fail_invariant(f"fact IDs must be unique: {fact.fact_id}")
        facts_by_id[fact.fact_id] = fact
    _validate_endpoint_and_identity(facts_by_id, relations)
    _validate_conflict_relations(facts_by_id, relations)
    _validate_derived_indexes(facts_by_id, relations)
    _validate_fact_statuses(facts_by_id, relations)
    _validate_provenance(facts_by_id, relations)
    _validate_supersedes_acyclic(tuple(sorted(facts_by_id)), relations)
    _validate_candidate_lineage(facts_by_id)


def validate_relation_graph(graph: RelationGraph) -> None:
    """Reject the deterministic first violation in a materialized graph."""

    if type(graph) is not RelationGraph:
        _fail_invariant("graph must be an exact RelationGraph")
    values = _read_record_fields(
        graph,
        ("facts", "relations"),
        "RelationGraph",
    )
    facts = _snapshot_validation_sequence(values["facts"], "graph facts")
    relations = _snapshot_validation_sequence(
        values["relations"],
        "graph relations",
    )
    _validate_graph_fields(facts, relations)
    _validate_graph_stage_authentication(graph, facts, relations)


def _partition_key(value: object) -> ReconciliationStatus | None:
    if type(value) is ReconciliationStatus:
        return value
    if type(value) is str:
        try:
            return ReconciliationStatus(value)
        except ValueError:
            return None
    return None


def _validate_status_partitions(
    facts: tuple[ReconciledFact, ...],
    partitions: object,
) -> None:
    if not isinstance(partitions, Mapping):
        _fail_invariant("partitions must be a mapping")
    try:
        partition_items = tuple(partitions.items())
    except Exception as error:
        raise ReconciliationInvariantError(
            "partitions must be a stable mapping"
        ) from error
    normalized: dict[ReconciliationStatus, tuple[ReconciledFact, ...]] = {}
    for key, partition in partition_items:
        status = _partition_key(key)
        if status is None or status in normalized:
            _fail_invariant("status partitions must use each canonical status once")
        normalized[status] = _snapshot_validation_sequence(
            partition,
            f"{status.value} status partition",
        )

    membership: dict[str, int] = {fact.fact_id: 0 for fact in facts}
    facts_by_id = {fact.fact_id: fact for fact in facts}
    for status in ReconciliationStatus:
        for fact in normalized.get(status, ()):
            if type(fact) is not ReconciledFact:
                _fail_invariant(
                    "status partitions must contain exact ReconciledFact records"
                )
            try:
                fact_id = object.__getattribute__(fact, "fact_id")
                fact_status = object.__getattribute__(fact, "status")
            except (AttributeError, TypeError) as error:
                raise ReconciliationInvariantError(
                    "status partition member must be complete"
                ) from error
            canonical = facts_by_id.get(fact_id)
            if canonical is None or fact_status is not status:
                _fail_invariant(
                    "each fact must appear in exactly one status partition "
                    "matching its status"
                )
            try:
                canonical_cohort = _validate_reconciliation_cohort(
                    object.__getattribute__(canonical, "_cohort"),
                    require_bound=True,
                )
                canonical_auth = object.__getattribute__(
                    canonical,
                    "_stage_auth",
                )
                graph_token = object.__getattribute__(
                    canonical_auth,
                    "graph_build_token",
                )
                _validate_final_fact_base_stage(
                    canonical,
                    canonical_cohort,
                    graph_token,
                )
                partition_cohort = _validate_reconciliation_cohort(
                    object.__getattribute__(fact, "_cohort"),
                    require_bound=True,
                )
                if partition_cohort.fingerprint != canonical_cohort.fingerprint:
                    raise ReconciliationInputError(
                        "partition and canonical fact cohorts differ"
                    )
                _validate_final_fact_base_stage(
                    fact,
                    partition_cohort,
                    graph_token,
                )
            except (AttributeError, TypeError, ReconciliationInputError) as error:
                raise ReconciliationInvariantError(
                    "status partition member must be an authenticated "
                    "materialized fact"
                ) from error
            if fact is not canonical:
                _fail_invariant(
                    "status partition must bind each canonical member"
                )
            membership[fact_id] += 1
    if any(count != 1 for count in membership.values()):
        _fail_invariant(
            "each fact must appear in exactly one status partition matching "
            "its status"
        )


def _validate_warning_records(warnings: tuple[object, ...]) -> None:
    for warning in warnings:
        if type(warning) is not ReconciliationWarning:
            _fail_invariant(
                "warnings must contain only exact ReconciliationWarning records"
            )
        values = _read_record_fields(
            warning,
            (
                "code",
                "message",
                "candidate_ids",
                "evidence_refs",
                "requires_human_review",
            ),
            "ReconciliationWarning",
        )
        if type(values["code"]) is not WarningCode:
            _fail_invariant("warning code must be an exact WarningCode")
        if not _is_exact_required_text(values["message"]):
            _fail_invariant(
                "warning message must be an exact non-empty NUL-free string"
            )
        if not values["candidate_ids"] or not values["evidence_refs"]:
            _fail_invariant("warning must have non-empty provenance")
        for field_name in ("candidate_ids", "evidence_refs"):
            _validate_exact_text_tuple(
                values[field_name],
                f"warning {field_name}",
                allow_empty=False,
            )
        if type(values["requires_human_review"]) is not bool:
            _fail_invariant(
                "warning requires_human_review must be an exact boolean"
            )


def _validate_unresolved_records(unresolved: tuple[object, ...]) -> None:
    for record in unresolved:
        if type(record) is not UnresolvedCandidate:
            _fail_invariant(
                "unresolved must contain only exact UnresolvedCandidate records"
            )
        values = _read_record_fields(
            record,
            (
                "candidate_id",
                "subject",
                "predicate",
                "evidence_status",
                "reason",
                "source_type",
                "source_ref",
                "field_source",
                "related_fact_id",
            ),
            "UnresolvedCandidate",
        )
        for field_name in (
            "candidate_id",
            "subject",
            "predicate",
            "reason",
            "source_ref",
            "field_source",
            "related_fact_id",
        ):
            if not _is_exact_required_text(values[field_name]):
                _fail_invariant(
                    f"unresolved {field_name} must be an exact non-empty "
                    "NUL-free string"
                )
        if (
            type(values["evidence_status"]) is not EvidenceStatus
            or values["evidence_status"] is EvidenceStatus.KNOWN
        ):
            _fail_invariant(
                "unresolved evidence_status must be UNKNOWN or UNAVAILABLE"
            )
        if type(values["source_type"]) is not SourceType:
            _fail_invariant("unresolved source_type must be an exact SourceType")


def _canonical_predicate_replay(
    source_decisions: object,
) -> tuple[
    tuple[FactDecision, ...],
    tuple[RelationRequest, ...],
    tuple[FactDecision, ...],
]:
    decision_snapshot = _snapshot_validation_sequence(
        source_decisions,
        "source_decisions",
    )
    grouped: dict[tuple[str, str], list[FactDecision]] = {}
    seen_fact_ids: set[str] = set()
    for decision in decision_snapshot:
        if type(decision) is not FactDecision:
            _fail_invariant(
                "source_decisions must contain only exact FactDecision records"
            )
        try:
            canonical = _validate_decision_candidate_context(decision)
        except Exception as error:
            raise ReconciliationInvariantError(
                "source_decisions must retain canonical candidate context"
            ) from error
        if canonical.fact_id in seen_fact_ids:
            _fail_invariant(
                "source_decisions must have unique canonical fact IDs: "
                f"{canonical.fact_id}"
            )
        seen_fact_ids.add(canonical.fact_id)
        grouped.setdefault(
            (canonical.subject, canonical.predicate),
            [],
        ).append(canonical)

    try:
        _validate_complete_decision_cohort(decision_snapshot)
    except (ReconciliationInputError, AttributeError, TypeError, ValueError) as error:
        raise ReconciliationInvariantError(
            "source_decisions must retain one complete reconciliation cohort"
        ) from error

    expected_decisions: list[FactDecision] = []
    expected_requests: list[RelationRequest] = []
    for key in sorted(grouped):
        canonical_group = tuple(
            sorted(grouped[key], key=lambda item: item.fact_id)
        )
        try:
            policy = canonical_group[0]._candidate_context.policy
            outcome = resolve_predicate(
                canonical_group,
                policy,
            )
        except Exception as error:
            raise ReconciliationInvariantError(
                "source_decisions could not produce canonical predicate replay"
            ) from error
        expected_decisions.extend(outcome.decisions)
        expected_requests.extend(outcome.relation_requests)
    return (
        tuple(sorted(expected_decisions, key=lambda item: item.fact_id)),
        tuple(
            sorted(
                expected_requests,
                key=lambda item: (
                    item.relation_type.value,
                    item.from_fact_id,
                    item.to_fact_id,
                ),
            )
        ),
        tuple(sorted(decision_snapshot, key=lambda item: item.fact_id)),
    )


def _validate_canonical_replay(
    facts: tuple[ReconciledFact, ...],
    relations: tuple[RelationRecord, ...],
    source_decisions: object,
) -> tuple[FactDecision, ...]:
    (
        expected_decisions,
        expected_requests,
        authenticated_source_decisions,
    ) = _canonical_predicate_replay(source_decisions)
    expected_facts = tuple(
        _base_fact_from_authenticated_decision(decision)
        for decision in expected_decisions
    )
    if not expected_decisions:
        _fail_invariant("source_decisions must retain a current cohort")
    _validate_replayed_payload_fields(
        facts,
        relations,
        expected_facts,
        expected_requests,
    )
    try:
        source_cohort = _validate_reconciliation_cohort(
            expected_decisions[0]._cohort,
            require_bound=True,
        )
        member_auths = tuple(
            object.__getattribute__(member, "_stage_auth")
            for member in (*facts, *relations)
        )
        graph_tokens = {
            object.__getattribute__(auth, "graph_build_token")
            for auth in member_auths
        }
        if len(graph_tokens) != 1:
            raise ReconciliationInputError(
                "result members must share one graph build token"
            )
        graph_token = next(iter(graph_tokens))
        if not _is_exact_required_text(graph_token):
            raise ReconciliationInputError("graph build token is invalid")
        for fact in facts:
            _validate_final_fact_base_stage(
                fact,
                source_cohort,
                graph_token,
            )
        for relation in relations:
            relation_cohort = _validate_reconciliation_cohort(
                relation._cohort,
                require_bound=True,
            )
            if relation_cohort.fingerprint != source_cohort.fingerprint:
                raise ReconciliationInputError(
                    "relation and source decision cohorts differ"
                )
            _validate_stage_authorization(
                relation,
                cohort=relation_cohort,
                expected_stage=_ReconciliationStage.MATERIALIZED_RELATION,
                expected_previous_stage=_ReconciliationStage.BASE_FACT,
                payload=_relation_stage_payload(relation),
                graph_build_token=graph_token,
            )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        raise ReconciliationInvariantError(
            "result cohort, stage, or source decision parent authentication "
            "is invalid"
        ) from error
    _validate_source_parent_chain(
        facts,
        relations,
        authenticated_source_decisions,
    )
    return expected_decisions


def _unresolved_payload(record: UnresolvedCandidate) -> tuple[str, ...]:
    return (
        record.candidate_id,
        record.subject,
        record.predicate,
        record.evidence_status.value,
        record.reason,
        record.source_type.value,
        record.source_ref,
        record.field_source,
        record.related_fact_id,
    )


def _validate_unresolved_provenance(
    unresolved: tuple[object, ...],
    canonical_decisions: tuple[FactDecision, ...],
) -> None:
    """Bind every unresolved record to one authenticated non-known value."""

    try:
        expected = _derive_unresolved_candidates(canonical_decisions)
        actual_payloads = tuple(
            sorted(_unresolved_payload(record) for record in unresolved)
        )
        expected_payloads = tuple(
            sorted(_unresolved_payload(record) for record in expected)
        )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        raise ReconciliationInvariantError(
            "unresolved records must match authenticated source candidates"
        ) from error
    if actual_payloads != expected_payloads:
        _fail_invariant(
            "unresolved records must match authenticated source candidates exactly"
        )


def _validate_replayed_payload_fields(
    facts: tuple[ReconciledFact, ...],
    relations: tuple[RelationRecord, ...],
    expected_facts: tuple[BaseFact, ...],
    expected_requests: tuple[RelationRequest, ...],
) -> None:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    expected_by_id = {fact.fact_id: fact for fact in expected_facts}
    if set(facts_by_id) != set(expected_by_id):
        _fail_invariant(
            "final facts and canonical predicate replay must have exact fact IDs"
        )

    fields_to_compare = (
        "fact_id",
        "subject",
        "predicate",
        "selected_value",
        "status",
        "confidence",
        "reason",
        "evidence_refs",
        "candidate_ids",
        "superseded_candidate_ids",
        "valid_from",
        "resolved_at",
        "resolution_method",
        "requires_human_review",
        "activation_witness_candidate_ids",
    )
    for fact_id in sorted(facts_by_id):
        fact = facts_by_id[fact_id]
        expected = expected_by_id[fact_id]
        for field_name in fields_to_compare:
            if field_name in (
                "selected_value",
                "confidence",
                "valid_from",
                "resolved_at",
            ):
                known_kind = {
                    "selected_value": "json",
                    "confidence": "float",
                    "valid_from": "string",
                    "resolved_at": "string",
                }[field_name]
                differs = _strict_evidence_signature(
                    getattr(fact, field_name),
                    field_name,
                    known_kind,
                ) != _strict_evidence_signature(
                    getattr(expected, field_name),
                    field_name,
                    known_kind,
                )
            else:
                differs = getattr(fact, field_name) != getattr(
                    expected,
                    field_name,
                )
            if differs:
                suffix = (
                    "canonical source decision from canonical predicate replay"
                    if field_name
                    in (
                        "candidate_ids",
                        "superseded_candidate_ids",
                        "activation_witness_candidate_ids",
                    )
                    else "canonical predicate replay"
                )
                _fail_invariant(
                    f"fact {fact_id} {field_name} must exactly match its {suffix}"
                )

    expected_relations = tuple(
        sorted(
            (
                RelationRecord.create(
                    request.relation_type,
                    request.from_fact_id,
                    request.to_fact_id,
                    request.reason,
                    request.candidate_ids,
                    request.evidence_refs,
                )
                for request in expected_requests
            ),
            key=_relation_key,
        )
    )
    if tuple(sorted(relations, key=_relation_key)) != expected_relations:
        _fail_invariant(
            "relations must exactly match the canonical predicate replay"
        )


def _validate_source_parent_chain(
    facts: tuple[ReconciledFact, ...],
    relations: tuple[RelationRecord, ...],
    source_decisions: tuple[FactDecision, ...],
) -> None:
    try:
        source_by_fact_id = {
            decision.fact_id: decision
            for decision in source_decisions
        }
        base_auth_by_fact_id: dict[str, object] = {}
        for fact in facts:
            fact_auth = object.__getattribute__(fact, "_stage_auth")
            base_fact = object.__getattribute__(fact, "_base_fact")
            if type(base_fact) is not BaseFact:
                raise ReconciliationInputError(
                    "materialized fact has no BaseFact parent"
                )
            source_decision = source_by_fact_id.get(fact.fact_id)
            if source_decision is None or object.__getattribute__(
                base_fact,
                "_source_decision",
            ) is not source_decision:
                raise ReconciliationInputError(
                    "BaseFact parent does not bind the supplied source decision"
                )
            source_auth = object.__getattribute__(
                source_decision,
                "_stage_auth",
            )
            base_auth = object.__getattribute__(base_fact, "_stage_auth")
            source_stage = object.__getattribute__(source_auth, "stage")
            if (
                source_stage not in (
                    _ReconciliationStage.CLASSIFIED,
                    _ReconciliationStage.COORDINATED,
                )
                or object.__getattribute__(base_auth, "previous_stage")
                is not source_stage
                or object.__getattribute__(base_auth, "parent_fingerprint")
                != object.__getattribute__(source_auth, "fingerprint")
                or object.__getattribute__(fact_auth, "previous_stage")
                is not _ReconciliationStage.BASE_FACT
                or object.__getattribute__(fact_auth, "parent_fingerprint")
                != object.__getattribute__(base_auth, "fingerprint")
            ):
                raise ReconciliationInputError(
                    "materialized fact BaseFact parent chain is invalid"
                )
            base_auth_by_fact_id[fact.fact_id] = base_auth
        for relation in relations:
            relation_auth = object.__getattribute__(relation, "_stage_auth")
            if (
                object.__getattribute__(relation_auth, "parent_fingerprint")
                != _relation_parent_fingerprint(
                    base_auth_by_fact_id,
                    relation.from_fact_id,
                    relation.to_fact_id,
                )
            ):
                raise ReconciliationInputError(
                    "materialized relation parent does not match source decisions"
                )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        raise ReconciliationInvariantError(
            "result source decision parent chain is invalid"
        ) from error


def validate_result_fields(
    *,
    facts: Sequence[ReconciledFact],
    relations: Sequence[RelationRecord],
    partitions: Mapping[object, Sequence[ReconciledFact]],
    unresolved: Sequence[object],
    warnings: Sequence[object],
    human_review_required: bool,
    source_decisions: Sequence[FactDecision],
) -> None:
    """Validate graph invariants plus result partitions and aggregate fields."""

    fact_snapshot = _snapshot_validation_sequence(facts, "facts")
    relation_snapshot = _snapshot_validation_sequence(relations, "relations")
    unresolved_snapshot = _snapshot_validation_sequence(
        unresolved,
        "unresolved",
    )
    warning_snapshot = _snapshot_validation_sequence(warnings, "warnings")
    source_decision_snapshot = _snapshot_validation_sequence(
        source_decisions,
        "source_decisions",
    )
    _validate_graph_fields(fact_snapshot, relation_snapshot)
    if not source_decision_snapshot:
        _validate_status_partitions(fact_snapshot, partitions)
        _validate_warning_records(warning_snapshot)
        _validate_unresolved_records(unresolved_snapshot)
        if (
            fact_snapshot
            or relation_snapshot
            or unresolved_snapshot
            or warning_snapshot
        ):
            _fail_invariant(
                "empty source_decisions require an entirely empty result"
            )
        if type(human_review_required) is not bool or human_review_required:
            _fail_invariant(
                "an empty result cannot require human review"
            )
        return
    canonical_decisions = _validate_canonical_replay(
        fact_snapshot,
        relation_snapshot,
        source_decision_snapshot,
    )
    _validate_status_partitions(fact_snapshot, partitions)
    _validate_warning_records(warning_snapshot)
    _validate_unresolved_records(unresolved_snapshot)
    _validate_unresolved_provenance(
        unresolved_snapshot,
        canonical_decisions,
    )
    expected_review = (
        any(fact.requires_human_review for fact in fact_snapshot)
        or bool(unresolved_snapshot)
        or any(
            warning.requires_human_review for warning in warning_snapshot
        )
    )
    if type(human_review_required) is not bool or (
        human_review_required != expected_review
    ):
        _fail_invariant(
            "human_review_required must equal the exact review aggregation formula"
        )


__all__ = [
    "aggregate_human_review",
    "assemble_result",
    "materialize_relation_graph",
    "reconcile",
    "validate_relation_graph",
    "validate_result_fields",
]
