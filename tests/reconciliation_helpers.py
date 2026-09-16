"""Deterministic synthetic fixtures shared by Phase 2 tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TypeVar

from agent_memory_os.evidence.models import (
    ChangedFileRecord,
    CollectorEvidence,
    CommitRecord,
    DiscoveredCommand,
    DocsEvidence,
    DocumentRecord,
    EvidenceSnapshot,
    EvidenceStatus,
    EvidenceValue,
    GitEvidence,
    RecentChangesEvidence,
    RecordedTestResult,
    RepositoryEvidence,
    ShadowGuardReport,
    TestsEvidence,
)
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    FactDecision,
    MemoryCandidate,
    ReconciliationPolicy,
    ReconciliationInputError,
    ReconciliationResult,
    ReconciliationWarning,
    ReconciledFact,
    ReconciliationStatus,
    RelationGraph,
    RelationRecord,
    RelationType,
    ResolutionMethod,
    SourceType,
    TemporalAssessment,
    UnresolvedCandidate,
    _freeze_reconciled_json_value,
)
from agent_memory_os.reconcile.reconciler import (
    _merge_requests,
    _snapshot_request,
    _validate_conflict_symmetry,
    materialize_relation_graph,
)
from agent_memory_os.reconcile.rules import (
    CandidateGroup,
    RelationRequest,
    assess_temporal,
    classify_group,
    group_candidates,
    make_fact_id,
)
from agent_memory_os.reconcile.serialization import make_snapshot_id


T = TypeVar("T")

CAPTURED_AT = "2026-09-14T00:00:00+00:00"
NOW = datetime.fromisoformat(CAPTURED_AT)


def iso_after(moment: datetime, seconds: int) -> str:
    """Return a deterministic ISO-8601 timestamp offset from ``moment``."""

    return (moment + timedelta(seconds=seconds)).isoformat()


def fixed_clock() -> str:
    """Return the deterministic reconciliation clock used by tests."""

    return NOW.isoformat()


def default_policy(**changes: object) -> ReconciliationPolicy:
    """Build the default frozen policy with selected test overrides."""

    return replace(ReconciliationPolicy(), **changes)


def known(value: T, *, source: str = "synthetic:test") -> EvidenceValue[T]:
    return EvidenceValue.known(value, source=source)


def unknown(
    reason: str,
    *,
    source: str = "synthetic:test",
) -> EvidenceValue[T]:
    return EvidenceValue.unknown(reason=reason, source=source)


def unavailable(
    reason: str,
    *,
    source: str = "synthetic:test",
) -> EvidenceValue[T]:
    return EvidenceValue.unavailable(reason=reason, source=source)


def _evidence(value: object, source: str) -> EvidenceValue:
    return value if isinstance(value, EvidenceValue) else known(value, source=source)


def make_candidate(
    *,
    candidate_id: str = "candidate",
    subject: str = "project",
    predicate: str = "setting",
    value: object = "value",
    status_hint: object = CandidateStatusHint.CURRENT_FACT,
    source_type: SourceType = SourceType.CURRENT_EVIDENCE,
    source_ref: str = "synthetic:candidate",
    observed_at: object = CAPTURED_AT,
    valid_from: object = CAPTURED_AT,
    valid_until: object = None,
    confidence: object = 1.0,
    explicit_user_instruction: object = False,
    supersedes: object = (),
    deprecated: object = False,
    metadata: object = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        subject=subject,
        predicate=predicate,
        value=_evidence(value, f"synthetic:{candidate_id}:value"),
        status_hint=_evidence(status_hint, f"synthetic:{candidate_id}:hint"),
        source_type=source_type,
        source_ref=source_ref,
        observed_at=_evidence(observed_at, f"synthetic:{candidate_id}:observed"),
        valid_from=_evidence(valid_from, f"synthetic:{candidate_id}:from"),
        valid_until=(
            unknown("no expiry", source=f"synthetic:{candidate_id}:until")
            if valid_until is None
            else _evidence(valid_until, f"synthetic:{candidate_id}:until")
        ),
        confidence=_evidence(
            confidence,
            f"synthetic:{candidate_id}:confidence",
        ),
        explicit_user_instruction=_evidence(
            explicit_user_instruction,
            f"synthetic:{candidate_id}:explicit",
        ),
        supersedes=_evidence(
            supersedes,
            f"synthetic:{candidate_id}:supersedes",
        ),
        deprecated=_evidence(
            deprecated,
            f"synthetic:{candidate_id}:deprecated",
        ),
        metadata=_evidence(
            {} if metadata is None else metadata,
            f"synthetic:{candidate_id}:metadata",
        ),
    )


def make_current_candidate(**fields: object) -> MemoryCandidate:
    values = {
        "source_type": SourceType.CURRENT_EVIDENCE,
        "status_hint": CandidateStatusHint.CURRENT_FACT,
    }
    values.update(fields)
    return make_candidate(**values)


def make_historical_candidate(**fields: object) -> MemoryCandidate:
    values = {
        "source_type": SourceType.HISTORICAL_MEMORY,
        "status_hint": CandidateStatusHint.HISTORICAL,
    }
    values.update(fields)
    return make_candidate(**values)


def make_source_of_truth_candidate(**fields: object) -> MemoryCandidate:
    values = {
        "source_type": SourceType.USER_EXPLICIT,
        "status_hint": CandidateStatusHint.SOURCE_OF_TRUTH,
        "explicit_user_instruction": True,
    }
    values.update(fields)
    return make_candidate(**values)


def synthetic_five_status_candidate_set() -> list[MemoryCandidate]:
    """Return independent predicates covering every public result status."""

    return [
        make_historical_candidate(
            candidate_id="path:old",
            predicate="project-path",
            value=r"C:\Users\demo\projects\interview-agent-v1",
        ),
        make_source_of_truth_candidate(
            candidate_id="path:current",
            predicate="project-path",
            value=r"C:\Users\demo\projects\interview-agent-finals",
        ),
        make_historical_candidate(
            candidate_id="port:historical",
            predicate="runtime-port",
            value=8010,
        ),
        make_current_candidate(
            candidate_id="port:current",
            predicate="runtime-port",
            value=8000,
        ),
        make_candidate(
            candidate_id="docs:left",
            predicate="project-doc-port",
            value=8020,
            source_type=SourceType.PROJECT_DOC,
        ),
        make_candidate(
            candidate_id="docs:right",
            predicate="project-doc-port",
            value=8030,
            source_type=SourceType.PROJECT_DOC,
        ),
        make_candidate(
            candidate_id="plan:future",
            predicate="planned-feature",
            value="dashboard",
            status_hint=CandidateStatusHint.PLAN,
            valid_from=iso_after(NOW, 60),
        ),
        make_candidate(
            candidate_id="retired:legacy",
            predicate="legacy-mode",
            value=True,
            status_hint=CandidateStatusHint.DEPRECATED,
            deprecated=True,
        ),
    ]


def single_group(*candidates: MemoryCandidate) -> CandidateGroup:
    """Group same-value candidates through the production grouping rule."""

    groups = group_candidates("synthetic-project", candidates)
    assert len(groups) == 1
    return groups[0]


def fact_id_for(candidate: MemoryCandidate) -> str:
    """Return the canonical synthetic-project fact ID for a candidate."""

    return make_fact_id(
        "synthetic-project",
        candidate.subject,
        candidate.predicate,
        candidate.value.value,
    )


def decisions_for(
    *candidates: MemoryCandidate,
    policy: ReconciliationPolicy | None = None,
) -> tuple[FactDecision, ...]:
    """Classify exact-value groups under one deterministic policy."""

    selected_policy = default_policy() if policy is None else policy
    groups = group_candidates("synthetic-project", candidates)
    return tuple(
        classify_group(group, NOW, selected_policy) for group in groups
    )


def provisional_fact(
    fact_id: str,
    status: ReconciliationStatus = ReconciliationStatus.PENDING,
    *,
    candidate_ids: tuple[str, ...] | None = None,
    selected_value: object | None = None,
    evidence_refs: tuple[str, ...] | None = None,
    superseded_candidate_ids: tuple[str, ...] = (),
    requires_human_review: bool = False,
    activation_witness_candidate_ids: tuple[str, ...] | None = None,
) -> ReconciledFact:
    """Build one graph-ready fact with deliberately empty derived indexes."""

    selected_candidate_ids = (
        (f"candidate:{fact_id}",)
        if candidate_ids is None
        else candidate_ids
    )
    witnesses = (
        (
            selected_candidate_ids
            if status is ReconciliationStatus.ACTIVE
            else ()
        )
        if activation_witness_candidate_ids is None
        else activation_witness_candidate_ids
    )
    method = (
        ResolutionMethod.DIRECT_CURRENT
        if status is ReconciliationStatus.ACTIVE
        else ResolutionMethod.PENDING_SEMANTICS
    )
    selected = (
        known(fact_id)
        if selected_value is None
        else _evidence(selected_value, f"synthetic:{fact_id}:value")
    )
    if selected.status is EvidenceStatus.KNOWN:
        selected = EvidenceValue.known(
            _freeze_reconciled_json_value(selected.value),
            source=selected.source,
        )
    fact = object.__new__(ReconciledFact)
    values = {
        "fact_id": fact_id,
        "subject": "project",
        "predicate": "setting",
        "selected_value": selected,
        "status": status,
        "confidence": known(1.0),
        "reason": "synthetic provisional fact",
        "evidence_refs": (
            (f"synthetic:{fact_id}",)
            if evidence_refs is None
            else evidence_refs
        ),
        "candidate_ids": selected_candidate_ids,
        "superseded_candidate_ids": superseded_candidate_ids,
        "conflict_candidate_ids": (),
        "valid_from": known("2026-09-14T00:00:00+00:00"),
        "resolved_at": known(
            "2026-09-14T00:00:00+00:00",
            source="reconciliation:clock",
        ),
        "resolution_method": method,
        "requires_human_review": requires_human_review,
        "activation_witness_candidate_ids": witnesses,
        "relation_ids": (),
        "supersedes_fact_ids": (),
        "superseded_by_fact_ids": (),
        "conflict_fact_ids": (),
    }
    for field_name, value in values.items():
        object.__setattr__(fact, field_name, value)
    return fact


def provisional_facts(*fact_ids: str) -> tuple[ReconciledFact, ...]:
    """Build graph-ready synthetic facts in caller-specified order."""

    return tuple(provisional_fact(fact_id) for fact_id in fact_ids)


def _test_only_copy(instance: object, **changes: object) -> object:
    """Copy a token-gated frozen record for deliberate corruption tests."""

    copied = object.__new__(type(instance))
    for field_name in instance.__dataclass_fields__:
        if field_name in changes:
            value = changes[field_name]
        else:
            value = object.__getattribute__(instance, field_name)
        object.__setattr__(copied, field_name, value)
    for hidden_field in ("_candidate_context", "_cohort", "_stage_auth"):
        try:
            value = object.__getattribute__(instance, hidden_field)
        except AttributeError:
            continue
        object.__setattr__(copied, hidden_field, value)
    return copied


def _test_only_graph(
    facts: tuple[ReconciledFact, ...],
    relations: tuple[object, ...],
) -> RelationGraph:
    """Assemble a deliberately unchecked graph without exposing production APIs."""

    facts_by_id: dict[str, object] = {}
    for fact in facts:
        try:
            fact_id = object.__getattribute__(fact, "fact_id")
        except AttributeError:
            continue
        if type(fact_id) is str:
            facts_by_id[fact_id] = fact
    graph = object.__new__(RelationGraph)
    object.__setattr__(graph, "facts", facts)
    object.__setattr__(graph, "relations", relations)
    object.__setattr__(
        graph,
        "_facts_by_id",
        MappingProxyType(facts_by_id),
    )
    return graph


def _test_only_materialize_graph(
    facts: tuple[ReconciledFact, ...],
    requests: tuple[RelationRequest, ...],
) -> RelationGraph:
    """Populate derived fields for a raw corruption-first graph fixture."""

    requests = tuple(_snapshot_request(request) for request in requests)
    if len({fact.fact_id for fact in facts}) != len(facts):
        raise ReconciliationInputError("facts must have unique fact_id values")
    if any(
        any(
            (
                fact.conflict_candidate_ids,
                fact.relation_ids,
                fact.supersedes_fact_ids,
                fact.superseded_by_fact_ids,
                fact.conflict_fact_ids,
            )
        )
        for fact in facts
    ):
        raise ReconciliationInputError("provisional fact derived indexes must be empty")
    fact_ids = {fact.fact_id for fact in facts}
    for request in requests:
        if request.from_fact_id not in fact_ids or request.to_fact_id not in fact_ids:
            raise ReconciliationInputError(
                "every relation request endpoint must reference an input fact"
            )
    merged_requests = _merge_requests(requests)
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
    relation_ids = {fact.fact_id: set() for fact in facts}
    supersedes = {fact.fact_id: set() for fact in facts}
    superseded_by = {fact.fact_id: set() for fact in facts}
    conflicts = {fact.fact_id: set() for fact in facts}
    conflict_candidates = {fact.fact_id: set() for fact in facts}
    for relation in relations:
        relation_ids[relation.from_fact_id].add(relation.relation_id)
        relation_ids[relation.to_fact_id].add(relation.relation_id)
        if relation.relation_type is RelationType.SUPERSEDES:
            supersedes[relation.from_fact_id].add(relation.to_fact_id)
            superseded_by[relation.to_fact_id].add(relation.from_fact_id)
        else:
            conflicts[relation.from_fact_id].add(relation.to_fact_id)
            conflicts[relation.to_fact_id].add(relation.from_fact_id)
            conflict_candidates[relation.from_fact_id].update(
                set(relation.candidate_ids)
                - set(next(
                    fact.candidate_ids
                    for fact in facts
                    if fact.fact_id == relation.from_fact_id
                ))
            )
            conflict_candidates[relation.to_fact_id].update(
                set(relation.candidate_ids)
                - set(next(
                    fact.candidate_ids
                    for fact in facts
                    if fact.fact_id == relation.to_fact_id
                ))
            )
    materialized = tuple(
        _test_only_copy(
            fact,
            relation_ids=tuple(sorted(relation_ids[fact.fact_id])),
            supersedes_fact_ids=tuple(sorted(supersedes[fact.fact_id])),
            superseded_by_fact_ids=tuple(sorted(superseded_by[fact.fact_id])),
            conflict_fact_ids=tuple(sorted(conflicts[fact.fact_id])),
            conflict_candidate_ids=tuple(
                sorted(conflict_candidates[fact.fact_id])
            ),
        )
        for fact in sorted(facts, key=lambda item: item.fact_id)
    )
    return _test_only_graph(
        materialized,
        tuple(
            sorted(
                relations,
                key=lambda item: (
                    item.relation_type.value,
                    item.from_fact_id,
                    item.to_fact_id,
                ),
            )
        ),
    )


def graph_with_supersedes_edges(
    *edges: tuple[str, str],
) -> RelationGraph:
    """Build an unchecked graph whose statuses match incoming edge topology."""

    nodes = sorted({node for edge in edges for node in edge})
    incoming = {target for _, target in edges}
    facts = tuple(
        provisional_fact(
            node,
            (
                ReconciliationStatus.SUPERSEDED
                if node in incoming
                else ReconciliationStatus.ACTIVE
            ),
        )
        for node in nodes
    )
    return _test_only_materialize_graph(
        facts,
        tuple(supersedes_request(source, target) for source, target in edges),
    )


def corrupt_graph_missing_superseded_by_index() -> RelationGraph:
    """Remove exactly one inverse supersession index from a valid graph."""

    graph = graph_with_supersedes_edges(("new", "old"))
    facts = tuple(
        _test_only_copy(fact, superseded_by_fact_ids=())
        if fact.fact_id == "old"
        else fact
        for fact in graph.facts
    )
    return _test_only_graph(facts, graph.relations)


def supersedes_request(
    source: str,
    target: str,
    *,
    reason: str = "explicit candidate supersedes relation",
    candidate_ids: tuple[str, ...] | None = None,
    evidence_refs: tuple[str, ...] | None = None,
    method: ResolutionMethod = ResolutionMethod.EXPLICIT_SUPERSEDES,
) -> RelationRequest:
    """Build one directed synthetic supersession request."""

    return RelationRequest.supersedes(
        source,
        target,
        reason,
        (
            (f"candidate:{source}", f"candidate:{target}")
            if candidate_ids is None
            else candidate_ids
        ),
        (
            (f"evidence:{source}", f"evidence:{target}")
            if evidence_refs is None
            else evidence_refs
        ),
        method=method,
    )


def symmetric_conflict_requests(
    left: str,
    right: str,
    *,
    reason: str = "unresolved competing current facts",
) -> tuple[RelationRequest, RelationRequest]:
    """Build the two canonical directed requests for one conflict pair."""

    candidates = tuple(sorted((f"candidate:{left}", f"candidate:{right}")))
    evidence = tuple(sorted((f"synthetic:{left}", f"synthetic:{right}")))
    return (
        RelationRequest.conflicts(
            left,
            right,
            reason,
            candidates,
            evidence,
        ),
        RelationRequest.conflicts(
            right,
            left,
            reason,
            candidates,
            evidence,
        ),
    )


def temporal_map(
    group: CandidateGroup,
    now: datetime,
    policy: ReconciliationPolicy | None = None,
) -> dict[str, TemporalAssessment]:
    """Assess every candidate through the production temporal rule."""

    selected_policy = default_policy() if policy is None else policy
    return {
        item.candidate_id: assess_temporal(item, now, selected_policy)
        for item in group.candidates
    }


def _document(name: str) -> DocumentRecord:
    source = f"synthetic:docs:{name}"
    return DocumentRecord(
        path=known(f"docs/{name}.md", source=source),
        title=known(name.replace("_", " ").title(), source=source),
        summary=known(f"Synthetic {name} document", source=source),
        last_updated=known("2026-09-13T00:00:00+00:00", source=source),
    )


def make_snapshot() -> EvidenceSnapshot:
    """Build a complete Phase 1 snapshot without touching a real project."""

    commit = CommitRecord(
        sha=known("a" * 40, source="synthetic:git:log"),
        short_sha=known("a" * 12, source="synthetic:git:log"),
        authored_at=known(
            "2026-09-13T00:00:00+00:00",
            source="synthetic:git:log",
        ),
        subject=known("synthetic commit", source="synthetic:git:log"),
    )
    changed_file = ChangedFileRecord(
        path=known("README.md", source="synthetic:git:diff"),
        status=known("M", source="synthetic:git:diff"),
    )
    command = DiscoveredCommand(
        name=known("test", source="synthetic:tests:discovery"),
        command=known("pytest", source="synthetic:tests:discovery"),
        kind=known("python", source="synthetic:tests:discovery"),
    )
    test_result = RecordedTestResult(
        summary=known("116 passed", source="synthetic:tests:recorded"),
        execution_status=known(
            "recorded_not_executed",
            source="synthetic:tests:recorded",
        ),
        source_document=known(
            "docs/STATE.md",
            source="synthetic:tests:recorded",
        ),
    )
    guard = ShadowGuardReport(
        head_before=known("a" * 40, source="synthetic:guard:before"),
        head_after=known("a" * 40, source="synthetic:guard:after"),
        index_hash_before=known("b" * 64, source="synthetic:guard:before"),
        index_hash_after=known("b" * 64, source="synthetic:guard:after"),
        status_hash_before=known("c" * 64, source="synthetic:guard:before"),
        status_hash_after=known("c" * 64, source="synthetic:guard:after"),
        tracked_manifest_before=known(
            "d" * 64,
            source="synthetic:guard:before",
        ),
        tracked_manifest_after=known(
            "d" * 64,
            source="synthetic:guard:after",
        ),
        head_unchanged=known(True, source="synthetic:guard:comparison"),
        index_unchanged=known(True, source="synthetic:guard:comparison"),
        status_unchanged=known(True, source="synthetic:guard:comparison"),
        tracked_content_unchanged=known(
            True,
            source="synthetic:guard:comparison",
        ),
        changed_components=known([], source="synthetic:guard:comparison"),
        verdict=known(
            "SHADOW_COLLECTION_PASS",
            source="synthetic:guard:comparison",
        ),
    )
    return EvidenceSnapshot(
        schema_version=known("1.0.0", source="synthetic:collector:schema"),
        project_id=known(
            "synthetic-project",
            source="synthetic:collector:project-id",
        ),
        captured_at=known(CAPTURED_AT, source="synthetic:collector:clock"),
        source_mode=known(
            "shadow_read_only",
            source="synthetic:collector:mode",
        ),
        repository=RepositoryEvidence(
            path=known(
                r"D:\synthetic\project",
                source="synthetic:repository:path",
            ),
            exists=known(True, source="synthetic:repository:exists"),
            project_name=known(
                "project",
                source="synthetic:repository:name",
            ),
        ),
        git=GitEvidence(
            is_repository=known(True, source="synthetic:git:repository"),
            branch=known("main", source="synthetic:git:branch"),
            head_sha=known("a" * 40, source="synthetic:git:head"),
            head_short=known("a" * 12, source="synthetic:git:head-short"),
            tags_at_head=known([], source="synthetic:git:tags"),
            remote_count=known(0, source="synthetic:git:remotes"),
            staged_count=known(0, source="synthetic:git:staged"),
            modified_count=known(1, source="synthetic:git:modified"),
            untracked_count=known(0, source="synthetic:git:untracked"),
            recent_commits=known([commit], source="synthetic:git:log"),
        ),
        docs=DocsEvidence(
            discovered=known(
                [
                    "AGENTS.md",
                    "README.md",
                    "docs/DECISIONS.md",
                    "docs/NEXT.md",
                    "docs/PROJECT_CONTEXT.md",
                    "docs/SESSION_LOG.md",
                    "docs/STATE.md",
                ],
                source="synthetic:docs:discovery",
            ),
            agents=known(_document("AGENTS"), source="synthetic:docs:AGENTS"),
            project_context=known(
                _document("PROJECT_CONTEXT"),
                source="synthetic:docs:PROJECT_CONTEXT",
            ),
            state=known(_document("STATE"), source="synthetic:docs:STATE"),
            next=known(_document("NEXT"), source="synthetic:docs:NEXT"),
            decisions=known(
                _document("DECISIONS"),
                source="synthetic:docs:DECISIONS",
            ),
            session_log=known(
                _document("SESSION_LOG"),
                source="synthetic:docs:SESSION_LOG",
            ),
            readme=known(_document("README"), source="synthetic:docs:README"),
        ),
        tests=TestsEvidence(
            discovered_test_roots=known(
                ["tests"],
                source="synthetic:tests:discovery",
            ),
            python_test_files=known(
                ["tests/test_example.py"],
                source="synthetic:tests:discovery",
            ),
            frontend_test_files=known(
                [],
                source="synthetic:tests:discovery",
            ),
            discovered_commands=known(
                [command],
                source="synthetic:tests:commands",
            ),
            last_recorded_results=known(
                [test_result],
                source="synthetic:tests:recorded",
            ),
        ),
        recent_changes=RecentChangesEvidence(
            changed_files=known(
                [changed_file],
                source="synthetic:changes:files",
            ),
            diff_stat=known(
                "1 file changed",
                source="synthetic:changes:diff-stat",
            ),
            recent_commit_summary=known(
                ["aaaaaaaaaaaa synthetic commit"],
                source="synthetic:changes:commits",
            ),
        ),
        collector=CollectorEvidence(
            warnings=known([], source="synthetic:collector:warnings"),
            errors=known([], source="synthetic:collector:errors"),
            evidence_sources=known(
                ["synthetic:docs", "synthetic:git", "synthetic:tests"],
                source="synthetic:collector:sources",
            ),
            shadow_guard=known(guard, source="synthetic:guard:report"),
        ),
    )


def active_fact(label: str) -> ReconciledFact:
    """Build one authenticated ACTIVE fact for serialization-focused tests."""

    snapshot_id = make_snapshot_id(make_snapshot())
    candidate = make_current_candidate(
        candidate_id=f"candidate:{label}",
        predicate=label,
        value=label,
    )
    groups = group_candidates(
        "synthetic-project",
        (candidate,),
        snapshot_id=snapshot_id,
    )
    decision = classify_group(groups[0], NOW, default_policy())
    return materialize_relation_graph((decision,), ()).facts[0]


def make_result(
    *,
    graph: RelationGraph,
    source_decisions: tuple[FactDecision, ...] | list[FactDecision],
    project_id: str = "synthetic-project",
    snapshot_id: str,
    reconciled_at: str = CAPTURED_AT,
    unresolved: tuple[UnresolvedCandidate, ...]
    | list[UnresolvedCandidate] = (),
    warnings: tuple[ReconciliationWarning, ...]
    | list[ReconciliationWarning] = (),
) -> ReconciliationResult:
    """Invoke the sole safe result constructor with caller-owned containers."""

    return ReconciliationResult.create(
        project_id=project_id,
        snapshot_id=snapshot_id,
        reconciled_at=reconciled_at,
        graph=graph,
        source_decisions=source_decisions,
        unresolved=unresolved,
        warnings=warnings,
    )


def complete_result(
    *,
    clock_value: str = CAPTURED_AT,
    reverse_inputs: bool = False,
) -> ReconciliationResult:
    """Build a two-predicate result whose input order cannot affect bytes."""

    try:
        selected_clock = datetime.fromisoformat(clock_value)
    except (TypeError, ValueError) as error:
        raise ReconciliationInputError(
            "synthetic complete_result clock must be ISO-8601"
        ) from error
    snapshot_id = make_snapshot_id(make_snapshot())
    candidates = [
        make_current_candidate(
            candidate_id="candidate:path",
            predicate="project-path",
            value=r"D:\模拟\项目",
        ),
        make_current_candidate(
            candidate_id="candidate:port",
            predicate="runtime-port",
            value=8000,
        ),
    ]
    if reverse_inputs:
        candidates.reverse()
    groups = group_candidates(
        "synthetic-project",
        candidates,
        snapshot_id=snapshot_id,
    )
    decisions = tuple(
        classify_group(group, selected_clock, default_policy())
        for group in groups
    )
    graph = materialize_relation_graph(decisions, ())
    return make_result(
        graph=graph,
        source_decisions=list(decisions),
        snapshot_id=snapshot_id,
        reconciled_at=clock_value,
    )
