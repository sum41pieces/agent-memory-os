"""Task 21 unresolved provenance and exact result review aggregation."""

from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationWarning,
    WarningCode,
)
from agent_memory_os.reconcile.reconciler import (
    aggregate_human_review,
    assemble_result,
    materialize_relation_graph,
)
from agent_memory_os.reconcile.rules import (
    classify_group,
    group_candidates,
    resolve_non_known_groups,
    resolve_predicate,
)
from agent_memory_os.reconcile.serialization import make_snapshot_id
from reconciliation_helpers import (
    NOW,
    default_policy,
    iso_after,
    known,
    make_candidate,
    make_current_candidate,
    make_snapshot,
    unknown,
    unavailable,
)


def _groups(*candidates):
    snapshot = make_snapshot()
    groups = group_candidates(
        snapshot.project_id.value,
        candidates,
        snapshot_id=make_snapshot_id(snapshot),
    )
    return snapshot, groups


def test_unknown_and_unavailable_are_pending_and_not_discarded() -> None:
    unknown_value = unknown(
        "not enough evidence",
        source="collector:unknown:value",
    )
    unavailable_value = unavailable(
        "source cannot provide",
        source="collector:unavailable:value",
    )
    snapshot, groups = _groups(
        make_candidate(
            candidate_id="unknown",
            value=unknown_value,
            source_ref="collector:unknown",
        ),
        make_candidate(
            candidate_id="unavailable",
            value=unavailable_value,
            source_ref="collector:unavailable",
        ),
    )

    decisions, unresolved, warnings = resolve_non_known_groups(
        groups,
        NOW,
        default_policy(),
    )
    result = assemble_result(
        snapshot,
        decisions,
        (),
        unresolved,
        warnings,
        NOW,
    )

    assert result.active == ()
    assert {fact.status for fact in result.pending} == {
        ReconciliationStatus.PENDING
    }
    assert {item.candidate_id for item in result.unresolved} == {
        "unknown",
        "unavailable",
    }
    by_id = {item.candidate_id: item for item in result.unresolved}
    assert by_id["unknown"].evidence_status is EvidenceStatus.UNKNOWN
    assert by_id["unknown"].reason == "not enough evidence"
    assert by_id["unknown"].field_source == "collector:unknown:value"
    assert by_id["unknown"].source_ref == "collector:unknown"
    assert by_id["unavailable"].evidence_status is EvidenceStatus.UNAVAILABLE
    assert by_id["unavailable"].reason == "source cannot provide"
    assert result.unresolved_count == 2
    assert result.summary_counts.pending == 2
    assert result.summary_counts.unresolved == 2
    assert result.summary_counts.warnings == 2
    assert result.human_review_required is True


def test_pending_plan_alone_does_not_set_result_review() -> None:
    snapshot, groups = _groups(
        make_candidate(
            candidate_id="plan",
            status_hint=known(CandidateStatusHint.PLAN),
            valid_from=known(iso_after(NOW, 1)),
            confidence=known(0.9),
        )
    )
    decision = classify_group(groups[0], NOW, default_policy())

    result = assemble_result(
        snapshot,
        (decision,),
        (),
        (),
        decision.warnings,
        NOW,
    )

    assert len(result.pending) == 1
    assert result.pending[0].requires_human_review is False
    assert result.human_review_required is False


def test_insufficient_confidence_alone_does_not_set_result_review() -> None:
    snapshot, groups = _groups(
        make_current_candidate(
            candidate_id="low-confidence",
            confidence=known(0.49),
        )
    )
    decision = classify_group(groups[0], NOW, default_policy())

    result = assemble_result(
        snapshot,
        (decision,),
        (),
        (),
        (),
        NOW,
    )

    assert result.pending[0].requires_human_review is False
    assert tuple(warning.code for warning in result.warnings) == (
        WarningCode.INSUFFICIENT_CONFIDENCE,
    )
    assert result.warnings[0].requires_human_review is False
    assert result.human_review_required is False


def test_conflict_sets_result_review() -> None:
    snapshot, groups = _groups(
        make_current_candidate(candidate_id="left", value=8000),
        make_current_candidate(candidate_id="right", value=8010),
    )
    classified = tuple(
        classify_group(group, NOW, default_policy()) for group in groups
    )
    outcome = resolve_predicate(classified, default_policy())

    result = assemble_result(
        snapshot,
        outcome.decisions,
        outcome.relation_requests,
        (),
        (),
        NOW,
    )

    assert len(result.conflicted) == 2
    assert all(fact.requires_human_review for fact in result.conflicted)
    assert result.human_review_required is True


def test_assemble_result_preserves_complete_warning_union_once() -> None:
    snapshot, groups = _groups(
        make_current_candidate(
            candidate_id="low-confidence",
            confidence=known(0.49),
        )
    )
    decision = classify_group(groups[0], NOW, default_policy())
    extra = ReconciliationWarning(
        code=WarningCode.EVIDENCE_UNKNOWN,
        message="optional metadata was not recorded",
        candidate_ids=("low-confidence",),
        evidence_refs=("collector:metadata",),
        requires_human_review=False,
    )

    result = assemble_result(
        snapshot,
        (decision,),
        (),
        (),
        (extra, *decision.warnings, extra),
        NOW,
    )

    assert tuple(warning.code for warning in result.warnings) == (
        WarningCode.EVIDENCE_UNKNOWN,
        WarningCode.INSUFFICIENT_CONFIDENCE,
    )
    assert len(result.warnings) == 2
    assert result.summary_counts.warnings == 2
    assert result.human_review_required is False


def test_result_factory_restores_omitted_source_decision_warnings() -> None:
    snapshot, groups = _groups(
        make_current_candidate(
            candidate_id="low-confidence",
            confidence=known(0.49),
        )
    )
    decision = classify_group(groups[0], NOW, default_policy())
    graph = materialize_relation_graph((decision,), ())

    result = ReconciliationResult.create(
        project_id=snapshot.project_id.value,
        snapshot_id=make_snapshot_id(snapshot),
        reconciled_at=NOW.isoformat(),
        graph=graph,
        source_decisions=(decision,),
        warnings=(),
    )

    assert result.warnings == decision.warnings
    assert result.summary_counts.warnings == 1


def test_result_factory_deduplicates_decision_warnings_and_keeps_extra() -> None:
    snapshot, groups = _groups(
        make_current_candidate(
            candidate_id="low-confidence",
            confidence=known(0.49),
        )
    )
    decision = classify_group(groups[0], NOW, default_policy())
    graph = materialize_relation_graph((decision,), ())
    extra = ReconciliationWarning(
        code=WarningCode.EVIDENCE_UNKNOWN,
        message="optional aggregate context was not recorded",
        candidate_ids=("low-confidence",),
        evidence_refs=("aggregate:context",),
        requires_human_review=False,
    )

    result = ReconciliationResult.create(
        project_id=snapshot.project_id.value,
        snapshot_id=make_snapshot_id(snapshot),
        reconciled_at=NOW.isoformat(),
        graph=graph,
        source_decisions=(decision,),
        warnings=(extra, *decision.warnings, extra, *decision.warnings),
    )

    assert tuple(warning.code for warning in result.warnings) == (
        WarningCode.EVIDENCE_UNKNOWN,
        WarningCode.INSUFFICIENT_CONFIDENCE,
    )
    assert len(result.warnings) == 2
    assert result.summary_counts.warnings == 2


def test_aggregate_human_review_is_exact_disjunction() -> None:
    non_review_warning = ReconciliationWarning(
        code=WarningCode.INSUFFICIENT_CONFIDENCE,
        message="confidence is below the active threshold",
        candidate_ids=("candidate",),
        evidence_refs=("candidate:confidence",),
        requires_human_review=False,
    )
    review_warning = ReconciliationWarning(
        code=WarningCode.INVALID_TEMPORAL_ORDER,
        message="candidate interval is invalid",
        candidate_ids=("candidate",),
        evidence_refs=("candidate:validity",),
        requires_human_review=True,
    )

    assert aggregate_human_review((), (), ()) is False
    assert aggregate_human_review((), (), (non_review_warning,)) is False
    assert aggregate_human_review((), (), (review_warning,)) is True
