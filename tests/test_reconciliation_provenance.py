"""Task 21 provenance binding and review-critical warning semantics."""

from dataclasses import replace

import pytest

from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationStatus,
    SourceType,
    UnresolvedCandidate,
    WarningCode,
)
from agent_memory_os.reconcile.reconciler import (
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
    known,
    make_candidate,
    make_current_candidate,
    make_snapshot,
    unknown,
    unavailable,
)


def _nonknown_case():
    snapshot = make_snapshot()
    groups = group_candidates(
        snapshot.project_id.value,
        (
            make_candidate(
                candidate_id="unknown",
                value=unknown(
                    "value was not recorded",
                    source="collector:unknown:value",
                ),
                source_ref="collector:unknown",
            ),
            make_candidate(
                candidate_id="unavailable",
                value=unavailable(
                    "value source failed",
                    source="collector:unavailable:value",
                ),
                source_type=SourceType.PROJECT_DOC,
                source_ref="collector:unavailable",
            ),
        ),
        snapshot_id=make_snapshot_id(snapshot),
    )
    decisions, unresolved, warnings = resolve_non_known_groups(
        groups,
        NOW,
        default_policy(),
    )
    return snapshot, decisions, unresolved, warnings


def _nonknown_group(
    *,
    project_id: str,
    snapshot_id: str,
    candidate_id: str,
):
    return group_candidates(
        project_id,
        (
            make_candidate(
                candidate_id=candidate_id,
                value=unknown(
                    "value was not recorded",
                    source=f"{candidate_id}:value",
                ),
            ),
        ),
        snapshot_id=snapshot_id,
    )[0]


def test_nonknown_resolution_rejects_mixed_project_cohorts() -> None:
    groups = (
        _nonknown_group(
            project_id="project-one",
            snapshot_id="snapshot:shared",
            candidate_id="one",
        ),
        _nonknown_group(
            project_id="project-two",
            snapshot_id="snapshot:shared",
            candidate_id="two",
        ),
    )

    with pytest.raises(ReconciliationInputError, match="cohort"):
        resolve_non_known_groups(groups, NOW, default_policy())


def test_nonknown_resolution_rejects_mixed_snapshot_cohorts() -> None:
    groups = (
        _nonknown_group(
            project_id="synthetic-project",
            snapshot_id="snapshot:one",
            candidate_id="one",
        ),
        _nonknown_group(
            project_id="synthetic-project",
            snapshot_id="snapshot:two",
            candidate_id="two",
        ),
    )

    with pytest.raises(ReconciliationInputError, match="cohort"):
        resolve_non_known_groups(groups, NOW, default_policy())


def test_nonknown_resolution_rejects_mixed_run_cohorts() -> None:
    groups = (
        _nonknown_group(
            project_id="synthetic-project",
            snapshot_id="snapshot:shared",
            candidate_id="one",
        ),
        _nonknown_group(
            project_id="synthetic-project",
            snapshot_id="snapshot:shared",
            candidate_id="two",
        ),
    )

    with pytest.raises(ReconciliationInputError, match="cohort"):
        resolve_non_known_groups(groups, NOW, default_policy())


def test_unresolved_records_match_authenticated_candidates_exactly() -> None:
    snapshot, decisions, unresolved, warnings = _nonknown_case()

    result = assemble_result(
        snapshot,
        decisions,
        (),
        unresolved,
        warnings,
        NOW,
    )

    assert tuple(record.candidate_id for record in result.unresolved) == (
        "unavailable",
        "unknown",
    )
    assert all(fact.status is ReconciliationStatus.PENDING for fact in result.pending)
    assert {
        record.related_fact_id for record in result.unresolved
    } == {fact.fact_id for fact in result.pending}


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "ghost",
        "subject",
        "predicate",
        "status",
        "reason",
        "source-type",
        "source-ref",
        "field-source",
        "related-fact",
    ),
)
def test_rejects_unresolved_records_not_derived_from_source_decisions(
    mutation: str,
) -> None:
    snapshot, decisions, unresolved, warnings = _nonknown_case()
    first, second = unresolved
    if mutation == "missing":
        supplied = (first,)
    elif mutation == "duplicate":
        supplied = (first, first, second)
    elif mutation == "ghost":
        supplied = (
            first,
            second,
            replace(second, candidate_id="ghost"),
        )
    else:
        changes = {
            "subject": {"subject": "other-subject"},
            "predicate": {"predicate": "other-predicate"},
            "status": {"evidence_status": EvidenceStatus.UNKNOWN},
            "reason": {"reason": "different reason"},
            "source-type": {"source_type": SourceType.USER_EXPLICIT},
            "source-ref": {"source_ref": "other:source"},
            "field-source": {"field_source": "other:field"},
            "related-fact": {"related_fact_id": second.related_fact_id},
        }[mutation]
        supplied = (replace(first, **changes), second)

    with pytest.raises(
        ReconciliationInvariantError,
        match="unresolved.*authenticated source candidates",
    ):
        assemble_result(
            snapshot,
            decisions,
            (),
            supplied,
            warnings,
            NOW,
        )


def test_public_result_factory_rejects_ghost_unresolved_record() -> None:
    snapshot, decisions, unresolved, warnings = _nonknown_case()
    graph = materialize_relation_graph(decisions, ())
    ghost = replace(unresolved[0], candidate_id="ghost")

    from agent_memory_os.reconcile.models import ReconciliationResult

    with pytest.raises(
        ReconciliationInvariantError,
        match="unresolved.*authenticated source candidates",
    ):
        ReconciliationResult.create(
            project_id=snapshot.project_id.value,
            snapshot_id=make_snapshot_id(snapshot),
            reconciled_at=NOW.isoformat(),
            graph=graph,
            source_decisions=decisions,
            unresolved=(ghost, unresolved[1]),
            warnings=warnings,
        )


def test_missing_source_time_does_not_block_current_over_historical() -> None:
    historical = make_candidate(
        candidate_id="historical",
        value=8010,
        status_hint=known(CandidateStatusHint.HISTORICAL),
        source_type=SourceType.HISTORICAL_MEMORY,
        observed_at=unknown(
            "historical observation time was not recorded",
            source="historical:observed-at",
        ),
    )
    current = make_current_candidate(
        candidate_id="current",
        value=8000,
        observed_at=known(NOW.isoformat(), source="current:observed-at"),
    )

    outcome = resolve_predicate(
        tuple(
            classify_group(group, NOW, default_policy())
            for group in group_candidates(
                "synthetic-project",
                (historical, current),
            )
        ),
        default_policy(),
    )

    current_decision = next(
        decision
        for decision in outcome.decisions
        if "current" in decision.candidate_ids
    )
    historical_decision = next(
        decision
        for decision in outcome.decisions
        if "historical" in decision.candidate_ids
    )
    temporal = tuple(
        warning
        for decision in outcome.decisions
        for warning in decision.warnings
        if warning.code is WarningCode.UNRESOLVED_TEMPORAL_COMPARISON
    )
    assert current_decision.status is ReconciliationStatus.ACTIVE
    assert historical_decision.status is ReconciliationStatus.SUPERSEDED
    assert temporal == ()
    assert all(
        not decision.requires_human_review for decision in outcome.decisions
    )


@pytest.mark.parametrize("status", [EvidenceStatus.UNKNOWN, EvidenceStatus.UNAVAILABLE])
def test_nonknown_optional_metadata_is_active_and_not_review_critical(
    status: EvidenceStatus,
) -> None:
    metadata = (
        unknown("metadata was not recorded", source="candidate:metadata")
        if status is EvidenceStatus.UNKNOWN
        else unavailable("metadata source failed", source="candidate:metadata")
    )
    candidate = make_current_candidate(
        candidate_id="candidate",
        value="current-value",
        metadata=metadata,
    )
    snapshot = make_snapshot()
    group = group_candidates(
        snapshot.project_id.value,
        (candidate,),
        snapshot_id=make_snapshot_id(snapshot),
    )[0]
    decision = classify_group(
        group,
        NOW,
        default_policy(),
    )
    result = assemble_result(
        snapshot,
        (decision,),
        (),
        (),
        (),
        NOW,
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.activation_witness_candidate_ids == ("candidate",)
    assert decision.requires_human_review is False
    assert len(decision.warnings) == 1
    assert decision.warnings[0].code is (
        WarningCode.EVIDENCE_UNKNOWN
        if status is EvidenceStatus.UNKNOWN
        else WarningCode.EVIDENCE_UNAVAILABLE
    )
    assert decision.warnings[0].candidate_ids == ("candidate",)
    assert decision.warnings[0].evidence_refs == ("candidate:metadata",)
    assert decision.warnings[0].requires_human_review is False
    assert len(result.active) == 1
    assert result.pending == ()
    assert result.human_review_required is False
