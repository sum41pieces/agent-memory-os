from dataclasses import FrozenInstanceError
from datetime import timedelta, tzinfo
from pathlib import Path

import pytest

from agent_memory_os.reconcile import rules as reconciliation_rules
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    FactDecision,
    MemoryCandidate,
    ReconciliationInputError,
    ReconciliationPolicy,
    ReconciliationStatus,
    RelationType,
    ResolutionMethod,
    SourceType,
)
from agent_memory_os.reconcile.rules import (
    RelationRequest,
    classify_group,
    resolve_cross_value_replacements,
    resolve_predicate,
)

from reconciliation_helpers import (
    NOW,
    decisions_for,
    default_policy,
    fact_id_for,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    make_source_of_truth_candidate,
    single_group,
    unavailable,
    unknown,
)


def test_explicit_source_of_truth_replaces_old_synthetic_path_without_filesystem_access(
    monkeypatch,
) -> None:
    def explode_on_open(self: Path, *args: object, **kwargs: object) -> None:
        raise AssertionError(str(self))

    monkeypatch.setattr(Path, "open", explode_on_open)
    old = make_historical_candidate(
        candidate_id="old-path",
        predicate="project_path",
        value=r"C:\Users\demo\projects\interview-agent-v1",
    )
    new = make_source_of_truth_candidate(
        candidate_id="finals-path",
        predicate="project_path",
        value=r"C:\Users\demo\projects\interview-agent-finals",
        confidence=known(1.0),
    )

    outcome = resolve_predicate(decisions_for(old, new), default_policy())

    assert outcome.status_for(fact_id_for(new)) is ReconciliationStatus.ACTIVE
    assert (
        outcome.status_for(fact_id_for(old))
        is ReconciliationStatus.SUPERSEDED
    )
    assert outcome.relation_requests[0].reason == (
        "explicit current source-of-truth designation"
    )
    assert outcome.relation_requests[0].method is (
        ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"explicit_user_instruction": known(False)},
        {"source_type": SourceType.CURRENT_EVIDENCE},
        {"status_hint": known(CandidateStatusHint.PLAN)},
        {"confidence": known(0.1)},
        {
            "valid_from": known("2026-09-14T00:00:10+00:00"),
            "valid_until": known("2026-09-14T00:00:05+00:00"),
        },
    ),
    ids=(
        "missing-explicit-flag",
        "wrong-source",
        "plan-hint",
        "low-confidence",
        "invalid-temporal-order",
    ),
)
def test_invalid_source_of_truth_does_not_replace_competing_fact(
    changes: dict[str, object],
) -> None:
    old = make_historical_candidate(candidate_id="old", value="old")
    invalid = make_source_of_truth_candidate(
        candidate_id="invalid-source-of-truth",
        value="new",
        **changes,
    )

    outcome = resolve_predicate(
        decisions_for(old, invalid),
        default_policy(),
    )

    assert outcome.relation_requests == ()
    assert (
        outcome.status_for(fact_id_for(old))
        is not ReconciliationStatus.SUPERSEDED
    )


def test_disabled_source_of_truth_policy_does_not_replace_competing_fact(
) -> None:
    policy = default_policy(allow_explicit_source_of_truth_override=False)
    old = make_historical_candidate(candidate_id="old", value="old")
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="new",
    )

    outcome = resolve_predicate(
        decisions_for(old, designated, policy=policy),
        policy,
    )

    assert outcome.relation_requests == ()
    assert (
        outcome.status_for(fact_id_for(old))
        is not ReconciliationStatus.SUPERSEDED
    )


def test_incompatible_source_of_truth_groups_become_conflicted(
) -> None:
    left = make_source_of_truth_candidate(candidate_id="left", value="left")
    right = make_source_of_truth_candidate(candidate_id="right", value="right")

    outcome = resolve_predicate(
        decisions_for(right, left),
        default_policy(),
    )

    assert len(outcome.relation_requests) == 2
    assert all(
        request.relation_type is RelationType.CONFLICTS
        for request in outcome.relation_requests
    )
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert all(
        decision.requires_human_review for decision in outcome.decisions
    )


def test_same_value_corroboration_is_retained_in_source_of_truth_provenance(
) -> None:
    old = make_historical_candidate(
        candidate_id="old",
        value="old",
        source_ref="history:old",
    )
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="new",
        source_ref="user:designation",
    )
    corroborating = make_historical_candidate(
        candidate_id="corroborating",
        value="new",
        source_ref="history:corroboration",
    )

    outcome = resolve_predicate(
        decisions_for(old, designated, corroborating),
        default_policy(),
    )

    winner = next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(designated)
    )
    request = outcome.relation_requests[0]
    assert winner.candidate_ids == ("corroborating", "designated")
    assert request.candidate_ids == (
        "corroborating",
        "designated",
        "old",
    )
    assert {
        "history:corroboration",
        "history:old",
        "user:designation",
    }.issubset(request.evidence_refs)


def test_explicit_supersedes_beats_same_direction_source_of_truth_rule() -> None:
    old = make_historical_candidate(candidate_id="old", value="old")
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="new",
        supersedes=known(("old",)),
    )

    outcome = resolve_predicate(
        decisions_for(old, designated),
        default_policy(),
    )

    replacement = outcome.relation_requests[0]
    winner = next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(designated)
    )
    replaced = next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(old)
    )
    assert outcome.relation_requests == (replacement,)
    assert replacement.method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert replacement.reason == "explicit candidate supersedes relation"
    assert winner.status is ReconciliationStatus.ACTIVE
    assert winner.resolution_method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert winner.reason == "explicit candidate supersedes relation"
    assert replaced.status is ReconciliationStatus.SUPERSEDED
    assert replaced.resolution_method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert replaced.reason == "explicit candidate supersedes relation"


def test_explicit_competitor_supersedes_source_of_truth_without_reverse_edge(
) -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    explicit = make_current_candidate(
        candidate_id="explicit",
        value="explicit",
        supersedes=known(("designated",)),
    )

    outcome = resolve_predicate(
        decisions_for(designated, explicit),
        default_policy(),
    )

    replacement = outcome.relation_requests[0]
    assert len(outcome.relation_requests) == 1
    assert replacement.from_fact_id == fact_id_for(explicit)
    assert replacement.to_fact_id == fact_id_for(designated)
    assert replacement.method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert (
        outcome.status_for(fact_id_for(explicit))
        is ReconciliationStatus.ACTIVE
    )
    assert (
        outcome.status_for(fact_id_for(designated))
        is ReconciliationStatus.SUPERSEDED
    )
    assert all(
        decision.resolution_method is ResolutionMethod.EXPLICIT_SUPERSEDES
        for decision in outcome.decisions
    )


def test_mutual_explicit_declarations_with_source_of_truth_conflict(
) -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
        supersedes=known(("explicit",)),
    )
    explicit = make_current_candidate(
        candidate_id="explicit",
        value="explicit",
        supersedes=known(("designated",)),
    )

    outcome = resolve_predicate(
        decisions_for(designated, explicit),
        default_policy(),
    )

    assert len(outcome.relation_requests) == 2
    assert all(
        request.relation_type is RelationType.CONFLICTS
        and request.reason
        == "contradictory explicit supersedes declarations"
        for request in outcome.relation_requests
    )
    assert all(
        decision.status is ReconciliationStatus.CONFLICTED
        and decision.requires_human_review
        for decision in outcome.decisions
    )
    assert outcome.active_fact_ids == ()


def test_singleton_source_of_truth_preserves_base_active_classification() -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    base = decisions_for(designated)[0]

    outcome = resolve_predicate((base,), default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == (base,)
    assert outcome.decisions[0].resolution_method is ResolutionMethod.DIRECT_CURRENT


def test_source_of_truth_with_only_deprecated_and_nonknown_groups_is_not_relabelled(
) -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    deprecated = make_historical_candidate(
        candidate_id="deprecated",
        value="deprecated",
        deprecated=known(True),
    )
    unresolved = make_candidate(
        candidate_id="unresolved",
        value=unknown("value unavailable"),
        status_hint=CandidateStatusHint.HISTORICAL,
        source_type=SourceType.HISTORICAL_MEMORY,
    )

    outcome = resolve_predicate(
        decisions_for(designated, deprecated, unresolved),
        default_policy(),
    )

    assert outcome.relation_requests == ()
    assert outcome.status_for(fact_id_for(designated)) is ReconciliationStatus.ACTIVE
    assert next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(designated)
    ).resolution_method is ResolutionMethod.DIRECT_CURRENT
    assert all(
        decision.status is not ReconciliationStatus.SUPERSEDED
        for decision in outcome.decisions
    )


def test_source_of_truth_does_not_displace_plan_competitor() -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    plan = make_candidate(
        candidate_id="plan",
        value="planned",
        status_hint=CandidateStatusHint.PLAN,
        source_type=SourceType.USER_EXPLICIT,
        explicit_user_instruction=known(True),
    )

    outcome = resolve_predicate(
        decisions_for(designated, plan),
        default_policy(),
    )

    assert outcome.relation_requests == ()
    assert outcome.status_for(fact_id_for(plan)) is ReconciliationStatus.PENDING
    assert next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(designated)
    ).resolution_method is ResolutionMethod.DIRECT_CURRENT


@pytest.mark.parametrize(
    "changes",
    (
        {
            "valid_from": known("2026-09-14T00:00:10+00:00"),
            "valid_until": known("2026-09-14T00:00:05+00:00"),
        },
        {"valid_from": known("2026-09-14T00:00:01+00:00")},
        {
            "valid_from": known("2026-09-13T23:00:00+00:00"),
            "valid_until": known("2026-09-13T23:30:00+00:00"),
        },
        {"confidence": known(0.1)},
    ),
    ids=("invalid-order", "future", "expired", "insufficient-confidence"),
)
def test_source_of_truth_does_not_displace_ineligible_current_competitor(
    changes: dict[str, object],
) -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    ineligible = make_current_candidate(
        candidate_id="ineligible",
        value="ineligible",
        **changes,
    )

    outcome = resolve_predicate(
        decisions_for(designated, ineligible),
        default_policy(),
    )

    assert outcome.relation_requests == ()
    assert (
        outcome.status_for(fact_id_for(ineligible))
        is ReconciliationStatus.PENDING
    )
    assert next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == fact_id_for(designated)
    ).resolution_method is ResolutionMethod.DIRECT_CURRENT


def test_current_direct_evidence_supersedes_historical_memory() -> None:
    historical = make_historical_candidate(
        candidate_id="historical-port",
        value=8010,
        confidence=known(0.9),
    )
    current = make_current_candidate(
        candidate_id="current-port",
        value=8000,
        confidence=known(0.9),
    )

    outcome = resolve_predicate(
        decisions_for(historical, current),
        default_policy(),
    )

    assert outcome.status_for(fact_id_for(current)) is ReconciliationStatus.ACTIVE
    assert (
        outcome.status_for(fact_id_for(historical))
        is ReconciliationStatus.SUPERSEDED
    )
    assert len(outcome.relation_requests) == 1
    replacement = outcome.relation_requests[0]
    assert replacement.from_fact_id == fact_id_for(current)
    assert replacement.to_fact_id == fact_id_for(historical)
    assert replacement.reason == (
        "current direct evidence supersedes historical memory"
    )
    assert replacement.method is (
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
    )


def test_low_confidence_historical_memory_is_still_superseded() -> None:
    policy = default_policy(active_confidence_threshold=0.5)
    historical = make_historical_candidate(
        candidate_id="historical-port",
        value=8010,
        confidence=known(0.1),
    )
    current = make_current_candidate(
        candidate_id="current-port",
        value=8000,
        confidence=known(0.5),
    )

    outcome = resolve_predicate(
        decisions_for(historical, current, policy=policy),
        policy,
    )

    assert outcome.status_for(fact_id_for(current)) is ReconciliationStatus.ACTIVE
    assert (
        outcome.status_for(fact_id_for(historical))
        is ReconciliationStatus.SUPERSEDED
    )
    assert outcome.relation_requests[0].method is (
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
    )


@pytest.mark.parametrize(
    "mismatched",
    (
        make_historical_candidate(
            candidate_id="wrong-source",
            value=8010,
            source_type=SourceType.PROJECT_DOC,
        ),
        make_historical_candidate(
            candidate_id="wrong-hint",
            value=8010,
            status_hint=CandidateStatusHint.HYPOTHESIS,
        ),
        make_historical_candidate(
            candidate_id="unknown-hint",
            value=8010,
            status_hint=unknown("historical hint unknown"),
        ),
        make_historical_candidate(
            candidate_id="unavailable-hint",
            value=8010,
            status_hint=unavailable("historical hint unavailable"),
        ),
    ),
    ids=("wrong-source", "wrong-hint", "unknown-hint", "unavailable-hint"),
)
def test_every_candidate_in_historical_group_must_be_strict_historical(
    mismatched: MemoryCandidate,
) -> None:
    strict = make_historical_candidate(
        candidate_id="strict-historical",
        value=8010,
    )
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(strict, mismatched, current)

    outcome = resolve_predicate(base, default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


def test_nonhistorical_third_group_blocks_current_over_historical_rule() -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current = make_current_candidate(candidate_id="current", value=8000)
    plan = make_candidate(
        candidate_id="plan",
        value=8020,
        source_type=SourceType.PROJECT_DOC,
        status_hint=CandidateStatusHint.PLAN,
    )

    base = decisions_for(historical, current, plan)
    outcome = resolve_predicate(base, default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


def test_disabled_current_over_historical_policy_preserves_base_decisions() -> None:
    policy = default_policy(allow_current_evidence_over_historical=False)
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(historical, current, policy=policy)

    outcome = resolve_predicate(base, policy)

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


@pytest.mark.parametrize(
    "changes",
    (
        {"confidence": known(0.1)},
        {"valid_from": known("2026-09-14T00:00:01+00:00")},
        {
            "valid_from": known("2026-09-13T23:00:00+00:00"),
            "valid_until": known("2026-09-13T23:30:00+00:00"),
        },
    ),
    ids=("low-confidence", "future", "expired"),
)
def test_ineligible_current_evidence_does_not_supersede_history(
    changes: dict[str, object],
) -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current = make_current_candidate(
        candidate_id="current",
        value=8000,
        **changes,
    )
    base = decisions_for(historical, current)

    outcome = resolve_predicate(base, default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


@pytest.mark.parametrize(
    ("competitor", "conflict_expected"),
    (
        (
            make_candidate(
                candidate_id="plan",
                value=8010,
                source_type=SourceType.HISTORICAL_MEMORY,
                status_hint=CandidateStatusHint.PLAN,
            ),
            False,
        ),
        (
            make_candidate(
                candidate_id="hypothesis",
                value=8010,
                source_type=SourceType.PROJECT_DOC,
                status_hint=CandidateStatusHint.HYPOTHESIS,
            ),
            False,
        ),
        (
            make_candidate(
                candidate_id="document-current",
                value=8010,
                source_type=SourceType.PROJECT_DOC,
                status_hint=CandidateStatusHint.CURRENT_FACT,
            ),
            True,
        ),
        (
            make_current_candidate(
                candidate_id="other-current",
                value=8010,
            ),
            True,
        ),
    ),
    ids=("plan", "hypothesis", "document-current", "current"),
)
def test_nonhistorical_competitor_is_not_superseded_and_current_claims_conflict(
    competitor: MemoryCandidate,
    conflict_expected: bool,
) -> None:
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(competitor, current)

    outcome = resolve_predicate(base, default_policy())

    if conflict_expected:
        assert len(outcome.relation_requests) == 2
        assert set(outcome.statuses.values()) == {
            ReconciliationStatus.CONFLICTED
        }
        assert outcome.active_fact_ids == ()
        assert all(
            decision.requires_human_review for decision in outcome.decisions
        )
    else:
        assert outcome.relation_requests == ()
        assert outcome.decisions == base


@pytest.mark.parametrize(
    "changes",
    (
        {"status_hint": CandidateStatusHint.HYPOTHESIS},
        {"source_type": SourceType.PROJECT_DOC},
    ),
    ids=("historical-source-with-wrong-hint", "historical-hint-with-wrong-source"),
)
def test_historical_competitor_requires_matching_source_and_hint(
    changes: dict[str, object],
) -> None:
    mismatched = make_historical_candidate(
        candidate_id="mismatched-history",
        value=8010,
        **changes,
    )
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(mismatched, current)

    outcome = resolve_predicate(base, default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


def test_multiple_current_witnesses_in_one_group_replace_all_historical_groups(
) -> None:
    current_a = make_current_candidate(
        candidate_id="current-a",
        value=8000,
        source_ref="runtime:a",
    )
    current_b = make_current_candidate(
        candidate_id="current-b",
        value=8000,
        source_ref="runtime:b",
    )
    historical_a = make_historical_candidate(
        candidate_id="historical-a",
        value=8010,
        source_ref="history:a",
    )
    historical_b = make_historical_candidate(
        candidate_id="historical-b",
        value=8020,
        source_ref="history:b",
    )

    forward = resolve_predicate(
        decisions_for(current_b, historical_b, historical_a, current_a),
        default_policy(),
    )
    reverse = resolve_predicate(
        tuple(
            reversed(
                decisions_for(current_a, historical_a, historical_b, current_b)
            )
        ),
        default_policy(),
    )

    assert reverse == forward
    assert len(forward.relation_requests) == 2
    assert all(
        request.from_fact_id == fact_id_for(current_a)
        and request.reason == "current direct evidence supersedes historical memory"
        and request.method is ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
        and {"current-a", "current-b"}.issubset(request.candidate_ids)
        and {"runtime:a", "runtime:b"}.issubset(request.evidence_refs)
        for request in forward.relation_requests
    )
    assert all(
        decision.status is ReconciliationStatus.SUPERSEDED
        and decision.resolution_method
        is ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
        and decision.reason == "current direct evidence supersedes historical memory"
        for decision in forward.decisions
        if decision.fact_id != fact_id_for(current_a)
    )
    winner = next(
        decision
        for decision in forward.decisions
        if decision.fact_id == fact_id_for(current_a)
    )
    assert winner.status is ReconciliationStatus.ACTIVE
    assert winner.activation_witness_candidate_ids == ("current-a", "current-b")
    assert winner.resolution_method is (
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
    )


def test_incompatible_current_groups_conflict_without_superseding_history() -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current_a = make_current_candidate(candidate_id="current-a", value=8000)
    current_b = make_current_candidate(candidate_id="current-b", value=8020)
    base = decisions_for(historical, current_a, current_b)

    outcome = resolve_predicate(tuple(reversed(base)), default_policy())

    assert len(outcome.relation_requests) == 2
    assert all(
        request.relation_type is RelationType.CONFLICTS
        for request in outcome.relation_requests
    )
    assert outcome.status_for(fact_id_for(current_a)) is (
        ReconciliationStatus.CONFLICTED
    )
    assert outcome.status_for(fact_id_for(current_b)) is (
        ReconciliationStatus.CONFLICTED
    )
    assert outcome.status_for(fact_id_for(historical)) is (
        ReconciliationStatus.PENDING
    )
    assert outcome.active_fact_ids == ()


def test_same_value_current_and_history_merge_without_fact_relation() -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8000)
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(historical, current)

    outcome = resolve_predicate(base, default_policy())

    assert outcome.relation_requests == ()
    assert outcome.decisions == base


def test_current_over_historical_ignores_rank_recency_majority_and_filename() -> None:
    precedence = tuple(
        (
            source_type,
            1
            if source_type is SourceType.CURRENT_EVIDENCE
            else 99
            if source_type is SourceType.HISTORICAL_MEMORY
            else rank
        )
        for source_type, rank in default_policy().source_precedence
    )
    policy = default_policy(source_precedence=precedence)
    current = make_current_candidate(
        candidate_id="current",
        value=8000,
        source_ref="archive/old-memory.md",
        observed_at=known("2026-09-13T00:00:00+00:00"),
    )
    historical_a = make_historical_candidate(
        candidate_id="historical-a",
        value=8010,
        source_ref="runtime/current-state.json",
        observed_at=known("2026-09-14T00:00:00+00:00"),
    )
    historical_b = make_historical_candidate(
        candidate_id="historical-b",
        value=8010,
        source_ref="runtime/live-state.json",
        observed_at=known("2026-09-14T00:00:00+00:00"),
    )

    outcome = resolve_predicate(
        decisions_for(historical_a, historical_b, current, policy=policy),
        policy,
    )

    assert outcome.status_for(fact_id_for(current)) is ReconciliationStatus.ACTIVE
    assert (
        outcome.status_for(fact_id_for(historical_a))
        is ReconciliationStatus.SUPERSEDED
    )
    assert outcome.relation_requests[0].candidate_ids == (
        "current",
        "historical-a",
        "historical-b",
    )


def test_explicit_supersedes_precedes_current_over_historical_rule() -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current = make_current_candidate(
        candidate_id="current",
        value=8000,
        supersedes=known(("historical",)),
    )

    outcome = resolve_predicate(
        decisions_for(historical, current),
        default_policy(),
    )

    assert len(outcome.relation_requests) == 1
    assert outcome.relation_requests[0].method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert all(
        decision.resolution_method is ResolutionMethod.EXPLICIT_SUPERSEDES
        for decision in outcome.decisions
    )


def test_source_of_truth_precedes_current_over_historical_rule() -> None:
    historical = make_historical_candidate(candidate_id="historical", value=8010)
    current = make_current_candidate(candidate_id="current", value=8000)
    designated = make_source_of_truth_candidate(candidate_id="designated", value=8020)

    outcome = resolve_predicate(
        decisions_for(historical, current, designated),
        default_policy(),
    )

    assert len(outcome.relation_requests) == 2
    assert all(
        request.method is ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH
        for request in outcome.relation_requests
    )
    assert (
        outcome.status_for(fact_id_for(designated))
        is ReconciliationStatus.ACTIVE
    )


@pytest.mark.parametrize(
    "target",
    (
        make_historical_candidate(
            candidate_id="deprecated-target",
            value="deprecated",
            deprecated=known(True),
        ),
        make_candidate(
            candidate_id="plan-target",
            value="planned",
            status_hint=CandidateStatusHint.PLAN,
            source_type=SourceType.USER_EXPLICIT,
            explicit_user_instruction=known(True),
        ),
        make_current_candidate(
            candidate_id="nonknown-target",
            value=unknown("value unavailable"),
        ),
        make_current_candidate(
            candidate_id="invalid-target",
            value="invalid",
            valid_from=known("2026-09-14T00:00:10+00:00"),
            valid_until=known("2026-09-14T00:00:05+00:00"),
        ),
        make_current_candidate(
            candidate_id="future-target",
            value="future",
            valid_from=known("2026-09-14T00:00:01+00:00"),
        ),
        make_current_candidate(
            candidate_id="expired-target",
            value="expired",
            valid_from=known("2026-09-13T23:00:00+00:00"),
            valid_until=known("2026-09-13T23:30:00+00:00"),
        ),
        make_current_candidate(
            candidate_id="insufficient-target",
            value="insufficient",
            confidence=known(0.1),
        ),
    ),
    ids=(
        "deprecated",
        "plan",
        "nonknown-value",
        "invalid-temporal-order",
        "future-validity",
        "expired",
        "insufficient-confidence",
    ),
)
def test_explicit_declaration_does_not_apply_to_ineligible_target(
    target: MemoryCandidate,
) -> None:
    source = make_current_candidate(
        candidate_id="explicit-source",
        value="current",
        supersedes=known((target.candidate_id,)),
    )
    base_decisions = decisions_for(target, source)
    base_target = next(
        decision
        for decision in base_decisions
        if target.candidate_id in decision.candidate_ids
    )
    base_source = next(
        decision
        for decision in base_decisions
        if source.candidate_id in decision.candidate_ids
    )

    outcome = resolve_predicate(base_decisions, default_policy())

    resolved_target = next(
        decision
        for decision in outcome.decisions
        if target.candidate_id in decision.candidate_ids
    )
    resolved_source = next(
        decision
        for decision in outcome.decisions
        if source.candidate_id in decision.candidate_ids
    )
    assert outcome.relation_requests == ()
    assert resolved_target.status is base_target.status
    assert resolved_target.resolution_method is base_target.resolution_method
    assert resolved_target.reason == base_target.reason
    assert resolved_source.status is base_source.status
    assert resolved_source.resolution_method is base_source.resolution_method
    assert resolved_source.reason == base_source.reason


@pytest.mark.parametrize(
    "changes",
    (
        {"subject": "other-project"},
        {"predicate": "other-setting"},
    ),
    ids=("mixed-subject", "mixed-predicate"),
)
def test_predicate_resolver_rejects_mixed_identity_before_resolution(
    changes: dict[str, str],
) -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="designated",
    )
    other = make_historical_candidate(
        candidate_id="other",
        value="other",
        **changes,
    )

    with pytest.raises(
        ReconciliationInputError,
        match="one exact subject and predicate",
    ):
        resolve_predicate(decisions_for(designated, other), default_policy())


def test_predicate_resolver_inherits_policy_mismatch_boundary() -> None:
    classified_policy = default_policy(active_confidence_threshold=0.5)
    resolver_policy = default_policy(active_confidence_threshold=0.9)
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value="new",
    )

    with pytest.raises(ReconciliationInputError, match="classification policy"):
        resolve_predicate(
            decisions_for(designated, policy=classified_policy),
            resolver_policy,
        )


def test_predicate_resolver_inherits_clock_mismatch_boundary() -> None:
    policy = default_policy()
    left = classify_group(
        single_group(
            make_source_of_truth_candidate(candidate_id="left", value="left")
        ),
        NOW,
        policy,
    )
    right = classify_group(
        single_group(
            make_historical_candidate(candidate_id="right", value="right")
        ),
        NOW + timedelta(seconds=1),
        policy,
    )

    with pytest.raises(ReconciliationInputError, match="one reconciliation clock"):
        resolve_predicate((left, right), policy)


def test_predicate_resolver_inherits_canonical_context_boundary() -> None:
    decision = decisions_for(
        make_source_of_truth_candidate(candidate_id="designated")
    )[0]
    object.__delattr__(decision, "_candidate_context")

    with pytest.raises(
        ReconciliationInputError,
        match="canonical candidate context",
    ):
        resolve_predicate((decision,), default_policy())


def test_explicit_candidate_supersession_selects_new_and_old_fact_ids() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.9),
    )
    decisions = decisions_for(old, new)

    replacement = resolve_cross_value_replacements(
        decisions,
        default_policy(),
    )[0]

    assert replacement.from_fact_id == fact_id_for(new)
    assert replacement.to_fact_id == fact_id_for(old)
    assert replacement.reason == "explicit candidate supersedes relation"


def test_superseding_candidate_must_be_an_activation_witness() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    ineligible = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.1),
    )

    requests = resolve_cross_value_replacements(
        decisions_for(old, ineligible),
        default_policy(),
    )

    assert requests == ()


def test_explicit_request_has_direction_method_and_complete_provenance() -> None:
    plausible = make_candidate(
        candidate_id="plausible-old",
        value=8010,
        source_type=SourceType.PROJECT_DOC,
        status_hint=CandidateStatusHint.HYPOTHESIS,
        source_ref="docs:ports.md",
    )
    current = make_current_candidate(
        candidate_id="current-new",
        value=8000,
        source_ref="runtime:server",
        supersedes=known(
            ("plausible-old",),
            source="runtime:server:supersedes",
        ),
        confidence=known(0.9),
    )

    request = resolve_cross_value_replacements(
        decisions_for(plausible, current),
        default_policy(),
    )[0]

    expected_refs = {
        candidate.source_ref
        for candidate in (plausible, current)
    }
    expected_refs.update(
        evidence.source
        for candidate in (plausible, current)
        for evidence in (
            candidate.value,
            candidate.status_hint,
            candidate.observed_at,
            candidate.valid_from,
            candidate.valid_until,
            candidate.confidence,
            candidate.explicit_user_instruction,
            candidate.supersedes,
            candidate.deprecated,
            candidate.metadata,
        )
    )
    assert request.relation_type is RelationType.SUPERSEDES
    assert request.method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert request.from_fact_id == fact_id_for(current)
    assert request.to_fact_id == fact_id_for(plausible)
    assert request.candidate_ids == ("current-new", "plausible-old")
    assert request.evidence_refs == tuple(sorted(expected_refs))


def test_dangling_supersedes_id_is_rejected_before_source_eligibility() -> None:
    ineligible = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("missing",)),
        confidence=known(0.1),
    )

    with pytest.raises(
        ReconciliationInputError,
        match=r"^dangling supersedes reference: new -> missing$",
    ):
        resolve_cross_value_replacements(
            decisions_for(ineligible),
            default_policy(),
        )


@pytest.mark.parametrize(
    "target_fields",
    (
        {"subject": "other-project"},
        {"predicate": "other-setting"},
    ),
    ids=("cross-subject", "cross-predicate"),
)
def test_supersedes_reference_must_preserve_exact_subject_and_predicate(
    target_fields: dict[str, str],
) -> None:
    old = make_historical_candidate(
        candidate_id="old",
        value=8010,
        **target_fields,
    )
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    with pytest.raises(
        ReconciliationInputError,
        match=r"^predicate cohort membership must use one exact subject and predicate$",
    ):
        resolve_cross_value_replacements(
            decisions_for(old, new),
            default_policy(),
        )


def test_candidate_ids_must_be_globally_unique_across_decisions() -> None:
    left = decisions_for(
        make_current_candidate(candidate_id="duplicate", value=8000)
    )
    right = decisions_for(
        make_current_candidate(candidate_id="duplicate", value=8010)
    )

    with pytest.raises(
        ReconciliationInputError,
        match=r"^predicate decisions must belong to one reconciliation cohort$",
    ):
        resolve_cross_value_replacements(
            (*right, *left),
            default_policy(),
        )


def test_fact_ids_must_be_globally_unique_across_decisions() -> None:
    left = decisions_for(
        make_current_candidate(candidate_id="left", value=8000)
    )
    right = decisions_for(
        make_current_candidate(candidate_id="right", value=8000)
    )

    with pytest.raises(
        ReconciliationInputError,
        match=r"^predicate decisions must belong to one reconciliation cohort$",
    ):
        resolve_cross_value_replacements(
            (*right, *left),
            default_policy(),
        )


def test_same_value_reference_stays_in_lineage_without_self_request() -> None:
    old = make_historical_candidate(candidate_id="old", value=8000)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.9),
    )
    decisions = decisions_for(old, new)

    requests = resolve_cross_value_replacements(
        decisions,
        default_policy(),
    )

    assert requests == ()
    assert decisions[0].superseded_candidate_ids == ("old",)


def test_duplicate_declarations_merge_one_request_and_union_provenance() -> None:
    old = make_historical_candidate(
        candidate_id="old",
        value=8010,
        source_ref="history:port",
    )
    first = make_current_candidate(
        candidate_id="new-a",
        value=8000,
        source_ref="runtime:a",
        supersedes=known(("old", "old"), source="runtime:a:edge"),
        confidence=known(0.9),
    )
    second = make_current_candidate(
        candidate_id="new-b",
        value=8000,
        source_ref="runtime:b",
        supersedes=known(("old",), source="runtime:b:edge"),
        confidence=known(0.8),
    )

    requests = resolve_cross_value_replacements(
        decisions_for(old, first, second),
        default_policy(),
    )

    assert len(requests) == 1
    assert requests[0].candidate_ids == ("new-a", "new-b", "old")
    assert set(requests[0].evidence_refs) == set(
        evidence.source
        for candidate in (old, first, second)
        for evidence in (
            candidate.value,
            candidate.status_hint,
            candidate.observed_at,
            candidate.valid_from,
            candidate.valid_until,
            candidate.confidence,
            candidate.explicit_user_instruction,
            candidate.supersedes,
            candidate.deprecated,
            candidate.metadata,
        )
    ) | {candidate.source_ref for candidate in (old, first, second)}


def test_request_order_is_deterministic_under_input_permutation() -> None:
    first_old = make_historical_candidate(
        candidate_id="a-target",
        value=8010,
    )
    second_old = make_historical_candidate(
        candidate_id="z-target",
        value=8020,
    )
    forward = make_current_candidate(
        candidate_id="source",
        value=8000,
        supersedes=known(("a-target", "z-target")),
        confidence=known(0.9),
    )
    reverse = make_current_candidate(
        candidate_id="source",
        value=8000,
        supersedes=known(("z-target", "a-target")),
        confidence=known(0.9),
    )

    forward_requests = resolve_cross_value_replacements(
        decisions_for(first_old, second_old, forward),
        default_policy(),
    )
    reverse_requests = resolve_cross_value_replacements(
        tuple(reversed(decisions_for(second_old, reverse, first_old))),
        default_policy(),
    )

    assert reverse_requests == forward_requests
    assert forward_requests == tuple(
        sorted(
            forward_requests,
            key=lambda request: (
                request.relation_type.value,
                request.from_fact_id,
                request.to_fact_id,
            ),
        )
    )


def test_mutual_explicit_supersedes_emits_no_provisional_relation() -> None:
    left = make_current_candidate(
        candidate_id="left",
        value=8000,
        supersedes=known(("right",), source="left:edge"),
        confidence=known(0.9),
    )
    right = make_current_candidate(
        candidate_id="right",
        value=8010,
        supersedes=known(("left",), source="right:edge"),
        confidence=known(0.9),
    )

    requests = resolve_cross_value_replacements(
        decisions_for(right, left),
        default_policy(),
    )

    assert requests == ()


def test_recency_without_explicit_declaration_creates_no_request() -> None:
    old = make_current_candidate(
        candidate_id="old",
        value=8010,
        observed_at=known("2026-09-13T00:00:00+00:00"),
        valid_from=known("2026-09-13T00:00:00+00:00"),
        confidence=known(0.9),
    )
    recent = make_current_candidate(
        candidate_id="recent",
        value=8000,
        observed_at=known("2026-09-14T00:00:00+00:00"),
        valid_from=known("2026-09-14T00:00:00+00:00"),
        confidence=known(0.9),
    )

    assert resolve_cross_value_replacements(
        decisions_for(old, recent),
        default_policy(),
    ) == ()


def test_relation_request_rejects_fact_self_edge() -> None:
    with pytest.raises(ReconciliationInputError, match="self-edge"):
        RelationRequest(
            relation_type=RelationType.SUPERSEDES,
            from_fact_id="fact-a",
            to_fact_id="fact-a",
            reason="explicit candidate supersedes relation",
            candidate_ids=("new", "old"),
            evidence_refs=("evidence:new", "evidence:old"),
            method=ResolutionMethod.EXPLICIT_SUPERSEDES,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("candidate_ids", ()),
        ("candidate_ids", ("z", "a")),
        ("candidate_ids", ("a", "a")),
        ("evidence_refs", ()),
        ("evidence_refs", ("z", "a")),
        ("evidence_refs", ("a", "a")),
    ),
)
def test_relation_request_requires_sorted_nonempty_unique_provenance(
    field_name: str,
    invalid: tuple[str, ...],
) -> None:
    fields = {
        "relation_type": RelationType.SUPERSEDES,
        "from_fact_id": "fact-new",
        "to_fact_id": "fact-old",
        "reason": "explicit candidate supersedes relation",
        "candidate_ids": ("new", "old"),
        "evidence_refs": ("evidence:new", "evidence:old"),
        "method": ResolutionMethod.EXPLICIT_SUPERSEDES,
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        RelationRequest(**fields)


def test_relation_request_is_frozen() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.9),
    )
    request = resolve_cross_value_replacements(
        decisions_for(old, new),
        default_policy(),
    )[0]

    with pytest.raises(FrozenInstanceError):
        request.reason = "changed"


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("relation_type", "SUPERSEDES"),
        ("from_fact_id", ""),
        ("to_fact_id", "  "),
        ("reason", ""),
        ("method", "EXPLICIT_SUPERSEDES"),
    ),
)
def test_relation_request_validates_typed_required_fields(
    field_name: str,
    invalid: object,
) -> None:
    fields = {
        "relation_type": RelationType.SUPERSEDES,
        "from_fact_id": "fact-new",
        "to_fact_id": "fact-old",
        "reason": "explicit candidate supersedes relation",
        "candidate_ids": ("new", "old"),
        "evidence_refs": ("evidence:new", "evidence:old"),
        "method": ResolutionMethod.EXPLICIT_SUPERSEDES,
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        RelationRequest(**fields)


def test_cross_value_resolver_requires_a_reconciliation_policy() -> None:
    with pytest.raises(ReconciliationInputError, match="policy"):
        resolve_cross_value_replacements((), object())


def test_resolver_rejects_decisions_classified_under_another_policy() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.7),
    )
    classified_policy = default_policy(active_confidence_threshold=0.5)
    stricter_policy = default_policy(active_confidence_threshold=0.9)
    decisions = decisions_for(old, new, policy=classified_policy)

    with pytest.raises(
        ReconciliationInputError,
        match="classification policy",
    ):
        resolve_cross_value_replacements(decisions, stricter_policy)


def test_resolver_rejects_decisions_from_different_reconciliation_clocks() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.9),
    )
    policy = default_policy()
    old_decision = classify_group(single_group(old), NOW, policy)
    new_decision = classify_group(
        single_group(new),
        NOW + timedelta(seconds=1),
        policy,
    )

    with pytest.raises(ReconciliationInputError, match="one reconciliation clock"):
        resolve_cross_value_replacements(
            (old_decision, new_decision),
            policy,
        )


def test_resolver_rejects_fact_decision_missing_canonical_context() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    object.__delattr__(decision, "_candidate_context")

    with pytest.raises(
        ReconciliationInputError,
        match="canonical candidate context",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_resolver_rejects_incomplete_exact_candidate_context() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    incomplete = object.__new__(
        reconciliation_rules._DecisionCandidateContext
    )
    object.__setattr__(decision, "_candidate_context", incomplete)

    with pytest.raises(
        ReconciliationInputError,
        match="canonical candidate context",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_resolver_rejects_incomplete_exact_candidate_group() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    context = decision._candidate_context
    incomplete_group = object.__new__(reconciliation_rules.CandidateGroup)
    object.__setattr__(context, "group", incomplete_group)

    with pytest.raises(
        ReconciliationInputError,
        match="canonical candidate context",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_resolver_rejects_fact_decision_subclass() -> None:
    class HostileFactDecision(FactDecision):
        pass

    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    hostile = object.__new__(HostileFactDecision)
    hostile.__dict__.update(decision.__dict__)

    with pytest.raises(ReconciliationInputError, match="exact FactDecision"):
        resolve_cross_value_replacements(
            (hostile,),
            default_policy(),
        )


@pytest.mark.parametrize(
    ("field_name", "forged"),
    (
        ("fact_id", "fact:v1:forged"),
        ("subject", "forged-subject"),
        ("predicate", "forged-predicate"),
        ("candidate_ids", ("forged-candidate",)),
        ("evidence_refs", ("forged:evidence",)),
    ),
)
def test_resolver_rejects_type_valid_decision_field_forgery(
    field_name: str,
    forged: object,
) -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    object.__setattr__(decision, field_name, forged)

    with pytest.raises(
        ReconciliationInputError,
        match="stage|payload|candidate context",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("fact_id", object()),
        ("candidate_ids", ["current"]),
        ("status", "ACTIVE"),
        ("requires_human_review", 0),
        ("warnings", []),
    ),
)
def test_resolver_rejects_invalid_fact_decision_field_types(
    field_name: str,
    invalid: object,
) -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    object.__setattr__(decision, field_name, invalid)

    with pytest.raises(
        ReconciliationInputError,
        match="canonical FactDecision",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


@pytest.mark.parametrize("invalid", ("not-decisions", (object(),)))
def test_cross_value_resolver_requires_fact_decisions(invalid: object) -> None:
    with pytest.raises(ReconciliationInputError, match="decisions"):
        resolve_cross_value_replacements(invalid, default_policy())


@pytest.mark.parametrize(
    ("factory_name", "relation_type", "method"),
    (
        (
            "supersedes",
            RelationType.SUPERSEDES,
            ResolutionMethod.EXPLICIT_SUPERSEDES,
        ),
        (
            "conflicts",
            RelationType.CONFLICTS,
            ResolutionMethod.UNRESOLVED_CONFLICT,
        ),
    ),
)
def test_relation_request_factories_normalize_provenance(
    factory_name: str,
    relation_type: RelationType,
    method: ResolutionMethod,
) -> None:
    factory = getattr(RelationRequest, factory_name)

    request = factory(
        "fact-new",
        "fact-old",
        "synthetic reason",
        ("old", "new", "old"),
        ("z:evidence", "a:evidence", "z:evidence"),
    )

    assert request.relation_type is relation_type
    assert request.method is method
    assert request.candidate_ids == ("new", "old")
    assert request.evidence_refs == ("a:evidence", "z:evidence")


def test_duplicate_edge_primary_method_uses_fixed_priority_under_permutation(
) -> None:
    explicit = RelationRequest.supersedes(
        "fact-new",
        "fact-old",
        "explicit candidate supersedes relation",
        ("explicit",),
        ("evidence:explicit",),
        method=ResolutionMethod.EXPLICIT_SUPERSEDES,
    )
    source_of_truth = RelationRequest.supersedes(
        "fact-new",
        "fact-old",
        "explicit current source-of-truth designation",
        ("source-of-truth",),
        ("evidence:source-of-truth",),
        method=ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
    )

    forward = reconciliation_rules._merge_relation_requests(
        (source_of_truth, explicit)
    )
    reverse = reconciliation_rules._merge_relation_requests(
        (explicit, source_of_truth)
    )

    assert forward == reverse
    assert forward[0].method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert forward[0].reason == "explicit candidate supersedes relation"
    assert forward[0].candidate_ids == ("explicit", "source-of-truth")
    assert forward[0].evidence_refs == (
        "evidence:explicit",
        "evidence:source-of-truth",
    )


def test_duplicate_edge_merge_materializes_each_edge_once(monkeypatch) -> None:
    requests = tuple(
        RelationRequest.supersedes(
            "fact-new",
            "fact-old",
            "explicit candidate supersedes relation",
            (f"candidate-{index:04d}",),
            (f"evidence-{index:04d}",),
        )
        for index in range(200)
    )
    original = RelationRequest.supersedes
    materializations = 0

    def counting_factory(*args: object, **kwargs: object) -> RelationRequest:
        nonlocal materializations
        materializations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(RelationRequest, "supersedes", counting_factory)

    merged = reconciliation_rules._merge_relation_requests(requests)

    assert len(merged) == 1
    assert materializations == 1
    assert len(merged[0].candidate_ids) == 200
    assert len(merged[0].evidence_refs) == 200


class _UntrustedTuple(tuple):
    pass


class _UntrustedString(str):
    pass


class _AlwaysEqualWitnessTuple(tuple):
    def __eq__(self, other: object) -> bool:
        return True

    def __contains__(self, item: object) -> bool:
        return True


class _ExplosiveTuple(tuple):
    def __iter__(self):
        raise AssertionError("hostile tuple iteration dispatched")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile tuple equality dispatched")


class _ExplosiveString(str):
    def strip(self, chars=None):
        raise AssertionError("hostile string strip dispatched")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile string equality dispatched")


class _EvilRelationRequest(RelationRequest):
    pass


class _EvilTimezone(tzinfo):
    def utcoffset(self, dt):
        raise RuntimeError("hostile timezone offset dispatched")

    def dst(self, dt):
        return timedelta(0)


@pytest.mark.parametrize("factory_name", ("supersedes", "conflicts"))
@pytest.mark.parametrize("field_name", ("candidate_ids", "evidence_refs"))
@pytest.mark.parametrize(
    "invalid",
    (
        "candidate",
        ["candidate"],
        {"candidate": True},
        None,
        ("candidate", 1),
        _UntrustedTuple(("candidate",)),
        (_UntrustedString("candidate"),),
    ),
    ids=(
        "string",
        "list",
        "mapping",
        "none",
        "mixed",
        "tuple-subclass",
        "string-subclass",
    ),
)
def test_relation_request_factories_reject_untrusted_provenance_inputs(
    factory_name: str,
    field_name: str,
    invalid: object,
) -> None:
    factory = getattr(RelationRequest, factory_name)
    fields = {
        "candidate_ids": ("candidate",),
        "evidence_refs": ("evidence",),
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        factory(
            "fact-new",
            "fact-old",
            "synthetic reason",
            fields["candidate_ids"],
            fields["evidence_refs"],
        )


@pytest.mark.parametrize(
    ("factory_name", "method"),
    (
        ("supersedes", ResolutionMethod.UNRESOLVED_CONFLICT),
        ("supersedes", ResolutionMethod.DIRECT_CURRENT),
        ("conflicts", ResolutionMethod.EXPLICIT_SUPERSEDES),
        ("conflicts", ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH),
    ),
)
def test_relation_request_factories_reject_incoherent_methods(
    factory_name: str,
    method: ResolutionMethod,
) -> None:
    factory = getattr(RelationRequest, factory_name)

    with pytest.raises(ReconciliationInputError, match="method"):
        factory(
            "fact-new",
            "fact-old",
            "synthetic reason",
            ("candidate",),
            ("evidence",),
            method=method,
        )


def test_hostile_activation_witness_tuple_cannot_authorize_replacement() -> None:
    old = make_historical_candidate(candidate_id="old", value=8010)
    ineligible = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
        confidence=known(0.1),
    )
    decisions = list(decisions_for(old, ineligible))
    source_decision = next(
        decision
        for decision in decisions
        if decision.fact_id == fact_id_for(ineligible)
    )
    object.__setattr__(
        source_decision,
        "activation_witness_candidate_ids",
        _AlwaysEqualWitnessTuple(("forged-witness",)),
    )

    with pytest.raises(
        ReconciliationInputError,
        match="activation_witness_candidate_ids",
    ):
        resolve_cross_value_replacements(
            tuple(decisions),
            default_policy(),
        )


@pytest.mark.parametrize(
    ("relation_type", "method"),
    (
        (RelationType.SUPERSEDES, ResolutionMethod.UNRESOLVED_CONFLICT),
        (RelationType.CONFLICTS, ResolutionMethod.EXPLICIT_SUPERSEDES),
    ),
)
def test_relation_request_constructor_rejects_incoherent_method(
    relation_type: RelationType,
    method: ResolutionMethod,
) -> None:
    with pytest.raises(ReconciliationInputError, match="coherent"):
        RelationRequest(
            relation_type=relation_type,
            from_fact_id="fact-new",
            to_fact_id="fact-old",
            reason="synthetic reason",
            candidate_ids=("candidate",),
            evidence_refs=("evidence",),
            method=method,
        )


@pytest.mark.parametrize("field_name", ("candidate_ids", "evidence_refs"))
@pytest.mark.parametrize(
    "invalid",
    (
        _ExplosiveTuple(("candidate",)),
        (_ExplosiveString("candidate"),),
    ),
    ids=("tuple-subclass", "string-subclass"),
)
def test_relation_request_constructor_rejects_hostile_provenance_without_dispatch(
    field_name: str,
    invalid: tuple[str, ...],
) -> None:
    fields = {
        "candidate_ids": ("candidate",),
        "evidence_refs": ("evidence",),
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        RelationRequest(
            relation_type=RelationType.SUPERSEDES,
            from_fact_id="fact-new",
            to_fact_id="fact-old",
            reason="synthetic reason",
            candidate_ids=fields["candidate_ids"],
            evidence_refs=fields["evidence_refs"],
            method=ResolutionMethod.EXPLICIT_SUPERSEDES,
        )


@pytest.mark.parametrize("factory_name", ("supersedes", "conflicts"))
def test_relation_request_factory_rejects_subclass_dispatch(
    factory_name: str,
) -> None:
    factory = getattr(_EvilRelationRequest, factory_name)

    with pytest.raises(ReconciliationInputError, match="exact RelationRequest"):
        factory(
            "fact-new",
            "fact-old",
            "synthetic reason",
            ("candidate",),
            ("evidence",),
        )


def test_tampered_retained_policy_is_rejected_before_reclassification() -> None:
    candidate = make_current_candidate(candidate_id="current", value=8000)
    decision = decisions_for(candidate)[0]
    context = decision._candidate_context
    object.__setattr__(
        context.policy,
        "active_confidence_threshold",
        "hostile-threshold",
    )

    with pytest.raises(
        ReconciliationInputError,
        match="candidate context policy",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_uninitialized_exact_retained_policy_is_rejected_deterministically(
) -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    context = decision._candidate_context
    object.__setattr__(
        context,
        "policy",
        object.__new__(ReconciliationPolicy),
    )

    with pytest.raises(
        ReconciliationInputError,
        match="candidate context policy",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_hostile_reconciliation_clock_timezone_is_rejected_deterministically(
) -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    context = decision._candidate_context
    object.__setattr__(
        context,
        "reconciled_at",
        NOW.replace(tzinfo=_EvilTimezone()),
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical candidate context",
    ):
        resolve_cross_value_replacements(
            (decision,),
            default_policy(),
        )


def test_uninitialized_exact_decision_with_stolen_context_is_rejected(
) -> None:
    canonical = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    malformed = object.__new__(FactDecision)
    object.__setattr__(
        malformed,
        "_candidate_context",
        canonical._candidate_context,
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical FactDecision",
    ):
        resolve_cross_value_replacements(
            (malformed,),
            default_policy(),
        )
