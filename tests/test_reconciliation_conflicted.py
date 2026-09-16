"""Conflict classification for unresolved cross-value predicate claims."""

from dataclasses import replace
from itertools import permutations

import pytest

from agent_memory_os.reconcile import rules as reconciliation_rules
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    MemoryCandidate,
    ReconciliationInputError,
    ReconciliationStatus,
    RelationType,
    ResolutionMethod,
    SourceType,
    WarningCode,
)
from agent_memory_os.reconcile.rules import PredicateOutcome, resolve_predicate

from reconciliation_helpers import (
    decisions_for,
    default_policy,
    fact_id_for,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    make_source_of_truth_candidate,
    unknown,
)


def equal_current_conflict() -> tuple[MemoryCandidate, MemoryCandidate]:
    """Return the exact shared equal-current Task 16 fixture."""

    return (
        make_current_candidate(
            candidate_id="state",
            value=8000,
            source_type=SourceType.PROJECT_DOC,
            confidence=known(0.9),
        ),
        make_current_candidate(
            candidate_id="readme",
            value=8010,
            source_type=SourceType.PROJECT_DOC,
            confidence=known(0.9),
        ),
    )


def _permuted_outcomes(
    candidates: tuple[MemoryCandidate, ...],
) -> tuple[PredicateOutcome, ...]:
    policy = default_policy()
    outcomes = []
    for candidate_order in permutations(candidates):
        decisions = decisions_for(*candidate_order, policy=policy)
        outcomes.append(resolve_predicate(decisions, policy))
        outcomes.append(resolve_predicate(tuple(reversed(decisions)), policy))
    return tuple(outcomes)


def test_equal_current_8000_and_8010_are_conflicted_without_winner() -> None:
    left, right = equal_current_conflict()

    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert all(
        request.relation_type is RelationType.CONFLICTS
        for request in outcome.relation_requests
    )
    assert all(
        decision.requires_human_review for decision in outcome.decisions
    )


def test_equal_current_conflict_has_exact_symmetric_requests() -> None:
    left, right = equal_current_conflict()

    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    left_id = fact_id_for(left)
    right_id = fact_id_for(right)
    assert {
        (request.from_fact_id, request.to_fact_id)
        for request in outcome.relation_requests
    } == {(left_id, right_id), (right_id, left_id)}
    assert all(
        request.reason == "unresolved competing current facts"
        and request.method is ResolutionMethod.UNRESOLVED_CONFLICT
        and request.candidate_ids == ("readme", "state")
        for request in outcome.relation_requests
    )
    assert all(
        decision.reason == "unresolved competing current facts"
        and decision.resolution_method
        is ResolutionMethod.UNRESOLVED_CONFLICT
        and decision.activation_witness_candidate_ids == ()
        for decision in outcome.decisions
    )


def test_multiple_incompatible_source_of_truth_values_are_conflicted() -> None:
    left = make_source_of_truth_candidate(
        candidate_id="explicit-8000",
        value=8000,
    )
    right = make_source_of_truth_candidate(
        candidate_id="explicit-8010",
        value=8010,
    )

    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert len(outcome.relation_requests) == 2


def test_unknown_source_time_blocks_cross_value_winner_selection() -> None:
    left = make_current_candidate(
        candidate_id="state",
        value=8000,
        source_type=SourceType.PROJECT_DOC,
        observed_at=unknown("document observation time is unknown"),
    )
    right = make_current_candidate(
        candidate_id="readme",
        value=8010,
        source_type=SourceType.PROJECT_DOC,
        observed_at=unknown("document observation time is unknown"),
    )

    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()


def test_mutual_explicit_supersedes_is_a_review_requiring_conflict() -> None:
    left = make_current_candidate(
        candidate_id="left",
        value=8000,
        supersedes=known(("right",), source="left:edge"),
    )
    right = make_current_candidate(
        candidate_id="right",
        value=8010,
        supersedes=known(("left",), source="right:edge"),
    )

    outcome = resolve_predicate(
        decisions_for(right, left),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert all(
        request.reason == "contradictory explicit supersedes declarations"
        for request in outcome.relation_requests
    )
    assert all(
        decision.reason == "contradictory explicit supersedes declarations"
        and decision.requires_human_review
        and any(
            warning.code is WarningCode.CONTRADICTORY_SUPERSEDES
            and warning.requires_human_review
            for warning in decision.warnings
        )
        for decision in outcome.decisions
    )


def test_mutual_explicit_contradiction_conflicts_every_current_value() -> None:
    left = make_current_candidate(
        candidate_id="left",
        value=8000,
        supersedes=known(("right",)),
    )
    right = make_current_candidate(
        candidate_id="right",
        value=8010,
        supersedes=known(("left",)),
    )
    third = make_current_candidate(
        candidate_id="third",
        value=8020,
        source_type=SourceType.PROJECT_DOC,
    )

    outcome = resolve_predicate(
        decisions_for(third, right, left),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert len(outcome.relation_requests) == 6
    assert all(
        request.reason == "contradictory explicit supersedes declarations"
        for request in outcome.relation_requests
    )


def test_explicit_replacement_with_unrelated_current_rolls_back_to_conflict(
) -> None:
    replaced = make_current_candidate(candidate_id="replaced", value=8010)
    explicit = make_current_candidate(
        candidate_id="explicit",
        value=8000,
        supersedes=known(("replaced",)),
    )
    unrelated = make_current_candidate(
        candidate_id="unrelated",
        value=8020,
        source_type=SourceType.PROJECT_DOC,
    )

    outcomes = _permuted_outcomes((replaced, explicit, unrelated))

    assert all(outcome == outcomes[0] for outcome in outcomes)
    outcome = outcomes[0]
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert len(outcome.relation_requests) == 6
    assert all(
        request.relation_type is RelationType.CONFLICTS
        for request in outcome.relation_requests
    )


def test_rolled_back_explicit_target_keeps_its_pending_base_status() -> None:
    historical = make_historical_candidate(
        candidate_id="historical",
        value=8010,
    )
    explicit = make_current_candidate(
        candidate_id="explicit",
        value=8000,
        supersedes=known(("historical",)),
    )
    unrelated = make_current_candidate(
        candidate_id="unrelated",
        value=8020,
        source_type=SourceType.PROJECT_DOC,
    )

    outcomes = _permuted_outcomes((historical, explicit, unrelated))

    assert all(outcome == outcomes[0] for outcome in outcomes)
    outcome = outcomes[0]
    assert outcome.status_for(fact_id_for(historical)) is (
        ReconciliationStatus.PENDING
    )
    assert outcome.status_for(fact_id_for(explicit)) is (
        ReconciliationStatus.CONFLICTED
    )
    assert outcome.status_for(fact_id_for(unrelated)) is (
        ReconciliationStatus.CONFLICTED
    )
    assert outcome.active_fact_ids == ()
    assert len(outcome.relation_requests) == 2
    assert all(
        request.relation_type is RelationType.CONFLICTS
        for request in outcome.relation_requests
    )


def test_mutual_contradiction_outranks_unrelated_source_of_truth() -> None:
    left = make_current_candidate(
        candidate_id="left",
        value=8000,
        supersedes=known(("right",)),
    )
    right = make_current_candidate(
        candidate_id="right",
        value=8010,
        supersedes=known(("left",)),
    )
    source_of_truth = make_source_of_truth_candidate(
        candidate_id="source-of-truth",
        value=8020,
    )

    outcomes = _permuted_outcomes((left, right, source_of_truth))

    assert all(outcome == outcomes[0] for outcome in outcomes)
    outcome = outcomes[0]
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert len(outcome.relation_requests) == 6
    assert all(
        request.reason == "contradictory explicit supersedes declarations"
        for request in outcome.relation_requests
    )
    assert all(
        decision.reason == "contradictory explicit supersedes declarations"
        and any(
            warning.code is WarningCode.CONTRADICTORY_SUPERSEDES
            and warning.candidate_ids == ("left", "right")
            and "synthetic:source-of-truth:value"
            not in warning.evidence_refs
            for warning in decision.warnings
        )
        for decision in outcome.decisions
    )


def test_mutual_contradiction_rolls_back_unrelated_explicit_replacement(
) -> None:
    left = make_current_candidate(
        candidate_id="left",
        value=8000,
        supersedes=known(("right",)),
    )
    right = make_current_candidate(
        candidate_id="right",
        value=8010,
        supersedes=known(("left",)),
    )
    replaced = make_current_candidate(candidate_id="replaced", value=8030)
    explicit = make_current_candidate(
        candidate_id="explicit",
        value=8020,
        supersedes=known(("replaced",)),
    )

    outcomes = _permuted_outcomes((left, right, explicit, replaced))

    assert all(outcome == outcomes[0] for outcome in outcomes)
    outcome = outcomes[0]
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()
    assert len(outcome.relation_requests) == 12
    assert all(
        request.relation_type is RelationType.CONFLICTS
        and request.reason
        == "contradictory explicit supersedes declarations"
        for request in outcome.relation_requests
    )


def test_isolated_explicit_replacement_chain_remains_authorized() -> None:
    oldest = make_current_candidate(candidate_id="oldest", value=8020)
    middle = make_current_candidate(
        candidate_id="middle",
        value=8010,
        supersedes=known(("oldest",)),
    )
    latest = make_current_candidate(
        candidate_id="latest",
        value=8000,
        supersedes=known(("middle",)),
    )

    outcomes = _permuted_outcomes((oldest, middle, latest))

    assert all(outcome == outcomes[0] for outcome in outcomes)
    outcome = outcomes[0]
    assert outcome.status_for(fact_id_for(latest)) is (
        ReconciliationStatus.ACTIVE
    )
    assert outcome.status_for(fact_id_for(middle)) is (
        ReconciliationStatus.SUPERSEDED
    )
    assert outcome.status_for(fact_id_for(oldest)) is (
        ReconciliationStatus.SUPERSEDED
    )
    assert len(outcome.relation_requests) == 2
    assert all(
        request.relation_type is RelationType.SUPERSEDES
        for request in outcome.relation_requests
    )


def test_three_incompatible_values_emit_every_symmetric_pair() -> None:
    candidates = tuple(
        make_current_candidate(
            candidate_id=f"doc-{value}",
            value=value,
            source_type=SourceType.PROJECT_DOC,
            source_ref=f"docs/{value}.md",
        )
        for value in (8000, 8010, 8020)
    )

    outcome = resolve_predicate(
        decisions_for(*candidates),
        default_policy(),
    )

    fact_ids = {fact_id_for(candidate) for candidate in candidates}
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert len(outcome.relation_requests) == 6
    assert {
        (request.from_fact_id, request.to_fact_id)
        for request in outcome.relation_requests
    } == {
        (left_id, right_id)
        for left_id in fact_ids
        for right_id in fact_ids
        if left_id != right_id
    }
    decisions_by_id = {
        decision.fact_id: decision for decision in outcome.decisions
    }
    assert all(
        request.candidate_ids
        == tuple(
            sorted(
                set(
                    (*decisions_by_id[request.from_fact_id].candidate_ids,
                     *decisions_by_id[request.to_fact_id].candidate_ids)
                )
            )
        )
        and request.evidence_refs
        == tuple(
            sorted(
                set(
                    (*decisions_by_id[request.from_fact_id].evidence_refs,
                     *decisions_by_id[request.to_fact_id].evidence_refs)
                )
            )
        )
        for request in outcome.relation_requests
    )


def test_conflict_resolution_is_input_order_independent() -> None:
    left, right = equal_current_conflict()
    forward = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )
    reverse = resolve_predicate(
        tuple(reversed(decisions_for(right, left))),
        default_policy(),
    )

    assert reverse == forward


def test_filename_recency_and_majority_do_not_create_a_winner() -> None:
    older_majority = (
        make_current_candidate(
            candidate_id="state-old",
            value=8000,
            source_type=SourceType.PROJECT_DOC,
            source_ref="docs/STATE.md",
            observed_at=known("2026-09-12T00:00:00+00:00"),
        ),
        make_current_candidate(
            candidate_id="state-copy",
            value=8000,
            source_type=SourceType.PROJECT_DOC,
            source_ref="docs/STATE-COPY.md",
            observed_at=known("2026-09-12T00:00:00+00:00"),
        ),
    )
    newer_minority = make_current_candidate(
        candidate_id="readme-new",
        value=8010,
        source_type=SourceType.PROJECT_DOC,
        source_ref="README.md",
        observed_at=known("2026-09-14T00:00:00+00:00"),
    )

    outcome = resolve_predicate(
        decisions_for(*older_majority, newer_minority),
        default_policy(),
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()


def test_disabling_equal_precedence_conflict_cannot_authorize_a_winner() -> None:
    left, right = equal_current_conflict()
    policy = default_policy(
        conflict_on_equal_precedence_disagreement=False,
    )

    outcome = resolve_predicate(
        decisions_for(left, right, policy=policy),
        policy,
    )

    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.CONFLICTED
    }
    assert outcome.active_fact_ids == ()


@pytest.mark.parametrize(
    "excluded",
    (
        make_candidate(
            candidate_id="plan",
            value=8010,
            source_type=SourceType.PROJECT_DOC,
            status_hint=CandidateStatusHint.PLAN,
        ),
        make_candidate(
            candidate_id="hypothesis",
            value=8010,
            source_type=SourceType.PROJECT_DOC,
            status_hint=CandidateStatusHint.HYPOTHESIS,
        ),
        make_current_candidate(
            candidate_id="deprecated",
            value=8010,
            deprecated=known(True),
        ),
        make_candidate(
            candidate_id="nonknown",
            value=unknown("value not observed"),
            status_hint=CandidateStatusHint.CURRENT_FACT,
            source_type=SourceType.PROJECT_DOC,
        ),
    ),
    ids=("plan", "hypothesis", "deprecated", "nonknown-value"),
)
def test_pending_deprecated_and_nonknown_groups_are_not_conflict_eligible(
    excluded: MemoryCandidate,
) -> None:
    current = make_current_candidate(candidate_id="current", value=8000)
    base = decisions_for(current, excluded)

    outcome = resolve_predicate(base, default_policy())

    assert outcome.decisions == base
    assert outcome.relation_requests == ()


def test_explicit_supersedes_winner_remains_authorized() -> None:
    old = make_current_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
    )

    outcome = resolve_predicate(
        decisions_for(old, new),
        default_policy(),
    )

    assert outcome.status_for(fact_id_for(new)) is ReconciliationStatus.ACTIVE
    assert outcome.status_for(fact_id_for(old)) is ReconciliationStatus.SUPERSEDED
    assert outcome.relation_requests[0].method is ResolutionMethod.EXPLICIT_SUPERSEDES


def test_single_source_of_truth_winner_remains_authorized() -> None:
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value=8000,
    )
    current = make_current_candidate(candidate_id="current", value=8010)

    outcome = resolve_predicate(
        decisions_for(current, designated),
        default_policy(),
    )

    assert (
        outcome.status_for(fact_id_for(designated))
        is ReconciliationStatus.ACTIVE
    )
    assert (
        outcome.status_for(fact_id_for(current))
        is ReconciliationStatus.SUPERSEDED
    )
    assert outcome.relation_requests[0].method is (
        ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH
    )


def test_explicit_source_of_truth_can_apply_distinct_replacement_rules() -> None:
    historical = make_historical_candidate(
        candidate_id="historical",
        value=7990,
    )
    designated = make_source_of_truth_candidate(
        candidate_id="designated",
        value=8000,
        supersedes=known(("historical",)),
    )
    current = make_current_candidate(candidate_id="current", value=8010)

    outcome = resolve_predicate(
        decisions_for(historical, designated, current),
        default_policy(),
    )

    designated_id = fact_id_for(designated)
    historical_id = fact_id_for(historical)
    current_id = fact_id_for(current)
    assert {
        (
            request.from_fact_id,
            request.to_fact_id,
            request.method,
        )
        for request in outcome.relation_requests
    } == {
        (
            designated_id,
            historical_id,
            ResolutionMethod.EXPLICIT_SUPERSEDES,
        ),
        (
            designated_id,
            current_id,
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
        ),
    }
    winner = next(
        decision
        for decision in outcome.decisions
        if decision.fact_id == designated_id
    )
    assert winner.status is ReconciliationStatus.ACTIVE
    assert winner.resolution_method is ResolutionMethod.EXPLICIT_SUPERSEDES
    assert winner.reason == "explicit candidate supersedes relation"
    assert outcome.status_for(historical_id) is ReconciliationStatus.SUPERSEDED
    assert outcome.status_for(current_id) is ReconciliationStatus.SUPERSEDED


def test_current_over_historical_winner_remains_authorized() -> None:
    current = make_current_candidate(candidate_id="current", value=8000)
    historical = make_historical_candidate(
        candidate_id="historical",
        value=8010,
    )

    outcome = resolve_predicate(
        decisions_for(historical, current),
        default_policy(),
    )

    assert (
        outcome.status_for(fact_id_for(current))
        is ReconciliationStatus.ACTIVE
    )
    assert (
        outcome.status_for(fact_id_for(historical))
        is ReconciliationStatus.SUPERSEDED
    )
    assert outcome.relation_requests[0].method is (
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
    )


def test_predicate_outcome_rejects_asymmetric_conflict_requests() -> None:
    left, right = equal_current_conflict()
    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    with pytest.raises(ReconciliationInputError, match="exactly symmetric"):
        PredicateOutcome(outcome.decisions, outcome.relation_requests[:1])


def test_predicate_outcome_rejects_inexact_conflict_provenance() -> None:
    left, right = equal_current_conflict()
    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )
    malformed = replace(
        outcome.relation_requests[0],
        candidate_ids=(outcome.relation_requests[0].candidate_ids[0],),
    )

    with pytest.raises(ReconciliationInputError, match="exact endpoint provenance"):
        PredicateOutcome(
            outcome.decisions,
            (malformed, outcome.relation_requests[1]),
        )


def test_predicate_outcome_rejects_noncanonical_conflict_reason() -> None:
    left, right = equal_current_conflict()
    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )
    malformed = tuple(
        replace(request, reason="noncanonical conflict reason")
        for request in outcome.relation_requests
    )

    with pytest.raises(ReconciliationInputError, match="canonical reason"):
        PredicateOutcome(outcome.decisions, malformed)


def test_predicate_outcome_requires_every_conflicted_pair() -> None:
    candidates = tuple(
        make_current_candidate(
            candidate_id=f"doc-{value}",
            value=value,
            source_type=SourceType.PROJECT_DOC,
        )
        for value in (8000, 8010, 8020)
    )
    outcome = resolve_predicate(
        decisions_for(*candidates),
        default_policy(),
    )
    omitted_pair = {
        outcome.relation_requests[0].from_fact_id,
        outcome.relation_requests[0].to_fact_id,
    }
    incomplete = tuple(
        request
        for request in outcome.relation_requests
        if {request.from_fact_id, request.to_fact_id} != omitted_pair
    )

    with pytest.raises(ReconciliationInputError, match="every directed pair"):
        PredicateOutcome(outcome.decisions, incomplete)


def test_predicate_outcome_rejects_direct_construction() -> None:
    left, right = equal_current_conflict()
    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    with pytest.raises(ReconciliationInputError, match="canonical factory"):
        PredicateOutcome(outcome.decisions, outcome.relation_requests)


def test_predicate_outcome_rejects_dataclass_replace_forgery() -> None:
    left, right = equal_current_conflict()
    outcome = resolve_predicate(
        decisions_for(left, right),
        default_policy(),
    )

    with pytest.raises(ReconciliationInputError, match="canonical factory"):
        replace(outcome)


@pytest.mark.parametrize("mutation", ("duplicate", "reversed"))
def test_predicate_outcome_rejects_noncanonical_relation_keys_and_order(
    mutation: str,
) -> None:
    candidates = tuple(
        make_current_candidate(
            candidate_id=f"doc-{value}",
            value=value,
            source_type=SourceType.PROJECT_DOC,
        )
        for value in (8000, 8010, 8020)
    )
    outcome = resolve_predicate(
        decisions_for(*candidates),
        default_policy(),
    )
    malformed = (
        (*outcome.relation_requests, outcome.relation_requests[-1])
        if mutation == "duplicate"
        else tuple(reversed(outcome.relation_requests))
    )

    with pytest.raises(
        ReconciliationInputError,
        match="relation request keys must be sorted and unique",
    ):
        PredicateOutcome(outcome.decisions, malformed)


@pytest.mark.parametrize(
    "changes",
    (
        {"reason": "fabricated replacement reason"},
        {"candidate_ids": ("new",)},
        {"evidence_refs": ("fabricated:evidence",)},
        {
            "method": ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
            "reason": "explicit current source-of-truth designation",
        },
    ),
    ids=("reason", "candidate-provenance", "evidence-provenance", "method"),
)
def test_predicate_outcome_rejects_fabricated_supersedes_payload(
    changes: dict[str, object],
) -> None:
    old = make_current_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
    )
    outcome = resolve_predicate(
        decisions_for(old, new),
        default_policy(),
    )
    malformed = replace(outcome.relation_requests[0], **changes)

    with pytest.raises(
        ReconciliationInputError,
        match="SUPERSEDES request payload is not canonical",
    ):
        PredicateOutcome(outcome.decisions, (malformed,))


def test_canonical_factory_rejects_forged_source_of_truth_rule() -> None:
    old = make_current_candidate(candidate_id="old", value=8010)
    new = make_current_candidate(
        candidate_id="new",
        value=8000,
        supersedes=known(("old",)),
    )
    outcome = resolve_predicate(
        decisions_for(old, new),
        default_policy(),
    )
    reason = "explicit current source-of-truth designation"
    for decision in outcome.decisions:
        object.__setattr__(
            decision,
            "resolution_method",
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
        )
        object.__setattr__(decision, "reason", reason)
    malformed = replace(
        outcome.relation_requests[0],
        method=ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
        reason=reason,
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            outcome.decisions,
            (malformed,),
        )


def test_canonical_factory_rejects_explicit_ineligible_target() -> None:
    deprecated = make_current_candidate(
        candidate_id="deprecated",
        value=8010,
        deprecated=known(True),
    )
    explicit = make_current_candidate(
        candidate_id="explicit",
        value=8000,
        supersedes=known(("deprecated",)),
    )
    policy = default_policy()
    decisions = decisions_for(deprecated, explicit, policy=policy)
    declared = reconciliation_rules.resolve_cross_value_replacements(
        decisions,
        policy,
    )
    reason = "explicit candidate supersedes relation"
    for decision in decisions:
        object.__setattr__(
            decision,
            "resolution_method",
            ResolutionMethod.EXPLICIT_SUPERSEDES,
        )
        object.__setattr__(decision, "reason", reason)
        if decision.fact_id == fact_id_for(deprecated):
            object.__setattr__(
                decision,
                "status",
                ReconciliationStatus.SUPERSEDED,
            )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            decisions,
            declared,
        )


def test_canonical_factory_requires_every_eligible_explicit_target() -> None:
    first = make_historical_candidate(
        candidate_id="historical-one",
        value=7990,
    )
    second = make_historical_candidate(
        candidate_id="historical-two",
        value=7980,
    )
    explicit = make_current_candidate(
        candidate_id="explicit",
        value=8000,
        supersedes=known(("historical-one", "historical-two")),
    )
    policy = default_policy()
    base_decisions = decisions_for(first, second, explicit, policy=policy)
    outcome = resolve_predicate(base_decisions, policy)
    first_id = fact_id_for(first)
    second_id = fact_id_for(second)
    retained_request = next(
        request
        for request in outcome.relation_requests
        if request.to_fact_id == first_id
    )
    base_by_fact_id = {
        decision.fact_id: decision for decision in base_decisions
    }
    incomplete_decisions = tuple(
        base_by_fact_id[second_id]
        if decision.fact_id == second_id
        else decision
        for decision in outcome.decisions
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            incomplete_decisions,
            (retained_request,),
        )


def test_canonical_factory_rejects_conflicted_pending_plan_groups() -> None:
    plans = (
        make_current_candidate(
            candidate_id="plan-one",
            value=8000,
            status_hint=CandidateStatusHint.PLAN,
        ),
        make_current_candidate(
            candidate_id="plan-two",
            value=8010,
            status_hint=CandidateStatusHint.PLAN,
        ),
    )
    base_decisions = decisions_for(*plans)
    reason = "unresolved competing current facts"
    conflicted = tuple(
        reconciliation_rules._with_conflicted_status(decision, reason)
        for decision in base_decisions
    )
    left, right = conflicted
    candidate_ids = tuple((*left.candidate_ids, *right.candidate_ids))
    evidence_refs = tuple((*left.evidence_refs, *right.evidence_refs))
    requests = reconciliation_rules._merge_relation_requests(
        (
            reconciliation_rules.RelationRequest.conflicts(
                left.fact_id,
                right.fact_id,
                reason,
                candidate_ids,
                evidence_refs,
            ),
            reconciliation_rules.RelationRequest.conflicts(
                right.fact_id,
                left.fact_id,
                reason,
                candidate_ids,
                evidence_refs,
            ),
        )
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            conflicted,
            requests,
        )


def test_canonical_factory_rejects_pending_plan_changed_to_active() -> None:
    decision = decisions_for(
        make_current_candidate(
            candidate_id="plan",
            value=8000,
            status_hint=CandidateStatusHint.PLAN,
        )
    )[0]
    object.__setattr__(decision, "status", ReconciliationStatus.ACTIVE)

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome((decision,), ())


def test_canonical_factory_rejects_active_changed_to_pending() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    object.__setattr__(decision, "status", ReconciliationStatus.PENDING)

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome((decision,), ())


def test_canonical_factory_rejects_sot_request_with_multiple_sot_groups(
) -> None:
    first = make_source_of_truth_candidate(
        candidate_id="first-sot",
        value=8000,
    )
    second = make_source_of_truth_candidate(
        candidate_id="second-sot",
        value=8010,
    )
    decisions = decisions_for(first, second)
    source_id = fact_id_for(first)
    source = next(
        decision for decision in decisions if decision.fact_id == source_id
    )
    target = next(
        decision for decision in decisions if decision.fact_id != source_id
    )
    reason = "explicit current source-of-truth designation"
    for decision in decisions:
        object.__setattr__(
            decision,
            "resolution_method",
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
        )
        object.__setattr__(decision, "reason", reason)
    object.__setattr__(
        target,
        "status",
        ReconciliationStatus.SUPERSEDED,
    )
    request = reconciliation_rules.RelationRequest.supersedes(
        source.fact_id,
        target.fact_id,
        reason,
        tuple((*source.candidate_ids, *target.candidate_ids)),
        tuple((*source.evidence_refs, *target.evidence_refs)),
        method=ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            decisions,
            (request,),
        )


def test_canonical_factory_rejects_current_history_with_extra_current_root(
) -> None:
    selected = make_current_candidate(
        candidate_id="selected-current",
        value=8000,
    )
    extra = make_current_candidate(
        candidate_id="extra-current",
        value=8010,
    )
    historical = make_historical_candidate(
        candidate_id="historical",
        value=8020,
    )
    decisions = decisions_for(selected, extra, historical)
    source = next(
        decision
        for decision in decisions
        if decision.fact_id == fact_id_for(selected)
    )
    target = next(
        decision
        for decision in decisions
        if decision.fact_id == fact_id_for(historical)
    )
    reason = "current direct evidence supersedes historical memory"
    for decision in (source, target):
        object.__setattr__(
            decision,
            "resolution_method",
            ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
        )
        object.__setattr__(decision, "reason", reason)
    object.__setattr__(
        target,
        "status",
        ReconciliationStatus.SUPERSEDED,
    )
    request = reconciliation_rules.RelationRequest.supersedes(
        source.fact_id,
        target.fact_id,
        reason,
        tuple((*source.candidate_ids, *target.candidate_ids)),
        tuple((*source.evidence_refs, *target.evidence_refs)),
        method=ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
    )

    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome(
            decisions,
            (request,),
        )


def test_predicate_outcome_rejects_mixed_predicate_decisions() -> None:
    port = make_current_candidate(
        candidate_id="port",
        predicate="port",
        value=8000,
    )
    host = make_current_candidate(
        candidate_id="host",
        predicate="host",
        value="localhost",
    )
    decisions = tuple(
        sorted(
            (*decisions_for(port), *decisions_for(host)),
            key=lambda decision: decision.fact_id,
        )
    )

    with pytest.raises(
        ReconciliationInputError,
        match="one exact subject and predicate",
    ):
        PredicateOutcome(decisions, ())


def test_predicate_outcome_rejects_fabricated_fact_identity() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value=8000)
    )[0]
    object.__setattr__(decision, "fact_id", "fact:v1:fabricated")

    with pytest.raises(
        ReconciliationInputError,
        match="canonical FactDecision identity",
    ):
        PredicateOutcome((decision,), ())


def test_three_node_explicit_cycle_is_deferred_to_task_19() -> None:
    first = make_current_candidate(
        candidate_id="first",
        value=8000,
        supersedes=known(("second",)),
    )
    second = make_current_candidate(
        candidate_id="second",
        value=8010,
        supersedes=known(("third",)),
    )
    third = make_current_candidate(
        candidate_id="third",
        value=8020,
        supersedes=known(("first",)),
    )

    outcome = resolve_predicate(
        decisions_for(first, second, third),
        default_policy(),
    )

    assert len(outcome.relation_requests) == 3
    assert all(
        request.relation_type is RelationType.SUPERSEDES
        and request.method is ResolutionMethod.EXPLICIT_SUPERSEDES
        for request in outcome.relation_requests
    )
    assert set(outcome.statuses.values()) == {
        ReconciliationStatus.SUPERSEDED
    }
    assert not hasattr(reconciliation_rules, "ReconciledFact")
    assert not hasattr(reconciliation_rules, "ReconciliationResult")
    assert not hasattr(reconciliation_rules, "reconcile")
