from dataclasses import FrozenInstanceError

import pytest

from agent_memory_os.reconcile import rules as reconciliation_rules
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    LineageOutcome,
    ReconciliationInputError,
    ReconciliationStatus,
    ResolutionMethod,
    SameValueLineage,
    SourceType,
    WarningCode,
)
from agent_memory_os.reconcile.rules import (
    classify_group,
    group_candidates,
    make_fact_id,
    resolve_same_value_lineage,
)

from reconciliation_helpers import (
    NOW,
    default_policy,
    iso_after,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    make_source_of_truth_candidate,
    single_group,
    temporal_map,
)


@pytest.mark.parametrize(
    ("status_hint", "deprecated"),
    [
        (CandidateStatusHint.HISTORICAL, True),
        (CandidateStatusHint.DEPRECATED, False),
        (CandidateStatusHint.DEPRECATED, True),
    ],
    ids=("deprecated-flag", "deprecated-hint", "both-markers"),
)
def test_standalone_explicit_deprecation_is_classified_deprecated(
    status_hint: CandidateStatusHint,
    deprecated: bool,
) -> None:
    candidate = make_candidate(
        candidate_id="candidate",
        value="v1-only",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(status_hint),
        deprecated=known(deprecated),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ()
    assert decision.relation_requests == ()


def test_boolean_deprecation_overrides_same_candidate_current_hint() -> None:
    candidate = make_candidate(
        candidate_id="candidate",
        value="v1-only",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.CURRENT_FACT),
        deprecated=known(True),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ()
    assert decision.relation_requests == ()


def test_historical_corroboration_does_not_override_boolean_deprecation_of_current_hint(
) -> None:
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.CURRENT_FACT),
        deprecated=known(True),
        confidence=known(0.9),
    )
    historical = make_historical_candidate(
        candidate_id="historical",
        value="v1",
        confidence=known(0.99),
    )

    decision = classify_group(
        single_group(deprecated, historical),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ()
    assert decision.relation_requests == ()


def test_historical_corroboration_cannot_reactivate_explicit_deprecation() -> None:
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        confidence=known(0.9),
    )
    historical = make_historical_candidate(
        candidate_id="historical",
        value="v1",
        confidence=known(0.99),
    )

    decision = classify_group(
        single_group(deprecated, historical),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.activation_witness_candidate_ids == ()
    assert decision.relation_requests == ()


def test_later_explicit_deprecation_supersedes_same_value_current_candidate(
) -> None:
    current = make_current_candidate(
        candidate_id="current",
        value="v1",
        confidence=known(0.9),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        supersedes=known(("current",)),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(current, deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.superseded_candidate_ids == ("current",)
    assert decision.activation_witness_candidate_ids == ()
    assert decision.relation_requests == ()


def test_explicit_same_value_reactivation_precedes_deprecation() -> None:
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
    )
    reactivated = make_current_candidate(
        candidate_id="reactivated",
        value="v1",
        supersedes=known(("deprecated",)),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(deprecated, reactivated),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.SAME_VALUE_REACTIVATION
    assert decision.activation_witness_candidate_ids == ("reactivated",)
    assert decision.superseded_candidate_ids == ("deprecated",)
    assert decision.relation_requests == ()


def test_unresolved_current_and_deprecated_tips_remain_pending_for_review(
) -> None:
    current = make_current_candidate(
        candidate_id="current",
        value="v1",
        confidence=known(0.9),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
    )

    decision = classify_group(
        single_group(current, deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == "same-value candidate lineage is unresolved"
    assert decision.requires_human_review is True
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ()
    assert decision.relation_requests == ()


def test_explicit_deprecation_precedes_temporal_pending() -> None:
    deprecated = make_candidate(
        candidate_id="future-deprecation",
        value="v1",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        valid_from=known(iso_after(NOW, 1)),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()
    assert decision.relation_requests == ()


@pytest.mark.parametrize(
    "candidate",
    [
        make_candidate(
            candidate_id="future-skew",
            value="v1",
            source_type=SourceType.PROJECT_DOC,
            status_hint=known(CandidateStatusHint.DEPRECATED),
            deprecated=known(True),
            observed_at=known(iso_after(NOW, 301)),
            confidence=known(0.9),
        ),
        make_candidate(
            candidate_id="invalid-order",
            value="v1",
            source_type=SourceType.PROJECT_DOC,
            status_hint=known(CandidateStatusHint.DEPRECATED),
            deprecated=known(True),
            valid_from=known(iso_after(NOW, 10)),
            valid_until=known(iso_after(NOW, 5)),
            confidence=known(0.9),
        ),
    ],
    ids=("future-skew", "invalid-order"),
)
def test_explicit_deprecation_retains_temporal_review_requirement(
    candidate: object,
) -> None:
    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is True
    assert any(
        warning.requires_human_review for warning in decision.warnings
    )
    assert decision.activation_witness_candidate_ids == ()
    assert decision.relation_requests == ()


def test_explicit_deprecation_precedes_insufficient_evidence() -> None:
    current = make_current_candidate(
        candidate_id="low-confidence-current",
        value="v1",
        confidence=known(0.49),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        supersedes=known(("low-confidence-current",)),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(current, deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.DEPRECATED
    assert decision.resolution_method is ResolutionMethod.EXPLICIT_DEPRECATION
    assert decision.reason == "explicit deprecation"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ("low-confidence-current",)
    assert decision.relation_requests == ()


def test_deprecated_value_can_be_explicitly_reactivated_without_fact_edge() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    group = single_group(old, new)
    lineage = resolve_same_value_lineage(group, NOW)

    assert lineage.outcome is LineageOutcome.REACTIVATED
    assert lineage.superseded_candidate_ids == ("old",)
    assert lineage.fact_relation_requests == ()
    assert group.fact_id == make_fact_id(
        "synthetic-project",
        "project",
        "setting",
        "local",
    )
    assert group.candidate_ids == ("new", "old")


def test_unresolved_same_value_current_and_deprecated_requires_review() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    current = make_current_candidate(
        candidate_id="current",
        value="local",
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(old, current), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.warnings == ()


def test_explicit_same_value_lineage_can_deprecate_current_candidate() -> None:
    current = make_current_candidate(candidate_id="current", value="local")
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        supersedes=known(("current",)),
    )

    lineage = resolve_same_value_lineage(
        single_group(deprecated, current),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.DEPRECATED
    assert lineage.surviving_candidate_ids == ("deprecated",)
    assert lineage.superseded_candidate_ids == ("current",)
    assert lineage.fact_relation_requests == ()


@pytest.mark.parametrize("gap_seconds", [0, 1])
def test_interval_lineage_reactivates_at_or_after_deprecated_end(
    gap_seconds: int,
) -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=known(iso_after(NOW, -20)),
        valid_until=known(iso_after(NOW, -10)),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        valid_from=known(iso_after(NOW, -10 + gap_seconds)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(new, old), NOW)

    assert lineage.outcome is LineageOutcome.REACTIVATED
    assert lineage.surviving_candidate_ids == ("new",)
    assert lineage.superseded_candidate_ids == ("old",)


def test_future_interval_lineage_cannot_reactivate_deprecated_value() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_until=known(NOW.isoformat()),
    )
    future = make_current_candidate(
        candidate_id="future",
        value="local",
        valid_from=known(iso_after(NOW, 1)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(old, future), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_invalid_deprecated_interval_cannot_reactivate_lineage() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=known(iso_after(NOW, -5), source="old:valid-from"),
        valid_until=known(iso_after(NOW, -10), source="old:valid-until"),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        valid_from=known(iso_after(NOW, -9), source="new:valid-from"),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(new, old), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()
    invalid_order = next(
        warning
        for warning in lineage.warnings
        if warning.code is WarningCode.INVALID_TEMPORAL_ORDER
    )
    assert invalid_order.candidate_ids == ("old",)
    assert invalid_order.evidence_refs == ("old:valid-from", "old:valid-until")
    assert invalid_order.requires_human_review is True


def test_observed_recency_alone_never_reactivates_same_value_lineage() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        observed_at=known(iso_after(NOW, -100)),
    )
    newer = make_current_candidate(
        candidate_id="newer",
        value="local",
        observed_at=known(NOW.isoformat()),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(newer, old), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.superseded_candidate_ids == ()


def test_current_tip_must_be_activation_capable_to_reactivate_lineage() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    low_confidence = make_current_candidate(
        candidate_id="low",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.49),
    )

    lineage = resolve_same_value_lineage(
        single_group(old, low_confidence),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_wrong_source_type_blocks_source_of_truth_reactivation() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    invalid = make_candidate(
        candidate_id="invalid",
        value="local",
        source_type=SourceType.PROJECT_DOC,
        status_hint=known(CandidateStatusHint.SOURCE_OF_TRUTH),
        explicit_user_instruction=known(True),
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(invalid, old), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_nonexplicit_source_of_truth_blocks_reactivation() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    invalid = make_candidate(
        candidate_id="invalid",
        value="local",
        source_type=SourceType.USER_EXPLICIT,
        status_hint=known(CandidateStatusHint.SOURCE_OF_TRUTH),
        explicit_user_instruction=known(False),
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(old, invalid), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_disabled_policy_blocks_source_of_truth_reactivation() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    source_of_truth = make_source_of_truth_candidate(
        candidate_id="source-of-truth",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(
        single_group(old, source_of_truth),
        NOW,
        default_policy(allow_explicit_source_of_truth_override=False),
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_valid_source_of_truth_reactivation_uses_default_policy() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    source_of_truth = make_source_of_truth_candidate(
        candidate_id="source-of-truth",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(
        single_group(source_of_truth, old),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.REACTIVATED
    assert lineage.requires_human_review is False
    assert lineage.superseded_candidate_ids == ("old",)
    assert lineage.fact_relation_requests == ()


def test_historical_tip_cannot_reactivate_deprecated_lineage() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    historical = make_historical_candidate(
        candidate_id="historical",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(
        single_group(historical, old),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_neutral_descendant_of_current_with_current_tip_is_unresolved() -> None:
    ancestor = make_current_candidate(candidate_id="ancestor", value="local")
    neutral = make_historical_candidate(
        candidate_id="neutral",
        value="local",
        supersedes=known(("ancestor",)),
    )
    current = make_current_candidate(candidate_id="current", value="local")

    lineage = resolve_same_value_lineage(
        single_group(neutral, current, ancestor),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.surviving_candidate_ids == ("current", "neutral")
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_neutral_descendant_of_deprecated_with_deprecated_tip_is_unresolved() -> None:
    ancestor = make_candidate(
        candidate_id="ancestor",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    neutral = make_historical_candidate(
        candidate_id="neutral",
        value="local",
        supersedes=known(("ancestor",)),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )

    lineage = resolve_same_value_lineage(
        single_group(neutral, deprecated, ancestor),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.surviving_candidate_ids == ("deprecated", "neutral")
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()


def test_transitive_same_value_lineage_reports_every_replaced_ancestor() -> None:
    oldest = make_candidate(
        candidate_id="a-oldest",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    middle = make_candidate(
        candidate_id="b-middle",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        supersedes=known(("a-oldest",)),
    )
    newest = make_current_candidate(
        candidate_id="c-newest",
        value="local",
        supersedes=known(("b-middle",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(
        single_group(newest, oldest, middle),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.REACTIVATED
    assert lineage.surviving_candidate_ids == ("c-newest",)
    assert lineage.superseded_candidate_ids == ("a-oldest", "b-middle")


def test_cyclic_same_value_lineage_fails_closed_with_warning() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        supersedes=known(("new",)),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(old, new), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()
    assert lineage.fact_relation_requests == ()
    assert tuple(warning.code for warning in lineage.warnings) == (
        WarningCode.CONTRADICTORY_SUPERSEDES,
    )
    assert lineage.warnings[0].candidate_ids == ("new", "old")
    with pytest.raises(FrozenInstanceError):
        lineage.warnings[0].requires_human_review = False


def test_mixed_explicit_interval_cycle_reports_exact_edge_provenance() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=known(iso_after(NOW, -20), source="old:valid-from"),
        valid_until=known(iso_after(NOW, -10), source="old:valid-until"),
        supersedes=known(("new",), source="old:supersedes"),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        valid_from=known(iso_after(NOW, -10), source="new:valid-from"),
        confidence=known(0.9),
    )

    lineage = resolve_same_value_lineage(single_group(new, old), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()
    assert tuple(warning.code for warning in lineage.warnings) == (
        WarningCode.CONTRADICTORY_SUPERSEDES,
    )
    assert lineage.warnings[0].candidate_ids == ("new", "old")
    assert lineage.warnings[0].evidence_refs == (
        "new:valid-from",
        "old:supersedes",
        "old:valid-until",
    )


def test_lineage_cycle_warning_excludes_noncyclic_descendants() -> None:
    first = make_candidate(
        candidate_id="a",
        value="local",
        supersedes=known(("b",)),
    )
    second = make_candidate(
        candidate_id="b",
        value="local",
        supersedes=known(("a",)),
    )
    descendant = make_current_candidate(
        candidate_id="c",
        value="local",
        supersedes=known(("b",)),
    )

    lineage = resolve_same_value_lineage(
        single_group(descendant, second, first),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.warnings[0].candidate_ids == ("a", "b")


def test_cycle_analysis_visits_chain_adjacency_linearly() -> None:
    adjacency_visits = [0]

    class CountingSet(set[str]):
        def __iter__(self):
            adjacency_visits[0] += 1
            return super().__iter__()

    node_count = 40
    successors = {
        str(index): CountingSet(
            () if index == node_count - 1 else (str(index + 1),)
        )
        for index in range(node_count)
    }

    assert reconciliation_rules._cycle_nodes(successors) == ()
    assert adjacency_visits[0] <= node_count * 2


def test_long_lineage_does_not_materialize_all_ancestor_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_entries = [0]
    original_frozenset = frozenset

    def counting_frozenset(items=()):
        copied = tuple(items)
        frozen_entries[0] += len(copied)
        return original_frozenset(copied)

    monkeypatch.setattr(
        reconciliation_rules,
        "frozenset",
        counting_frozenset,
        raising=False,
    )
    node_count = 2_000
    candidates = []
    for index in range(node_count):
        candidate_id = f"node-{index:04d}"
        fields = {
            "candidate_id": candidate_id,
            "value": "local",
            "supersedes": (() if index == 0 else (f"node-{index - 1:04d}",)),
        }
        candidate = (
            make_current_candidate(**fields)
            if index == node_count - 1
            else make_historical_candidate(**fields)
        )
        candidates.append(candidate)

    lineage = resolve_same_value_lineage(single_group(*candidates), NOW)

    assert lineage.outcome is LineageOutcome.CURRENT
    assert len(lineage.superseded_candidate_ids) == node_count - 1
    assert frozen_entries[0] <= node_count * 4


def test_many_cycle_components_scan_edge_provenance_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge_visits = [0]
    original_build = reconciliation_rules._build_same_value_graph

    class CountingEdges(dict):
        def items(self):
            edge_visits[0] += len(self)
            return super().items()

    def counting_build(*args, **kwargs):
        graph, warning = original_build(*args, **kwargs)
        graph = reconciliation_rules._LineageGraph(
            graph.successors,
            CountingEdges(graph.edges),
        )
        return graph, warning

    monkeypatch.setattr(
        reconciliation_rules,
        "_build_same_value_graph",
        counting_build,
    )
    component_count = 100
    candidates = []
    for index in range(component_count):
        left_id = f"left-{index:03d}"
        right_id = f"right-{index:03d}"
        candidates.extend(
            (
                make_historical_candidate(
                    candidate_id=left_id,
                    value="local",
                    supersedes=known((right_id,)),
                ),
                make_historical_candidate(
                    candidate_id=right_id,
                    value="local",
                    supersedes=known((left_id,)),
                ),
            )
        )

    lineage = resolve_same_value_lineage(single_group(*candidates), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert len(lineage.warnings[0].candidate_ids) == component_count * 2
    assert edge_visits[0] <= component_count * 4


def test_incompatible_surviving_lineage_tips_do_not_report_a_winner() -> None:
    ancestor = make_historical_candidate(
        candidate_id="ancestor",
        value="local",
    )
    current = make_current_candidate(
        candidate_id="current",
        value="local",
        supersedes=known(("ancestor",)),
        confidence=known(0.9),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        supersedes=known(("ancestor",)),
    )

    lineage = resolve_same_value_lineage(
        single_group(deprecated, ancestor, current),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.surviving_candidate_ids == ("current", "deprecated")
    assert lineage.superseded_candidate_ids == ()
    assert lineage.requires_human_review is True
    assert tuple(warning.code for warning in lineage.warnings) == (
        WarningCode.CONTRADICTORY_SUPERSEDES,
    )
    assert lineage.warnings[0].candidate_ids == (
        "ancestor",
        "current",
        "deprecated",
    )


def test_transitive_explicit_fork_emits_contradictory_lineage_warning() -> None:
    ancestor = make_historical_candidate(candidate_id="a", value="local")
    branch = make_current_candidate(
        candidate_id="b",
        value="local",
        supersedes=known(("a",)),
    )
    current_tip = make_current_candidate(
        candidate_id="c",
        value="local",
        supersedes=known(("b",)),
    )
    deprecated_tip = make_candidate(
        candidate_id="d",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        supersedes=known(("a",)),
    )

    lineage = resolve_same_value_lineage(
        single_group(deprecated_tip, current_tip, ancestor, branch),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.surviving_candidate_ids == ("c", "d")
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()
    assert tuple(warning.code for warning in lineage.warnings) == (
        WarningCode.CONTRADICTORY_SUPERSEDES,
    )
    assert lineage.warnings[0].candidate_ids == ("a", "b", "c", "d")


def test_pure_current_and_historical_lineages_are_not_reactivations() -> None:
    current = resolve_same_value_lineage(
        single_group(
            make_current_candidate(candidate_id="b", value="local"),
            make_current_candidate(candidate_id="a", value="local"),
        ),
        NOW,
    )
    historical = resolve_same_value_lineage(
        single_group(
            make_historical_candidate(candidate_id="d", value="local"),
            make_historical_candidate(candidate_id="c", value="local"),
        ),
        NOW,
    )

    assert current.outcome is LineageOutcome.CURRENT
    assert current.surviving_candidate_ids == ("a", "b")
    assert historical.outcome is LineageOutcome.NEUTRAL
    assert historical.surviving_candidate_ids == ("c", "d")
    assert current.superseded_candidate_ids == ()
    assert historical.superseded_candidate_ids == ()


def test_self_referential_local_lineage_fails_closed() -> None:
    candidate = make_current_candidate(
        candidate_id="self",
        value="local",
        supersedes=known(("self",)),
    )

    lineage = resolve_same_value_lineage(single_group(candidate), NOW)

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert lineage.requires_human_review is True
    assert lineage.superseded_candidate_ids == ()
    assert lineage.fact_relation_requests == ()
    assert tuple(warning.code for warning in lineage.warnings) == (
        WarningCode.CONTRADICTORY_SUPERSEDES,
    )


def test_cross_value_reference_does_not_poison_same_value_lineage() -> None:
    old = make_candidate(candidate_id="old", value=1)
    new = make_current_candidate(
        candidate_id="new",
        value=2,
        supersedes=known(("old",)),
    )
    groups = group_candidates("synthetic-project", (new, old))
    new_group = next(group for group in groups if group.candidate_ids == ("new",))

    lineage = resolve_same_value_lineage(new_group, NOW)

    assert lineage.outcome is LineageOutcome.CURRENT
    assert lineage.surviving_candidate_ids == ("new",)
    assert lineage.superseded_candidate_ids == ()
    assert lineage.requires_human_review is False
    assert lineage.warnings == ()
    assert lineage.fact_relation_requests == ()


def test_same_value_lineage_output_is_deterministic() -> None:
    old = make_candidate(
        candidate_id="z-old",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
    )
    new = make_current_candidate(
        candidate_id="a-new",
        value="local",
        supersedes=known(("z-old",)),
        confidence=known(0.9),
    )

    first = resolve_same_value_lineage(single_group(old, new), NOW)
    second = resolve_same_value_lineage(single_group(new, old), NOW)

    assert first == second
    assert first.surviving_candidate_ids == ("a-new",)
    assert first.superseded_candidate_ids == ("z-old",)


def test_lineage_model_rejects_surviving_superseded_overlap() -> None:
    with pytest.raises(ReconciliationInputError, match="overlap"):
        SameValueLineage(
            outcome=LineageOutcome.CURRENT,
            surviving_candidate_ids=("candidate",),
            superseded_candidate_ids=("candidate",),
            requires_human_review=False,
        )


def test_lineage_model_requires_survivor_for_resolved_outcome() -> None:
    with pytest.raises(ReconciliationInputError, match="surviving"):
        SameValueLineage(
            outcome=LineageOutcome.CURRENT,
            surviving_candidate_ids=(),
            superseded_candidate_ids=(),
            requires_human_review=False,
        )


def test_lineage_model_requires_review_for_unresolved_outcome() -> None:
    with pytest.raises(ReconciliationInputError, match="human review"):
        SameValueLineage(
            outcome=LineageOutcome.UNRESOLVED,
            surviving_candidate_ids=("candidate",),
            superseded_candidate_ids=(),
            requires_human_review=False,
        )


def test_sparse_interval_builder_reuses_parsed_temporal_assessments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = [0]
    original_parse = reconciliation_rules.parse_known_timestamp

    def counting_parse(evidence, field_name):
        parse_calls[0] += 1
        return original_parse(evidence, field_name)

    monkeypatch.setattr(
        reconciliation_rules,
        "parse_known_timestamp",
        counting_parse,
    )
    pair_count = 100
    deprecated = tuple(
        make_candidate(
            candidate_id=f"deprecated-{index:03d}",
            value="local",
            deprecated=known(True),
            status_hint=known(CandidateStatusHint.DEPRECATED),
            valid_from=NOW.isoformat(),
            valid_until=iso_after(NOW, 10_000 + index),
        )
        for index in range(pair_count)
    )
    current = tuple(
        make_current_candidate(
            candidate_id=f"current-{index:03d}",
            value="local",
            valid_from=iso_after(NOW, -10_000 + index),
        )
        for index in range(pair_count)
    )

    lineage = resolve_same_value_lineage(
        single_group(*deprecated, *current),
        NOW,
    )

    assert lineage.outcome is LineageOutcome.UNRESOLVED
    assert parse_calls[0] <= (len(deprecated) + len(current)) * 3


def test_interval_sweep_preserves_equality_and_multiple_legal_edges() -> None:
    old_early = make_candidate(
        candidate_id="old-early",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=iso_after(NOW, -30),
        valid_until=iso_after(NOW, -20),
    )
    old_equal = make_candidate(
        candidate_id="old-equal",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=iso_after(NOW, -20),
        valid_until=iso_after(NOW, -10),
    )
    old_future = make_candidate(
        candidate_id="old-future",
        value="local",
        deprecated=known(True),
        status_hint=known(CandidateStatusHint.DEPRECATED),
        valid_from=NOW.isoformat(),
        valid_until=iso_after(NOW, 10),
    )
    new_early = make_current_candidate(
        candidate_id="new-early",
        value="local",
        valid_from=iso_after(NOW, -15),
    )
    new_equal = make_current_candidate(
        candidate_id="new-equal",
        value="local",
        valid_from=iso_after(NOW, -10),
    )
    group = single_group(
        old_early,
        old_equal,
        old_future,
        new_early,
        new_equal,
    )

    graph, warning = reconciliation_rules._build_same_value_graph(
        group,
        NOW,
        temporal_map(group, NOW),
        default_policy(),
    )

    assert warning is None
    assert tuple(graph.edges) == (
        ("old-early", "new-early"),
        ("old-early", "new-equal"),
        ("old-equal", "new-equal"),
    )
    assert graph.edges[("old-equal", "new-equal")].interval_evidence_refs == (
        "synthetic:new-equal:from",
        "synthetic:old-equal:until",
    )
