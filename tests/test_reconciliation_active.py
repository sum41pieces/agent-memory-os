from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json

import pytest

from agent_memory_os.reconcile import models as reconciliation_models
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    FactDecision,
    ReconciliationStatus,
    ReconciliationInputError,
    ResolutionMethod,
    SourceType,
)
from agent_memory_os.reconcile import rules
from agent_memory_os.reconcile.rules import (
    CandidateGroup,
    assess_temporal,
    canonical_typed_value,
    classify_group,
    find_activation_witnesses,
    group_candidates,
    is_activation_witness,
    make_fact_id,
)

from reconciliation_helpers import (
    NOW,
    default_policy,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    make_source_of_truth_candidate,
    single_group,
    temporal_map,
    unavailable,
    unknown,
)


def test_activation_witness_functions_are_formally_exported() -> None:
    assert "is_activation_witness" in rules.__all__
    assert "find_activation_witnesses" in rules.__all__
    assert "classify_group" in rules.__all__


def _unresolved_id(project_id, candidate) -> str:
    identity = {
        "namespace": "unresolved:v1",
        "project_id": project_id,
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "candidate_id": candidate.candidate_id,
        "evidence_status": candidate.value.status.value,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"unresolved:v1:{hashlib.sha256(encoded).hexdigest()}"


def test_historical_confidence_cannot_rescue_low_confidence_current_candidate() -> None:
    current = make_current_candidate(
        candidate_id="current",
        value=8000,
        confidence=known(0.2),
    )
    historical = make_historical_candidate(
        candidate_id="history",
        value=8000,
        confidence=known(0.9),
    )
    group = single_group(current, historical)

    witnesses = find_activation_witnesses(
        group,
        temporal_map(group, NOW),
        default_policy(),
    )

    assert witnesses == ()


def test_valid_direct_current_candidate_is_an_activation_witness() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        confidence=known(0.5),
    )
    group = single_group(candidate)

    assert is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(),
    )


def test_valid_explicit_source_of_truth_candidate_is_an_activation_witness() -> None:
    candidate = make_source_of_truth_candidate(
        candidate_id="source-of-truth",
        confidence=known(0.5),
    )
    group = single_group(candidate)

    assert is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_type": SourceType.CURRENT_EVIDENCE},
        {"explicit_user_instruction": known(False)},
        {"explicit_user_instruction": unknown("not established")},
        {"status_hint": CandidateStatusHint.HISTORICAL},
    ],
)
def test_source_of_truth_requires_exact_explicit_designation(changes) -> None:
    candidate = make_source_of_truth_candidate(
        candidate_id="invalid-source-of-truth",
        **changes,
    )
    group = single_group(candidate)

    assert not is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(),
    )


def test_source_of_truth_override_can_be_disabled_by_policy() -> None:
    candidate = make_source_of_truth_candidate(candidate_id="source-of-truth")
    group = single_group(candidate)

    assert not is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(allow_explicit_source_of_truth_override=False),
    )


@pytest.mark.parametrize(
    "value",
    [unknown("missing"), unavailable("offline")],
)
def test_nonknown_value_is_never_an_activation_witness(value) -> None:
    candidate = make_current_candidate(candidate_id="unresolved", value=value)
    group = single_group(candidate)

    assert not is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(),
    )


def test_temporally_ineligible_candidate_is_not_an_activation_witness() -> None:
    candidate = make_current_candidate(
        candidate_id="future",
        valid_from="2026-09-14T00:00:01+00:00",
    )
    group = single_group(candidate)
    assessment = temporal_map(group, NOW)[candidate.candidate_id]

    assert assessment.eligible is False
    assert not is_activation_witness(candidate, assessment, default_policy())


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (unknown("not measured"), False),
        (unavailable("not reported"), False),
        (known(0.499999), False),
        (known(0.5), True),
        (known(1.0), True),
    ],
)
def test_activation_witness_confidence_requires_known_threshold_or_above(
    confidence,
    expected,
) -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        confidence=confidence,
    )
    group = single_group(candidate)

    assert (
        is_activation_witness(
            candidate,
            temporal_map(group, NOW)[candidate.candidate_id],
            default_policy(),
        )
        is expected
    )


@pytest.mark.parametrize(
    "status_hint",
    [CandidateStatusHint.PLAN, CandidateStatusHint.HISTORICAL],
)
def test_plan_and_historical_candidates_are_never_activation_witnesses(
    status_hint,
) -> None:
    candidate = make_candidate(
        candidate_id="noncurrent",
        status_hint=status_hint,
        confidence=known(1.0),
    )
    group = single_group(candidate)

    assert not is_activation_witness(
        candidate,
        temporal_map(group, NOW)[candidate.candidate_id],
        default_policy(),
    )


def test_find_activation_witnesses_returns_candidates_in_candidate_id_order() -> None:
    last = make_current_candidate(candidate_id="z-last", value="same")
    historical = make_historical_candidate(candidate_id="m-history", value="same")
    first = make_source_of_truth_candidate(candidate_id="a-first", value="same")
    group = single_group(last, historical, first)

    witnesses = find_activation_witnesses(
        group,
        temporal_map(group, NOW),
        default_policy(),
    )

    assert tuple(candidate.candidate_id for candidate in witnesses) == (
        "a-first",
        "z-last",
    )


@pytest.mark.parametrize(
    ("assessments", "message"),
    [
        ([], "mapping"),
        ({}, "exactly"),
        ({"current": object()}, "TemporalAssessment"),
    ],
)
def test_find_activation_witnesses_validates_assessment_mapping(
    assessments,
    message,
) -> None:
    group = single_group(make_current_candidate(candidate_id="current"))

    with pytest.raises(ReconciliationInputError, match=message):
        find_activation_witnesses(group, assessments, default_policy())


def test_find_activation_witnesses_rejects_extra_assessment_keys() -> None:
    group = single_group(make_current_candidate(candidate_id="current"))
    assessments = temporal_map(group, NOW)
    assessments["extra"] = assessments["current"]

    with pytest.raises(ReconciliationInputError, match="exactly"):
        find_activation_witnesses(group, assessments, default_policy())


def test_find_activation_witnesses_rejects_assessments_swapped_between_candidates() -> None:
    future = make_current_candidate(
        candidate_id="future-current",
        value="same",
        valid_from="2026-09-14T00:00:01+00:00",
    )
    historical = make_historical_candidate(
        candidate_id="eligible-history",
        value="same",
    )
    group = single_group(future, historical)
    assessments = temporal_map(group, NOW)
    swapped = {
        future.candidate_id: assessments[historical.candidate_id],
        historical.candidate_id: assessments[future.candidate_id],
    }

    with pytest.raises(ReconciliationInputError, match="assessment"):
        find_activation_witnesses(group, swapped, default_policy())


def test_find_activation_witnesses_rejects_fabricated_eligible_assessment() -> None:
    future = make_current_candidate(
        candidate_id="future-current",
        valid_from="2026-09-14T00:00:01+00:00",
    )
    group = single_group(future)
    assessment = temporal_map(group, NOW)[future.candidate_id]
    fabricated = replace(
        assessment,
        eligible=True,
        pending_reason=None,
        warnings=(),
    )

    with pytest.raises(ReconciliationInputError, match="assessment"):
        find_activation_witnesses(
            group,
            {future.candidate_id: fabricated},
            default_policy(),
        )


def test_find_activation_witnesses_rejects_temporal_assessment_subclass() -> None:
    future = make_current_candidate(
        candidate_id="future-current",
        valid_from="2026-09-14T00:00:01+00:00",
    )
    group = single_group(future)
    assessment = temporal_map(group, NOW)[future.candidate_id]

    class ForgedAssessment(type(assessment)):
        def __eq__(self, other):
            return True

    forged = ForgedAssessment(
        candidate_id=assessment.candidate_id,
        assessed_at=assessment.assessed_at,
        max_future_clock_skew_seconds=(
            assessment.max_future_clock_skew_seconds
        ),
        observed_at=assessment.observed_at,
        valid_from=assessment.valid_from,
        valid_until=assessment.valid_until,
        eligible=True,
        pending_reason=None,
        warnings=(),
    )

    with pytest.raises(ReconciliationInputError, match="TemporalAssessment"):
        find_activation_witnesses(
            group,
            {future.candidate_id: forged},
            default_policy(),
        )


def test_find_activation_witnesses_snapshots_stateful_assessment_mapping_once() -> None:
    future = make_current_candidate(
        candidate_id="future-current",
        valid_from="2026-09-14T00:00:01+00:00",
    )
    group = single_group(future)
    canonical = temporal_map(group, NOW)[future.candidate_id]
    forged = replace(
        canonical,
        eligible=True,
        pending_reason=None,
        warnings=(),
    )

    class StatefulAssessments(Mapping):
        def __init__(self):
            self.reads = 0

        def __getitem__(self, key):
            if key != future.candidate_id:
                raise KeyError(key)
            self.reads += 1
            return canonical if self.reads == 1 else forged

        def __iter__(self):
            return iter((future.candidate_id,))

        def __len__(self):
            return 1

    assessments = StatefulAssessments()

    assert find_activation_witnesses(
        group,
        assessments,
        default_policy(),
    ) == ()
    assert assessments.reads == 1


def test_find_activation_witnesses_rejects_assessment_from_different_policy() -> None:
    current = make_current_candidate(
        candidate_id="current",
        observed_at="2026-09-14T00:00:01+00:00",
    )
    group = single_group(current)
    assessments = temporal_map(group, NOW)

    with pytest.raises(ReconciliationInputError, match="assessment"):
        find_activation_witnesses(
            group,
            assessments,
            default_policy(max_future_clock_skew_seconds=0),
        )


def test_find_activation_witnesses_rejects_mixed_assessment_clocks() -> None:
    first = make_current_candidate(candidate_id="first", value="same")
    second = make_current_candidate(candidate_id="second", value="same")
    group = single_group(first, second)
    policy = default_policy()
    assessments = {
        first.candidate_id: assess_temporal(first, NOW, policy),
        second.candidate_id: assess_temporal(
            second,
            NOW.replace(second=1),
            policy,
        ),
    }

    with pytest.raises(ReconciliationInputError, match="clock"):
        find_activation_witnesses(group, assessments, policy)


def test_temporal_map_uses_the_supplied_policy() -> None:
    current = make_current_candidate(
        candidate_id="current",
        observed_at="2026-09-14T00:00:01+00:00",
    )
    group = single_group(current)
    policy = default_policy(max_future_clock_skew_seconds=0)

    assessment = temporal_map(group, NOW, policy)[current.candidate_id]

    assert assessment.eligible is False


def test_find_activation_witnesses_validates_group_and_policy_types() -> None:
    group = single_group(make_current_candidate(candidate_id="current"))
    assessments = temporal_map(group, NOW)

    with pytest.raises(ReconciliationInputError, match="group"):
        find_activation_witnesses(object(), assessments, default_policy())
    with pytest.raises(ReconciliationInputError, match="policy"):
        find_activation_witnesses(group, assessments, object())


@pytest.mark.parametrize(
    ("candidate", "temporal", "message"),
    [
        (object(), object(), "candidate"),
        (make_current_candidate(), object(), "temporal"),
    ],
)
def test_is_activation_witness_validates_argument_types(
    candidate,
    temporal,
    message,
) -> None:
    with pytest.raises(ReconciliationInputError, match=message):
        is_activation_witness(candidate, temporal, default_policy())


def test_is_activation_witness_validates_policy_type() -> None:
    candidate = make_current_candidate(candidate_id="current")
    group = single_group(candidate)

    with pytest.raises(ReconciliationInputError, match="policy"):
        is_activation_witness(
            candidate,
            temporal_map(group, NOW)[candidate.candidate_id],
            object(),
        )


def test_group_same_typed_values_merges_in_candidate_id_order() -> None:
    candidates = [
        make_candidate(
            candidate_id="z",
            value="main",
            source_type=SourceType.HISTORICAL_MEMORY,
        ),
        make_candidate(
            candidate_id="a",
            value="main",
            source_type=SourceType.CURRENT_EVIDENCE,
        ),
    ]

    forward = group_candidates("project", candidates)
    reverse = group_candidates("project", list(reversed(candidates)))

    assert forward == reverse
    assert len(forward) == 1
    assert forward[0].candidate_ids == ("a", "z")


def test_group_unknown_values_do_not_form_a_fake_shared_value_group() -> None:
    candidates = [
        make_candidate(candidate_id="u1", value=unknown("missing")),
        make_candidate(candidate_id="u2", value=unavailable("unsupported")),
    ]

    groups = group_candidates("project", candidates)

    assert [group.candidate_ids for group in groups] == [("u1",), ("u2",)]


def test_group_sorting_covers_subject_predicate_and_canonical_value() -> None:
    candidates = [
        make_candidate(candidate_id="last", subject="z", predicate="a", value=0),
        make_candidate(candidate_id="value-z", subject="a", predicate="a", value="z"),
        make_candidate(candidate_id="predicate-z", subject="a", predicate="z", value=0),
        make_candidate(candidate_id="value-a", subject="a", predicate="a", value="a"),
    ]

    groups = group_candidates("project", list(reversed(candidates)))

    assert [
        (group.subject, group.predicate, group.canonical_value_key)
        for group in groups
    ] == [
        ("a", "a", canonical_typed_value("a")),
        ("a", "a", canonical_typed_value("z")),
        ("a", "z", canonical_typed_value(0)),
        ("z", "a", canonical_typed_value(0)),
    ]


def test_group_preserves_scalar_type_separation() -> None:
    candidates = [
        make_candidate(candidate_id="string", value="1"),
        make_candidate(candidate_id="integer", value=1),
        make_candidate(candidate_id="float", value=1.0),
        make_candidate(candidate_id="boolean", value=True),
    ]

    groups = group_candidates("project", candidates)

    assert len(groups) == 4
    assert {group.candidate_ids for group in groups} == {
        ("string",),
        ("integer",),
        ("float",),
        ("boolean",),
    }
    assert len({group.fact_id for group in groups}) == 4


def test_group_merges_nested_mappings_independent_of_key_order() -> None:
    left = {"z": {"second": 2, "first": [True, "x"]}, "a": 1}
    right = {"a": 1, "z": {"first": [True, "x"], "second": 2}}

    groups = group_candidates(
        "project",
        [
            make_candidate(candidate_id="left", value=left),
            make_candidate(candidate_id="right", value=right),
        ],
    )

    assert len(groups) == 1
    assert groups[0].candidate_ids == ("left", "right")


def test_group_preserves_nested_list_order_distinction() -> None:
    groups = group_candidates(
        "project",
        [
            make_candidate(candidate_id="ab", value={"items": ["a", "b"]}),
            make_candidate(candidate_id="ba", value={"items": ["b", "a"]}),
        ],
    )

    assert len(groups) == 2
    assert {group.candidate_ids for group in groups} == {("ab",), ("ba",)}


@pytest.mark.parametrize("reverse", [False, True])
def test_group_rejects_duplicate_candidate_ids_deterministically(reverse) -> None:
    candidates = [
        make_candidate(candidate_id="duplicate", value="first"),
        make_candidate(candidate_id="duplicate", value="second"),
    ]
    if reverse:
        candidates.reverse()

    with pytest.raises(
        ReconciliationInputError,
        match="duplicate candidate_id: duplicate",
    ):
        group_candidates("project", candidates)


def test_group_unresolved_ids_are_stable_and_separated_by_status_and_candidate_id() -> None:
    candidates = [
        make_candidate(candidate_id="same", value=unknown("missing")),
        make_candidate(candidate_id="other", value=unknown("missing")),
        make_candidate(candidate_id="same-status", value=unavailable("offline")),
    ]

    forward = group_candidates("project", candidates)
    reverse = group_candidates("project", list(reversed(candidates)))

    assert forward == reverse
    assert len({group.fact_id for group in forward}) == 3
    by_id = {group.candidate_ids[0]: group for group in forward}
    for candidate in candidates:
        assert by_id[candidate.candidate_id].fact_id == _unresolved_id(
            "project", candidate
        )


def test_group_copies_input_sequence_and_preserves_candidate_objects() -> None:
    first = make_candidate(candidate_id="z", value="main")
    second = make_candidate(candidate_id="a", value="main")
    candidates = [first, second]
    before = tuple(candidates)

    group = group_candidates("project", candidates)[0]

    assert tuple(candidates) == before
    assert candidates == [first, second]
    assert isinstance(group.candidates, tuple)
    assert group.candidates == (second, first)
    assert group.candidates[0] is second
    assert group.candidates[1] is first
    with pytest.raises(FrozenInstanceError):
        group.fact_id = "changed"


def test_group_computes_each_known_fact_id_exactly_once_and_stably(monkeypatch) -> None:
    original = rules.make_fact_id
    calls = []

    def counting_make_fact_id(project_id, subject, predicate, value):
        calls.append((project_id, subject, predicate, value))
        return original(project_id, subject, predicate, value)

    monkeypatch.setattr(rules, "make_fact_id", counting_make_fact_id)
    candidates = [
        make_candidate(candidate_id="a", value="main"),
        make_candidate(candidate_id="b", value="main"),
        make_candidate(candidate_id="c", value="other"),
        make_candidate(candidate_id="u", value=unknown("missing")),
    ]

    groups = group_candidates("project", candidates)

    assert len(calls) == 2
    known_groups = [
        group for group in groups if group.fact_id.startswith("fact:v1:")
    ]
    assert {group.fact_id for group in known_groups} == {
        original("project", "project", "setting", "main"),
        original("project", "project", "setting", "other"),
    }


@pytest.mark.parametrize("project_id", [None, 1, "", " ", "bad\x00project"])
def test_group_rejects_invalid_project_id(project_id) -> None:
    with pytest.raises(ReconciliationInputError, match="project_id"):
        group_candidates(project_id, [])


@pytest.mark.parametrize("field", ["subject", "predicate"])
def test_group_constructor_rejects_mixed_subject_or_predicate(field) -> None:
    first = make_candidate(candidate_id="a", value="main")
    changes = {field: "different"}
    second = make_candidate(candidate_id="b", value="main", **changes)

    with pytest.raises(TypeError, match="internal"):
        CandidateGroup("project", (first, second))


def test_group_constructor_rejects_different_known_values() -> None:
    candidates = (
        make_candidate(candidate_id="a", value="main"),
        make_candidate(candidate_id="b", value="other"),
    )

    with pytest.raises(TypeError, match="internal"):
        CandidateGroup("project", candidates)


def test_group_constructor_rejects_mixed_known_and_unresolved_values() -> None:
    candidates = (
        make_candidate(candidate_id="a", value="main"),
        make_candidate(candidate_id="b", value=unknown("missing")),
    )

    with pytest.raises(TypeError, match="internal"):
        CandidateGroup("project", candidates)


def test_group_constructor_rejects_multiple_unresolved_candidates() -> None:
    candidates = (
        make_candidate(candidate_id="a", value=unknown("missing")),
        make_candidate(candidate_id="b", value=unknown("also missing")),
    )

    with pytest.raises(TypeError, match="internal"):
        CandidateGroup("project", candidates)


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        ((), "non-empty"),
        (
            (
                make_candidate(candidate_id="b"),
                make_candidate(candidate_id="a"),
            ),
            "sorted",
        ),
        (
            (
                make_candidate(candidate_id="same"),
                make_candidate(candidate_id="same"),
            ),
            "duplicate",
        ),
    ],
)
def test_group_constructor_rejects_invalid_candidate_membership(
    candidates, message
) -> None:
    with pytest.raises(TypeError, match="internal"):
        CandidateGroup("project", candidates)


@pytest.mark.parametrize(
    "injected",
    [
        {"canonical_value_key": canonical_typed_value("wrong")},
        {"fact_id": "fact:v1:arbitrary"},
        {"subject": "injected"},
        {"predicate": "injected"},
    ],
)
def test_group_constructor_does_not_accept_injected_identity_fields(
    injected,
) -> None:
    candidate = make_candidate(candidate_id="a", value="main")
    values = {
        "project_id": "project",
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "fact_id": rules.make_fact_id(
            "project",
            candidate.subject,
            candidate.predicate,
            candidate.value.value,
        ),
        "canonical_value_key": canonical_typed_value(candidate.value.value),
        "candidates": (candidate,),
    }
    values.update(injected)

    with pytest.raises(TypeError):
        CandidateGroup(**values)


def test_group_constructor_derives_exact_known_identity_fields() -> None:
    candidate = make_candidate(
        candidate_id="a",
        subject="service",
        predicate="branch",
        value={"nested": ["main", 1]},
    )

    group = group_candidates("project", (candidate,))[0]

    assert group.project_id == "project"
    assert group.subject == "service"
    assert group.predicate == "branch"
    assert group.canonical_value_key == canonical_typed_value(
        {"nested": ["main", 1]}
    )
    assert group.fact_id == rules.make_fact_id(
        "project", "service", "branch", {"nested": ["main", 1]}
    )


@pytest.mark.parametrize(
    "value",
    [unknown("missing"), unavailable("offline")],
)
def test_group_constructor_derives_exact_unresolved_identity_fields(value) -> None:
    candidate = make_candidate(candidate_id="u", value=value)

    group = group_candidates("project", (candidate,))[0]

    assert group.canonical_value_key == (
        f"unresolved:{candidate.value.status.value}:u"
    )
    assert group.fact_id == _unresolved_id("project", candidate)


def test_two_same_value_historical_candidates_are_not_active() -> None:
    group = single_group(
        make_historical_candidate(
            candidate_id="h1",
            value="main",
            confidence=known(0.9),
        ),
        make_historical_candidate(
            candidate_id="h2",
            value="main",
            confidence=known(0.9),
        ),
    )

    decision = classify_group(group, NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == "candidate group does not satisfy ACTIVE gates"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()


def test_direct_current_candidate_produces_active_decision() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        value={"branch": ["main", 1]},
        confidence=known(0.8, source="current:confidence"),
    )

    decision = classify_group(
        single_group(candidate),
        NOW,
        default_policy(),
    )

    assert isinstance(decision, FactDecision)
    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.DIRECT_CURRENT
    assert decision.fact_id == make_fact_id(
        "synthetic-project",
        "project",
        "setting",
        {"branch": ["main", 1]},
    )
    assert decision.subject == "project"
    assert decision.predicate == "setting"
    assert decision.selected_value.value == {"branch": ("main", 1)}
    assert decision.candidate_ids == ("current",)
    assert decision.activation_witness_candidate_ids == ("current",)
    assert decision.confidence.value == 0.8
    assert decision.confidence.source == "current:confidence"
    assert decision.requires_human_review is False
    assert decision.relation_requests == ()


def test_current_witness_merges_historical_provenance_into_one_active_fact() -> None:
    current = make_current_candidate(
        candidate_id="current",
        value="main",
        source_ref="current:record",
        confidence=known(0.8, source="current:confidence"),
    )
    historical = make_historical_candidate(
        candidate_id="history",
        value="main",
        source_ref="history:record",
        confidence=known(0.9, source="history:confidence"),
    )

    decision = classify_group(
        single_group(current, historical),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.SAME_VALUE_MERGE
    assert decision.candidate_ids == ("current", "history")
    assert decision.activation_witness_candidate_ids == ("current",)
    assert decision.confidence.value == 0.8
    assert decision.confidence.source == "current:confidence"
    assert {
        "current:record",
        "history:record",
        "current:confidence",
        "history:confidence",
        current.value.source,
        historical.value.source,
    } <= set(decision.evidence_refs)


def test_active_confidence_is_maximum_among_witnesses_only() -> None:
    lower = make_current_candidate(
        candidate_id="a-lower",
        value="main",
        confidence=known(0.7, source="lower:confidence"),
    )
    higher = make_current_candidate(
        candidate_id="b-higher",
        value="main",
        confidence=known(0.8, source="higher:confidence"),
    )
    history = make_historical_candidate(
        candidate_id="history",
        value="main",
        confidence=known(0.99, source="history:confidence"),
    )

    decision = classify_group(
        single_group(history, higher, lower),
        NOW,
        default_policy(),
    )

    assert decision.activation_witness_candidate_ids == (
        "a-lower",
        "b-higher",
    )
    assert decision.confidence.value == 0.8
    assert decision.confidence.source == "higher:confidence"


def test_active_valid_from_is_latest_known_witness_start_only() -> None:
    earlier = make_current_candidate(
        candidate_id="earlier",
        value="main",
        valid_from=known(
            "2026-09-13T20:00:00+00:00",
            source="earlier:from",
        ),
    )
    latest = make_current_candidate(
        candidate_id="latest",
        value="main",
        valid_from=known(
            "2026-09-13T22:00:00+00:00",
            source="latest:from",
        ),
    )
    history = make_historical_candidate(
        candidate_id="history",
        value="main",
        valid_from=known(
            "2026-09-13T23:00:00+00:00",
            source="history:from",
        ),
    )

    decision = classify_group(
        single_group(history, latest, earlier),
        NOW,
        default_policy(),
    )

    assert decision.valid_from.value == "2026-09-13T22:00:00+00:00"
    assert decision.valid_from.source == "latest:from"


def test_unknown_valid_from_does_not_block_direct_current_activation() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        value="main",
        valid_from=unknown("effective start was not recorded", source="current:from"),
        confidence=known(0.8),
    )

    decision = classify_group(
        single_group(candidate),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.valid_from.status is candidate.valid_from.status
    assert decision.valid_from.reason == "effective start was not recorded"
    assert decision.valid_from.source == "current:from"


def test_known_witness_start_wins_over_another_witness_unknown_start() -> None:
    known_start = make_current_candidate(
        candidate_id="known",
        value="main",
        valid_from=known(
            "2026-09-13T22:00:00+00:00",
            source="known:from",
        ),
    )
    unknown_start = make_current_candidate(
        candidate_id="unknown",
        value="main",
        valid_from=unknown("not recorded", source="unknown:from"),
    )

    decision = classify_group(
        single_group(unknown_start, known_start),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.valid_from.value == "2026-09-13T22:00:00+00:00"
    assert decision.valid_from.source == "known:from"


def test_adding_same_value_history_does_not_change_stable_fact_id() -> None:
    current = make_current_candidate(candidate_id="current", value="main")
    history = make_historical_candidate(candidate_id="history", value="main")

    direct = classify_group(single_group(current), NOW, default_policy())
    merged = classify_group(
        single_group(current, history),
        NOW,
        default_policy(),
    )

    assert direct.fact_id == merged.fact_id


def test_same_value_reactivation_uses_lineage_method_and_candidate_ids() -> None:
    old = make_candidate(
        candidate_id="old",
        value="local",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(old, new),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.SAME_VALUE_REACTIVATION
    assert decision.activation_witness_candidate_ids == ("new",)
    assert decision.superseded_candidate_ids == ("old",)
    assert decision.relation_requests == ()


def test_reactivation_uses_only_surviving_tip_as_activation_witness() -> None:
    old = make_current_candidate(
        candidate_id="current-old",
        value="local",
        confidence=known(0.95, source="old:confidence"),
        valid_from=known(NOW.isoformat(), source="old:from"),
    )
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="local",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        supersedes=known(("current-old",)),
    )
    new = make_current_candidate(
        candidate_id="current-new",
        value="local",
        confidence=known(0.60, source="new:confidence"),
        valid_from=known(
            "2026-09-13T23:59:00+00:00",
            source="new:from",
        ),
        supersedes=known(("deprecated",)),
    )

    decision = classify_group(
        single_group(old, deprecated, new),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.SAME_VALUE_REACTIVATION
    assert decision.activation_witness_candidate_ids == ("current-new",)
    assert decision.superseded_candidate_ids == ("current-old", "deprecated")
    assert decision.confidence.value == 0.60
    assert decision.confidence.source == "new:confidence"
    assert decision.valid_from.value == "2026-09-13T23:59:00+00:00"
    assert decision.valid_from.source == "new:from"
    assert not (
        set(decision.activation_witness_candidate_ids)
        & set(decision.superseded_candidate_ids)
    )


def test_reactivation_excludes_superseded_current_deprecated_witness() -> None:
    old = make_current_candidate(
        candidate_id="old",
        value="local",
        deprecated=known(True),
        confidence=known(0.95),
    )
    new = make_current_candidate(
        candidate_id="new",
        value="local",
        supersedes=known(("old",)),
        confidence=known(0.60),
    )

    decision = classify_group(single_group(old, new), NOW, default_policy())

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.resolution_method is ResolutionMethod.SAME_VALUE_REACTIVATION
    assert decision.activation_witness_candidate_ids == ("new",)
    assert decision.superseded_candidate_ids == ("old",)
    assert decision.confidence.value == 0.60


def test_fact_decision_factory_rejects_witness_superseded_overlap() -> None:
    decision = classify_group(
        single_group(make_current_candidate(candidate_id="current")),
        NOW,
        default_policy(),
    )
    forged = {
        field.name: getattr(decision, field.name)
        for field in fields(FactDecision)
        if field.name != "relation_requests"
    }
    forged["superseded_candidate_ids"] = ("current",)

    with pytest.raises(ReconciliationInputError, match="overlap"):
        FactDecision._from_canonical_group(
            reconciliation_models._FACT_DECISION_CONSTRUCTION_TOKEN,
            **forged,
        )


def test_unresolved_same_value_lineage_is_provisionally_pending_for_review() -> None:
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="local",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
    )
    current = make_current_candidate(
        candidate_id="current",
        value="local",
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(deprecated, current),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.requires_human_review is True
    assert "lineage" in decision.reason.lower()
    assert decision.superseded_candidate_ids == ()


def test_explicit_deprecation_cannot_become_active_provisionally() -> None:
    deprecated = make_candidate(
        candidate_id="deprecated",
        value="v1",
        status_hint=known(CandidateStatusHint.DEPRECATED),
        deprecated=known(True),
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is not ReconciliationStatus.ACTIVE
    assert decision.activation_witness_candidate_ids == ()
    assert decision.relation_requests == ()


def test_winning_same_value_deprecation_blocks_an_older_witness() -> None:
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
        supersedes=known(("current",)),
    )

    decision = classify_group(
        single_group(current, deprecated),
        NOW,
        default_policy(),
    )

    assert decision.status is not ReconciliationStatus.ACTIVE
    assert decision.activation_witness_candidate_ids == ()
    assert decision.superseded_candidate_ids == ("current",)


def test_current_with_future_validity_remains_provisional_pending() -> None:
    future = make_current_candidate(
        candidate_id="future",
        valid_from=known("2026-09-14T00:00:01+00:00"),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(future), NOW, default_policy())

    assert decision.status is not ReconciliationStatus.ACTIVE
    assert decision.activation_witness_candidate_ids == ()


def test_low_confidence_current_remains_provisional_pending() -> None:
    candidate = make_current_candidate(
        candidate_id="low",
        confidence=known(0.49),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is not ReconciliationStatus.ACTIVE
    assert decision.activation_witness_candidate_ids == ()


def test_fact_decision_is_deeply_immutable() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        value={"nested": ["main", {"enabled": True}]},
    )
    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert isinstance(decision.candidate_ids, tuple)
    assert isinstance(decision.evidence_refs, tuple)
    assert isinstance(decision.activation_witness_candidate_ids, tuple)
    assert isinstance(decision.superseded_candidate_ids, tuple)
    assert isinstance(decision.warnings, tuple)
    assert isinstance(decision.relation_requests, tuple)
    with pytest.raises(FrozenInstanceError):
        decision.reason = "changed"
    with pytest.raises(TypeError):
        decision.selected_value.value["nested"] = ()
    with pytest.raises(TypeError):
        decision.selected_value.value["nested"][1]["enabled"] = False


def test_fact_decision_rejects_replaced_field_level_provenance() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        value="main",
    )
    decision = classify_group(single_group(candidate), NOW, default_policy())
    incomplete_refs = tuple(
        ref
        for ref in decision.evidence_refs
        if ref != candidate.value.source
    )

    with pytest.raises(TypeError, match="classify_group"):
        replace(decision, evidence_refs=incomplete_refs)


def test_fact_decision_rejects_direct_active_historical_witness_forgery() -> None:
    historical = make_historical_candidate(candidate_id="historical")
    decision = classify_group(
        single_group(historical),
        NOW,
        default_policy(),
    )
    forged = {
        field.name: getattr(decision, field.name)
        for field in fields(FactDecision)
    }
    forged.update(
        status=ReconciliationStatus.ACTIVE,
        resolution_method=ResolutionMethod.DIRECT_CURRENT,
        activation_witness_candidate_ids=("historical",),
    )

    with pytest.raises(TypeError, match="classify_group"):
        FactDecision(**forged)


def test_fact_decision_rejects_replaced_unrelated_fact_id() -> None:
    decision = classify_group(
        single_group(make_current_candidate(candidate_id="current")),
        NOW,
        default_policy(),
    )

    with pytest.raises(TypeError, match="classify_group"):
        replace(decision, fact_id=f"fact:v1:{'0' * 64}")


def test_fact_decision_rejects_replaced_incomplete_candidate_ids() -> None:
    decision = classify_group(
        single_group(
            make_current_candidate(candidate_id="current", value="main"),
            make_historical_candidate(candidate_id="history", value="main"),
        ),
        NOW,
        default_policy(),
    )

    with pytest.raises(TypeError, match="classify_group"):
        replace(decision, candidate_ids=("current",))


def test_fact_decision_rejects_active_pending_method_combination() -> None:
    decision = classify_group(
        single_group(make_current_candidate(candidate_id="current")),
        NOW,
        default_policy(),
    )

    with pytest.raises(TypeError, match="classify_group"):
        replace(
            decision,
            resolution_method=ResolutionMethod.PENDING_SEMANTICS,
        )


def test_fact_decision_rejects_pending_with_injected_witness() -> None:
    decision = classify_group(
        single_group(make_historical_candidate(candidate_id="historical")),
        NOW,
        default_policy(),
    )

    with pytest.raises(TypeError, match="classify_group"):
        replace(
            decision,
            activation_witness_candidate_ids=("historical",),
        )


def test_classify_group_factory_derives_all_identity_and_provenance_fields() -> None:
    current = make_current_candidate(candidate_id="current", value="main")
    history = make_historical_candidate(candidate_id="history", value="main")
    group = single_group(current, history)

    decision = classify_group(group, NOW, default_policy())

    assert decision.fact_id == group.fact_id
    assert decision.subject == group.subject
    assert decision.predicate == group.predicate
    assert decision.candidate_ids == group.candidate_ids
    assert decision.activation_witness_candidate_ids == ("current",)
    assert {current.source_ref, history.source_ref} <= set(decision.evidence_refs)
