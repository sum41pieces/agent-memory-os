"""PENDING classification semantics for same-value candidate groups."""

import pytest

from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    ReconciliationInputError,
    ReconciliationStatus,
    ResolutionMethod,
    SourceType,
    WarningCode,
)
from agent_memory_os.reconcile.rules import CandidateGroup, classify_group
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
    unavailable,
    unknown,
)


def test_future_plan_is_pending_without_human_review() -> None:
    candidate = make_candidate(
        candidate_id="plan",
        value=True,
        status_hint=known(CandidateStatusHint.PLAN),
        valid_from=known("2026-09-15T00:00:00+00:00"),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate is not yet valid"
    assert decision.requires_human_review is False
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.FUTURE_VALIDITY,
    )


@pytest.mark.parametrize(
    ("hint", "source_type"),
    [
        (CandidateStatusHint.PLAN, SourceType.USER_EXPLICIT),
        (CandidateStatusHint.HYPOTHESIS, SourceType.CURRENT_EVIDENCE),
    ],
)
def test_intent_remains_pending_even_from_authoritative_source_types(
    hint: CandidateStatusHint,
    source_type: SourceType,
) -> None:
    candidate = make_candidate(
        candidate_id=hint.value.lower(),
        value=True,
        status_hint=known(hint),
        source_type=source_type,
        explicit_user_instruction=source_type is SourceType.USER_EXPLICIT,
        confidence=known(0.99),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == f"{hint.value} is intent, not current fact evidence"
    assert decision.requires_human_review is False
    assert decision.activation_witness_candidate_ids == ()


@pytest.mark.parametrize(
    "candidate",
    [
        make_current_candidate(
            candidate_id="current",
            valid_from=known(iso_after(NOW, 1)),
            confidence=known(0.9),
        ),
        make_source_of_truth_candidate(
            candidate_id="source-of-truth",
            valid_from=known(iso_after(NOW, 1)),
            confidence=known(0.9),
        ),
    ],
    ids=("current", "source-of-truth"),
)
def test_future_current_semantics_are_temporally_pending_for_review(
    candidate: object,
) -> None:
    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate is not yet valid"
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.FUTURE_VALIDITY,
    )


def test_invalid_interval_is_temporally_pending_for_review() -> None:
    candidate = make_current_candidate(
        candidate_id="invalid",
        valid_from=known(iso_after(NOW, 10), source="invalid:from"),
        valid_until=known(iso_after(NOW, 5), source="invalid:until"),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate interval is invalid"
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.FUTURE_VALIDITY,
        WarningCode.INVALID_TEMPORAL_ORDER,
    )
    invalid_warning = next(
        warning
        for warning in decision.warnings
        if warning.code is WarningCode.INVALID_TEMPORAL_ORDER
    )
    assert invalid_warning.candidate_ids == ("invalid",)
    assert invalid_warning.evidence_refs == ("invalid:from", "invalid:until")


def test_intent_with_invalid_interval_uses_temporal_priority_and_requires_review(
) -> None:
    candidate = make_candidate(
        candidate_id="plan",
        status_hint=known(CandidateStatusHint.PLAN),
        valid_from=known(iso_after(NOW, 10)),
        valid_until=known(iso_after(NOW, 5)),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate interval is invalid"
    assert decision.requires_human_review is True


def test_future_clock_skew_is_temporally_pending_for_review() -> None:
    candidate = make_current_candidate(
        candidate_id="skewed",
        observed_at=known(iso_after(NOW, 301)),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate observation exceeds allowed future clock skew"
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.FUTURE_CLOCK_SKEW,
    )


def test_future_clock_skew_on_intent_still_requires_review() -> None:
    candidate = make_candidate(
        candidate_id="skewed-plan",
        status_hint=known(CandidateStatusHint.PLAN),
        observed_at=known(iso_after(NOW, 301)),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate observation exceeds allowed future clock skew"
    assert decision.requires_human_review is True


def test_known_below_threshold_current_is_insufficient_without_review() -> None:
    candidate = make_current_candidate(
        candidate_id="low",
        confidence=known(0.49, source="low:confidence"),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.INSUFFICIENT_EVIDENCE
    assert decision.reason == "current evidence confidence is below the activation threshold"
    assert decision.requires_human_review is False
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.INSUFFICIENT_CONFIDENCE,
    )
    assert decision.warnings[0].candidate_ids == ("low",)
    assert decision.warnings[0].evidence_refs == ("low:confidence",)
    assert decision.warnings[0].requires_human_review is False


def test_temporal_method_precedes_insufficient_confidence_without_dropping_warning(
) -> None:
    candidate = make_current_candidate(
        candidate_id="future-low",
        valid_from=known(iso_after(NOW, 1), source="future-low:from"),
        confidence=known(0.49, source="future-low:confidence"),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate is not yet valid"
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.FUTURE_VALIDITY,
        WarningCode.INSUFFICIENT_CONFIDENCE,
    )


@pytest.mark.parametrize(
    (
        "candidates",
        "expected_method",
        "expected_reason",
        "expected_warning_codes",
        "expected_review",
    ),
    [
        (
            (
                make_candidate(
                    candidate_id="plan",
                    status_hint=known(CandidateStatusHint.PLAN),
                    confidence=known(0.9),
                ),
            ),
            ResolutionMethod.PENDING_SEMANTICS,
            "PLAN is intent, not current fact evidence",
            (),
            False,
        ),
        (
            (
                make_candidate(
                    candidate_id="future-plan",
                    status_hint=known(CandidateStatusHint.PLAN),
                    valid_from=known(iso_after(NOW, 1)),
                    confidence=known(0.9),
                ),
            ),
            ResolutionMethod.TEMPORAL_PENDING,
            "candidate is not yet valid",
            (WarningCode.FUTURE_VALIDITY,),
            False,
        ),
        (
            (
                make_candidate(
                    candidate_id="plan",
                    status_hint=known(CandidateStatusHint.PLAN),
                    confidence=known(0.9),
                ),
                make_current_candidate(
                    candidate_id="current-low",
                    confidence=known(0.49),
                ),
            ),
            ResolutionMethod.INSUFFICIENT_EVIDENCE,
            "current evidence confidence is below the activation threshold",
            (WarningCode.INSUFFICIENT_CONFIDENCE,),
            False,
        ),
        (
            (
                make_candidate(
                    candidate_id="plan",
                    status_hint=known(CandidateStatusHint.PLAN),
                    confidence=known(0.9),
                ),
                make_current_candidate(
                    candidate_id="future-current-low",
                    valid_from=known(iso_after(NOW, 1)),
                    confidence=known(0.49),
                ),
            ),
            ResolutionMethod.TEMPORAL_PENDING,
            "candidate is not yet valid",
            (
                WarningCode.FUTURE_VALIDITY,
                WarningCode.INSUFFICIENT_CONFIDENCE,
            ),
            True,
        ),
    ],
    ids=(
        "other-pending",
        "temporal-over-other",
        "insufficient-over-other",
        "temporal-over-insufficient-over-other",
    ),
)
def test_pending_primary_method_priority_matrix(
    candidates: tuple[object, ...],
    expected_method: ResolutionMethod,
    expected_reason: str,
    expected_warning_codes: tuple[WarningCode, ...],
    expected_review: bool,
) -> None:
    decision = classify_group(single_group(*candidates), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is expected_method
    assert decision.reason == expected_reason
    assert tuple(warning.code for warning in decision.warnings) == (
        expected_warning_codes
    )
    assert decision.requires_human_review is expected_review


def test_classify_group_rejects_candidate_group_subclass_with_forged_identity(
) -> None:
    canonical = single_group(
        make_historical_candidate(candidate_id="history", value="main")
    )

    class ForgedCandidateGroup(CandidateGroup):
        def __getattribute__(self, name: str) -> object:
            if name == "fact_id":
                return f"fact:v1:{'0' * 64}"
            return super().__getattribute__(name)

    hostile = object.__new__(ForgedCandidateGroup)
    hostile.__dict__.update(canonical.__dict__)

    with pytest.raises(ReconciliationInputError, match="group"):
        classify_group(hostile, NOW, default_policy())


@pytest.mark.parametrize(
    ("confidence", "warning_code"),
    [
        (
            unknown("confidence was not recorded", source="confidence:unknown"),
            WarningCode.EVIDENCE_UNKNOWN,
        ),
        (
            unavailable("confidence source failed", source="confidence:unavailable"),
            WarningCode.EVIDENCE_UNAVAILABLE,
        ),
    ],
)
@pytest.mark.parametrize(
    ("status_hint", "source_type", "explicit_user_instruction"),
    [
        (
            CandidateStatusHint.CURRENT_FACT,
            SourceType.CURRENT_EVIDENCE,
            False,
        ),
        (
            CandidateStatusHint.SOURCE_OF_TRUTH,
            SourceType.USER_EXPLICIT,
            True,
        ),
    ],
    ids=("current", "source-of-truth"),
)
def test_nonknown_current_confidence_is_pending_and_requires_review(
    confidence: object,
    warning_code: WarningCode,
    status_hint: CandidateStatusHint,
    source_type: SourceType,
    explicit_user_instruction: bool,
) -> None:
    candidate = make_candidate(
        candidate_id="current",
        status_hint=known(status_hint),
        source_type=source_type,
        explicit_user_instruction=explicit_user_instruction,
        confidence=confidence,
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == (
        f"confidence is {confidence.status.value} and prevents current classification"
    )
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (warning_code,)
    assert decision.warnings[0].candidate_ids == ("current",)
    assert decision.warnings[0].evidence_refs == (confidence.source,)
    assert decision.warnings[0].requires_human_review is True


@pytest.mark.parametrize(
    ("status_hint", "warning_code"),
    [
        (
            unknown("status was not recorded", source="status:unknown"),
            WarningCode.EVIDENCE_UNKNOWN,
        ),
        (
            unavailable("status source failed", source="status:unavailable"),
            WarningCode.EVIDENCE_UNAVAILABLE,
        ),
    ],
)
@pytest.mark.parametrize(
    ("source_type", "explicit_user_instruction"),
    [
        (SourceType.CURRENT_EVIDENCE, False),
        (SourceType.USER_EXPLICIT, True),
    ],
    ids=("current", "source-of-truth"),
)
def test_nonknown_status_for_current_or_source_of_truth_requires_review(
    status_hint: object,
    warning_code: WarningCode,
    source_type: SourceType,
    explicit_user_instruction: bool,
) -> None:
    candidate = make_candidate(
        candidate_id="candidate",
        source_type=source_type,
        status_hint=status_hint,
        explicit_user_instruction=explicit_user_instruction,
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == (
        f"status_hint is {status_hint.status.value} and prevents current classification"
    )
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (warning_code,)
    assert decision.warnings[0].evidence_refs == (status_hint.source,)


def test_nonknown_status_is_not_critical_when_source_of_truth_override_disabled(
) -> None:
    status_hint = unknown(
        "status was not recorded",
        source="status:unknown",
    )
    candidate = make_candidate(
        candidate_id="candidate",
        source_type=SourceType.USER_EXPLICIT,
        status_hint=status_hint,
        explicit_user_instruction=True,
        confidence=known(0.9),
    )

    decision = classify_group(
        single_group(candidate),
        NOW,
        default_policy(allow_explicit_source_of_truth_override=False),
    )

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.requires_human_review is False
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.EVIDENCE_UNKNOWN,
    )
    assert decision.warnings[0].candidate_ids == ("candidate",)
    assert decision.warnings[0].evidence_refs == (status_hint.source,)
    assert decision.warnings[0].requires_human_review is False


def test_same_code_warnings_sort_by_candidate_id_before_message() -> None:
    status_unknown = make_historical_candidate(
        candidate_id="a-status",
        status_hint=unknown("status not recorded", source="a:status"),
        confidence=known(0.9),
    )
    confidence_unknown = make_current_candidate(
        candidate_id="z-confidence",
        confidence=unknown("confidence not recorded", source="z:confidence"),
    )

    decision = classify_group(
        single_group(confidence_unknown, status_unknown),
        NOW,
        default_policy(),
    )

    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.EVIDENCE_UNKNOWN,
        WarningCode.EVIDENCE_UNKNOWN,
    )
    assert tuple(warning.candidate_ids for warning in decision.warnings) == (
        ("a-status",),
        ("z-confidence",),
    )


@pytest.mark.parametrize(
    ("explicit_user_instruction", "warning_code"),
    [
        (
            unknown(
                "explicit designation was not recorded",
                source="explicit:unknown",
            ),
            WarningCode.EVIDENCE_UNKNOWN,
        ),
        (
            unavailable(
                "explicit designation source failed",
                source="explicit:unavailable",
            ),
            WarningCode.EVIDENCE_UNAVAILABLE,
        ),
    ],
)
def test_nonknown_explicit_designation_blocks_source_of_truth_classification(
    explicit_user_instruction: object,
    warning_code: WarningCode,
) -> None:
    candidate = make_candidate(
        candidate_id="source-of-truth",
        source_type=SourceType.USER_EXPLICIT,
        status_hint=known(CandidateStatusHint.SOURCE_OF_TRUTH),
        explicit_user_instruction=explicit_user_instruction,
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == (
        "explicit_user_instruction is "
        f"{explicit_user_instruction.status.value} and prevents current "
        "classification"
    )
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (warning_code,)
    assert decision.warnings[0].candidate_ids == ("source-of-truth",)
    assert decision.warnings[0].evidence_refs == (
        explicit_user_instruction.source,
    )
    assert decision.warnings[0].requires_human_review is True


def test_nonknown_explicit_flag_is_not_critical_for_ordinary_current_fact() -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        explicit_user_instruction=unknown(
            "not relevant to ordinary current evidence",
            source="current:explicit",
        ),
        confidence=known(0.9),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.requires_human_review is False
    assert decision.warnings == ()


@pytest.mark.parametrize(
    "candidate",
    [
        make_current_candidate(
            candidate_id="current",
            valid_from=known(iso_after(NOW, -10)),
            valid_until=known(iso_after(NOW, -1)),
        ),
        make_source_of_truth_candidate(
            candidate_id="source-of-truth",
            valid_from=known(iso_after(NOW, -10)),
            valid_until=known(iso_after(NOW, -1)),
        ),
    ],
    ids=("current", "source-of-truth"),
)
def test_expired_current_semantics_are_temporally_pending_for_review(
    candidate: object,
) -> None:
    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.status is not ReconciliationStatus.SUPERSEDED
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.reason == "candidate expired without an eligible replacement"
    assert decision.requires_human_review is True
    assert tuple(warning.code for warning in decision.warnings) == (
        WarningCode.EXPIRED_WITHOUT_REPLACEMENT,
    )


def test_expired_historical_evidence_is_temporally_pending_without_review() -> None:
    candidate = make_historical_candidate(
        candidate_id="history",
        valid_from=known(iso_after(NOW, -10)),
        valid_until=known(iso_after(NOW, -1)),
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.TEMPORAL_PENDING
    assert decision.requires_human_review is False


def test_historical_only_group_uses_pending_semantics_without_review() -> None:
    decision = classify_group(
        single_group(
            make_historical_candidate(candidate_id="history", confidence=known(0.99))
        ),
        NOW,
        default_policy(),
    )

    assert decision.status is ReconciliationStatus.PENDING
    assert decision.resolution_method is ResolutionMethod.PENDING_SEMANTICS
    assert decision.reason == "candidate group does not satisfy ACTIVE gates"
    assert decision.requires_human_review is False
    assert decision.warnings == ()


@pytest.mark.parametrize(
    "field_name",
    ["observed_at", "valid_from", "valid_until"],
)
def test_nonknown_optional_time_does_not_overwarn_or_block_current(
    field_name: str,
) -> None:
    candidate = make_current_candidate(
        candidate_id="current",
        confidence=known(0.9),
        **{field_name: unknown("time was not recorded")},
    )

    decision = classify_group(single_group(candidate), NOW, default_policy())

    assert decision.status is ReconciliationStatus.ACTIVE
    assert decision.requires_human_review is False
    assert decision.warnings == ()
