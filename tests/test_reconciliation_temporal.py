"""Temporal parsing and eligibility rules for reconciliation candidates."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    ReconciliationInputError,
    WarningCode,
)
from agent_memory_os.reconcile.rules import (
    assess_temporal,
    parse_known_timestamp,
    parse_reconciliation_clock,
)
from reconciliation_helpers import (
    NOW,
    default_policy,
    iso_after,
    known,
    make_candidate,
    unavailable,
    unknown,
)


@pytest.mark.parametrize("raw", ["not-a-time", "2026-09-14T00:00:00"])
def test_known_timestamp_must_parse_and_have_timezone(raw: str) -> None:
    candidate = make_candidate(
        observed_at=known(raw, source="synthetic:observed_at")
    )

    with pytest.raises(ReconciliationInputError, match="observed_at"):
        assess_temporal(candidate, NOW, default_policy())


@pytest.mark.parametrize(
    ("seconds", "eligible", "warning_code", "review"),
    [
        (300, True, None, False),
        (301, False, WarningCode.FUTURE_CLOCK_SKEW, True),
    ],
)
def test_future_clock_skew_boundary(
    seconds: int,
    eligible: bool,
    warning_code: WarningCode | None,
    review: bool,
) -> None:
    candidate = make_candidate(observed_at=known(iso_after(NOW, seconds)))

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is eligible
    assert tuple(item.code for item in assessment.warnings) == (
        () if warning_code is None else (warning_code,)
    )
    assert any(item.requires_human_review for item in assessment.warnings) is review


@pytest.mark.parametrize("field_name", ["observed_at", "valid_from", "valid_until"])
@pytest.mark.parametrize("raw", ["not-a-time", "2026-09-14T00:00:00"])
def test_every_known_candidate_timestamp_is_strictly_validated(
    field_name: str,
    raw: str,
) -> None:
    candidate = make_candidate(**{field_name: known(raw)})

    with pytest.raises(ReconciliationInputError, match=field_name):
        assess_temporal(candidate, NOW, default_policy())


@pytest.mark.parametrize(
    "raw",
    ["not-a-time", "2026-09-14T00:00:00", 123],
)
def test_reconciliation_clock_is_strictly_parsed(raw: object) -> None:
    with pytest.raises(ReconciliationInputError, match="reconciliation clock"):
        parse_reconciliation_clock(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-14X00:00:00+00:00",
        "2026-09-14@00:00:00+00:00",
        "2026-09-14 00:00:00+00:00",
    ],
)
def test_known_timestamp_rejects_noncanonical_datetime_separator(
    raw: str,
) -> None:
    with pytest.raises(ReconciliationInputError, match="observed_at"):
        parse_known_timestamp(known(raw), "observed_at")


@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-14X00:00:00+00:00",
        "2026-09-14@00:00:00+00:00",
        "2026-09-14 00:00:00+00:00",
    ],
)
def test_reconciliation_clock_rejects_noncanonical_datetime_separator(
    raw: str,
) -> None:
    with pytest.raises(ReconciliationInputError, match="reconciliation clock"):
        parse_reconciliation_clock(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-14T00:00:00.123456Z",
        "2026-09-14T08:00:00.5+08:00",
        "2026-09-13T19:00:00-05:00",
    ],
)
def test_strict_parser_preserves_intended_iso_8601_forms(raw: str) -> None:
    expected = datetime(2026, 9, 14, tzinfo=timezone.utc)

    parsed = parse_known_timestamp(known(raw), "observed_at")

    assert parsed is not None
    assert parsed.replace(microsecond=0) == expected
    assert parse_reconciliation_clock(raw).replace(microsecond=0) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-14T00:05:00.0000009+00:00",
        "2026-09-14T00:00:00.1234567Z",
    ],
)
def test_candidate_timestamp_rejects_more_than_six_fractional_digits(
    raw: str,
) -> None:
    candidate = make_candidate(observed_at=known(raw))

    with pytest.raises(ReconciliationInputError, match="observed_at"):
        assess_temporal(candidate, NOW, default_policy())


@pytest.mark.parametrize(
    "raw",
    [
        "2026-09-14T00:05:00.0000009+00:00",
        "2026-09-14T00:00:00.1234567Z",
    ],
)
def test_reconciliation_clock_rejects_more_than_six_fractional_digits(
    raw: str,
) -> None:
    with pytest.raises(ReconciliationInputError, match="reconciliation clock"):
        parse_reconciliation_clock(raw)


def test_clock_skew_compares_microseconds_without_rounding() -> None:
    candidate = make_candidate(
        observed_at=known("2026-09-14T00:05:00.000001+00:00")
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is False
    assert tuple(warning.code for warning in assessment.warnings) == (
        WarningCode.FUTURE_CLOCK_SKEW,
    )


@pytest.mark.parametrize(
    "parse",
    [
        lambda raw: parse_known_timestamp(known(raw), "observed_at"),
        parse_reconciliation_clock,
    ],
)
def test_extreme_offset_normalization_overflow_is_an_input_error(parse: object) -> None:
    with pytest.raises(ReconciliationInputError):
        parse("0001-01-01T00:00:00+23:59")


def test_datetime_max_now_does_not_overflow_clock_skew_boundary() -> None:
    assessment = assess_temporal(
        make_candidate(),
        datetime.max.replace(tzinfo=timezone.utc),
        default_policy(),
    )

    assert assessment.eligible is True
    assert assessment.warnings == ()


def test_huge_clock_skew_policy_does_not_overflow() -> None:
    assessment = assess_temporal(
        make_candidate(observed_at=known(iso_after(NOW, 301))),
        NOW,
        default_policy(max_future_clock_skew_seconds=10**100),
    )

    assert assessment.eligible is True
    assert assessment.warnings == ()


@pytest.mark.parametrize(
    "parse",
    [
        lambda raw: parse_known_timestamp(known(raw), "observed_at"),
        parse_reconciliation_clock,
    ],
)
def test_fractional_timezone_offsets_are_rejected(parse: object) -> None:
    with pytest.raises(ReconciliationInputError):
        parse("2026-09-14T00:00:00+08:00:00.5")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-14T00:00:00Z", datetime(2026, 9, 14, tzinfo=timezone.utc)),
        (
            "2026-09-14T08:00:00+08:00",
            datetime(2026, 9, 14, tzinfo=timezone.utc),
        ),
    ],
)
def test_timestamps_normalize_z_and_offsets_to_utc(
    raw: str,
    expected: datetime,
) -> None:
    parsed = parse_known_timestamp(known(raw), "observed_at")

    assert parsed == expected
    assert parsed is not None
    assert parsed.tzinfo is timezone.utc
    assert parse_reconciliation_clock(raw) == expected


@pytest.mark.parametrize(
    "evidence",
    [
        unknown("timestamp was not recorded"),
        unavailable("timestamp source could not be read"),
    ],
)
def test_nonknown_timestamp_parses_to_none(evidence: object) -> None:
    assert parse_known_timestamp(evidence, "observed_at") is None


def test_invalid_temporal_order_is_pending_and_requires_review() -> None:
    candidate = make_candidate(
        valid_from=known("2026-09-14T01:00:00+00:00", source="from:source"),
        valid_until=known("2026-09-14T00:30:00+00:00", source="until:source"),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is False
    assert assessment.pending_reason == WarningCode.INVALID_TEMPORAL_ORDER.value
    assert tuple(warning.code for warning in assessment.warnings) == (
        WarningCode.FUTURE_VALIDITY,
        WarningCode.INVALID_TEMPORAL_ORDER,
    )
    order_warning = next(
        warning
        for warning in assessment.warnings
        if warning.code is WarningCode.INVALID_TEMPORAL_ORDER
    )
    assert order_warning.requires_human_review is True
    assert order_warning.evidence_refs == ("from:source", "until:source")


@pytest.mark.parametrize(
    "status_hint",
    [CandidateStatusHint.PLAN, CandidateStatusHint.HYPOTHESIS],
)
def test_future_plan_or_hypothesis_validity_does_not_require_review(
    status_hint: CandidateStatusHint,
) -> None:
    candidate = make_candidate(
        status_hint=known(status_hint),
        valid_from=known(iso_after(NOW, 1)),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is False
    assert assessment.pending_reason == WarningCode.FUTURE_VALIDITY.value
    assert tuple(warning.code for warning in assessment.warnings) == (
        WarningCode.FUTURE_VALIDITY,
    )
    assert assessment.warnings[0].requires_human_review is False


@pytest.mark.parametrize(
    "status_hint",
    [CandidateStatusHint.CURRENT_FACT, CandidateStatusHint.SOURCE_OF_TRUTH],
)
def test_future_current_or_source_of_truth_validity_requires_review(
    status_hint: CandidateStatusHint,
) -> None:
    candidate = make_candidate(
        status_hint=known(status_hint),
        valid_from=known(iso_after(NOW, 1)),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is False
    assert assessment.warnings[0].code is WarningCode.FUTURE_VALIDITY
    assert assessment.warnings[0].requires_human_review is True


@pytest.mark.parametrize(
    ("status_hint", "review"),
    [
        (CandidateStatusHint.CURRENT_FACT, True),
        (CandidateStatusHint.SOURCE_OF_TRUTH, True),
        (CandidateStatusHint.PLAN, False),
        (CandidateStatusHint.HYPOTHESIS, False),
        (CandidateStatusHint.HISTORICAL, False),
    ],
)
def test_expired_without_replacement_is_pending_with_semantic_review_policy(
    status_hint: CandidateStatusHint,
    review: bool,
) -> None:
    candidate = make_candidate(
        status_hint=known(status_hint),
        valid_from=known(iso_after(NOW, -20)),
        valid_until=known(iso_after(NOW, -10)),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is False
    assert assessment.pending_reason == WarningCode.EXPIRED_WITHOUT_REPLACEMENT.value
    assert tuple(warning.code for warning in assessment.warnings) == (
        WarningCode.EXPIRED_WITHOUT_REPLACEMENT,
    )
    assert assessment.warnings[0].requires_human_review is review


@pytest.mark.parametrize("field_name", ["observed_at", "valid_from", "valid_until"])
@pytest.mark.parametrize("status", [EvidenceStatus.UNKNOWN, EvidenceStatus.UNAVAILABLE])
def test_nonknown_times_do_not_automatically_invalidate_direct_current(
    field_name: str,
    status: EvidenceStatus,
) -> None:
    evidence = (
        unknown("time is unknown")
        if status is EvidenceStatus.UNKNOWN
        else unavailable("time source unavailable")
    )
    candidate = make_candidate(**{field_name: evidence})

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is True
    assert assessment.pending_reason is None
    assert WarningCode.UNRESOLVED_TEMPORAL_COMPARISON not in {
        warning.code for warning in assessment.warnings
    }
    assert getattr(assessment, field_name) is None


def test_unknown_interval_endpoint_only_blocks_ordering_comparison() -> None:
    candidate = make_candidate(
        valid_from=unknown("validity start was not recorded"),
        valid_until=known(iso_after(NOW, 60)),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert assessment.eligible is True
    assert assessment.valid_from is None
    assert assessment.valid_until == NOW.replace(second=0) + (
        datetime.fromisoformat(iso_after(NOW, 60)) - NOW
    )
    assert assessment.warnings == ()


def test_custom_future_clock_skew_policy_is_honored() -> None:
    policy = default_policy(max_future_clock_skew_seconds=10)

    accepted = assess_temporal(
        make_candidate(observed_at=known(iso_after(NOW, 10))),
        NOW,
        policy,
    )
    rejected = assess_temporal(
        make_candidate(observed_at=known(iso_after(NOW, 11))),
        NOW,
        policy,
    )

    assert accepted.eligible is True
    assert accepted.warnings == ()
    assert rejected.eligible is False
    assert rejected.warnings[0].code is WarningCode.FUTURE_CLOCK_SKEW


def test_temporal_assessment_is_bound_to_candidate_clock_and_policy() -> None:
    candidate = make_candidate(candidate_id="bound-candidate")
    policy = default_policy(max_future_clock_skew_seconds=10)

    assessment = assess_temporal(candidate, NOW, policy)

    assert assessment.candidate_id == "bound-candidate"
    assert assessment.assessed_at == NOW
    assert assessment.max_future_clock_skew_seconds == 10


def test_assess_temporal_requires_an_aware_injected_now() -> None:
    candidate = make_candidate()

    with pytest.raises(ReconciliationInputError, match="now"):
        assess_temporal(
            candidate,
            datetime(2026, 9, 14),
            default_policy(),
        )


def test_temporal_assessment_is_frozen_and_warnings_are_sorted() -> None:
    candidate = make_candidate(
        observed_at=known(iso_after(NOW, 301)),
        valid_from=known(iso_after(NOW, 600)),
        valid_until=known(iso_after(NOW, 500)),
    )

    assessment = assess_temporal(candidate, NOW, default_policy())

    assert tuple(warning.code.value for warning in assessment.warnings) == tuple(
        sorted(warning.code.value for warning in assessment.warnings)
    )
    with pytest.raises(FrozenInstanceError):
        assessment.eligible = True
