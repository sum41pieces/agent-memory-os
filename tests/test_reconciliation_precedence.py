from dataclasses import FrozenInstanceError
from enum import Enum

import pytest

from agent_memory_os.reconcile.models import (
    ReconciliationInputError,
    ReconciliationPolicy,
    SourceType,
)
from agent_memory_os.reconcile.precedence import (
    DEFAULT_MAX_FUTURE_CLOCK_SKEW_SECONDS,
    DEFAULT_SOURCE_PRECEDENCE,
    validate_policy,
)

from reconciliation_helpers import default_policy


EXPECTED_SOURCE_PRECEDENCE = (
    (SourceType.USER_EXPLICIT, 7),
    (SourceType.CURRENT_EVIDENCE, 6),
    (SourceType.PROJECT_DOC, 5),
    (SourceType.PROJECT_CARD, 4),
    (SourceType.TEMPORAL_RECORD, 3),
    (SourceType.SESSION_LOG, 2),
    (SourceType.HISTORICAL_MEMORY, 1),
)


class ForeignSourceType(str, Enum):
    EXTERNAL = "EXTERNAL"


def test_default_policy_has_exact_conservative_configuration() -> None:
    policy = ReconciliationPolicy()

    assert DEFAULT_SOURCE_PRECEDENCE == EXPECTED_SOURCE_PRECEDENCE
    assert policy.source_precedence == EXPECTED_SOURCE_PRECEDENCE
    assert policy.active_confidence_threshold == 0.5
    assert policy.allow_explicit_source_of_truth_override is True
    assert policy.allow_current_evidence_over_historical is True
    assert policy.require_explicit_hint_for_user_override is True
    assert policy.conflict_on_equal_precedence_disagreement is True
    assert DEFAULT_MAX_FUTURE_CLOCK_SKEW_SECONDS == 300
    assert policy.max_future_clock_skew_seconds == 300


def test_policy_copies_precedence_and_defaults_to_300_second_skew() -> None:
    source = [list(entry) for entry in DEFAULT_SOURCE_PRECEDENCE]

    policy = ReconciliationPolicy(source_precedence=source)
    before = policy.source_precedence
    source[0][1] = 99
    source.append([SourceType.PROJECT_DOC, 98])

    assert policy.source_precedence == before
    assert policy.source_precedence == EXPECTED_SOURCE_PRECEDENCE
    assert isinstance(policy.source_precedence, tuple)
    assert all(isinstance(entry, tuple) for entry in policy.source_precedence)
    assert policy.max_future_clock_skew_seconds == 300


def test_policy_is_frozen_and_precedence_cannot_be_extended() -> None:
    policy = ReconciliationPolicy()

    with pytest.raises(FrozenInstanceError):
        policy.active_confidence_threshold = 0.75
    with pytest.raises(AttributeError):
        policy.source_precedence.append((SourceType.PROJECT_DOC, 99))


def test_rank_for_returns_each_configured_rank() -> None:
    policy = ReconciliationPolicy()

    assert tuple(
        (source_type, policy.rank_for(source_type))
        for source_type in SourceType
    ) == EXPECTED_SOURCE_PRECEDENCE


@pytest.mark.parametrize(
    "source_type",
    ["CURRENT_EVIDENCE", ForeignSourceType.EXTERNAL, None, 1],
)
def test_rank_for_rejects_invalid_runtime_source_type(source_type: object) -> None:
    with pytest.raises(ReconciliationInputError, match="source_type"):
        ReconciliationPolicy().rank_for(source_type)


def test_validate_policy_returns_the_same_valid_frozen_policy() -> None:
    policy = ReconciliationPolicy()

    assert validate_policy(policy) is policy


def test_policy_rejects_duplicate_source_types() -> None:
    precedence = list(DEFAULT_SOURCE_PRECEDENCE)
    precedence[-1] = (SourceType.USER_EXPLICIT, 1)

    with pytest.raises(ReconciliationInputError, match="duplicate source types"):
        default_policy(source_precedence=precedence)


def test_policy_rejects_duplicate_ranks() -> None:
    precedence = list(DEFAULT_SOURCE_PRECEDENCE)
    precedence[-1] = (SourceType.HISTORICAL_MEMORY, 2)

    with pytest.raises(ReconciliationInputError, match="duplicate ranks"):
        default_policy(source_precedence=precedence)


def test_policy_rejects_missing_source_type() -> None:
    precedence = DEFAULT_SOURCE_PRECEDENCE[:-1]

    with pytest.raises(ReconciliationInputError, match="missing SourceType"):
        default_policy(source_precedence=precedence)


@pytest.mark.parametrize(
    "invalid_source",
    ["USER_EXPLICIT", ForeignSourceType.EXTERNAL],
)
def test_policy_rejects_raw_or_foreign_source_enum(
    invalid_source: object,
) -> None:
    precedence = list(DEFAULT_SOURCE_PRECEDENCE)
    precedence[0] = (invalid_source, 7)

    with pytest.raises(ReconciliationInputError, match="source_precedence"):
        default_policy(source_precedence=precedence)


@pytest.mark.parametrize("rank", [True, False, 1.5, "7", None])
def test_policy_rejects_boolean_or_noninteger_rank(rank: object) -> None:
    precedence = list(DEFAULT_SOURCE_PRECEDENCE)
    precedence[0] = (SourceType.USER_EXPLICIT, rank)

    with pytest.raises(ReconciliationInputError, match="source_precedence"):
        default_policy(source_precedence=precedence)


@pytest.mark.parametrize(
    "precedence",
    [None, "not-a-table", ((SourceType.USER_EXPLICIT,),)],
)
def test_policy_rejects_malformed_precedence_table(precedence: object) -> None:
    with pytest.raises(ReconciliationInputError, match="source_precedence"):
        default_policy(source_precedence=precedence)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_policy_rejects_invalid_clock_skew(value: object) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="max_future_clock_skew_seconds",
    ):
        default_policy(max_future_clock_skew_seconds=value)


@pytest.mark.parametrize("value", [False, -1, 1.5, "300", None])
def test_policy_rejects_other_invalid_clock_skew(value: object) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="max_future_clock_skew_seconds",
    ):
        default_policy(max_future_clock_skew_seconds=value)


def test_policy_accepts_zero_clock_skew() -> None:
    policy = default_policy(max_future_clock_skew_seconds=0)

    assert policy.max_future_clock_skew_seconds == 0


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        1.01,
        10**1000,
        -(10**1000),
    ],
)
def test_policy_rejects_nonfinite_or_out_of_range_threshold(
    value: float | int,
) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="active_confidence_threshold",
    ):
        default_policy(active_confidence_threshold=value)


@pytest.mark.parametrize("value", [True, False, "0.5", None, object()])
def test_policy_rejects_boolean_or_nonnumeric_threshold(value: object) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="active_confidence_threshold",
    ):
        default_policy(active_confidence_threshold=value)


@pytest.mark.parametrize("value", [0.0, 1.0, 0, 1])
def test_policy_accepts_numeric_threshold_boundaries(value: float | int) -> None:
    policy = default_policy(active_confidence_threshold=value)

    assert policy.active_confidence_threshold == value


@pytest.mark.parametrize(
    "field",
    [
        "allow_explicit_source_of_truth_override",
        "allow_current_evidence_over_historical",
        "require_explicit_hint_for_user_override",
        "conflict_on_equal_precedence_disagreement",
    ],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_policy_flags_require_strict_booleans(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ReconciliationInputError, match=field):
        default_policy(**{field: value})
