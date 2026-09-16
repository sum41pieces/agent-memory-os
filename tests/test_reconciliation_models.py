from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
import hashlib
import json
from types import MappingProxyType

import pytest

import agent_memory_os.reconcile as reconciliation_package
import agent_memory_os.reconcile.models as reconciliation_models
from agent_memory_os.reconcile.models import (
    CandidateStatusHint,
    MemoryCandidate,
    ReconciliationInputError,
    ReconciliationStatus,
    ReconciliationWarning,
    RelationType,
    ResolutionMethod,
    SourceType,
    UnresolvedCandidate,
    WarningCode,
)
from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.evidence.models import EvidenceValue
from agent_memory_os.reconcile.rules import (
    canonical_json_bytes,
    canonical_typed_value,
    group_candidates,
    make_fact_id,
)

from reconciliation_helpers import (
    CAPTURED_AT,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    make_snapshot,
    make_source_of_truth_candidate,
    unknown,
)


def test_base_fact_is_independent_frozen_internal_stage_model() -> None:
    assert hasattr(reconciliation_models, "BaseFact")
    base_fact_type = reconciliation_models.BaseFact

    with pytest.raises(TypeError, match="BaseFact|reconciliation"):
        base_fact_type()
    assert not hasattr(reconciliation_package, "BaseFact")


class _EvilString(str):
    def strip(self, chars=None):
        return self

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile string equality dispatched")

    __hash__ = str.__hash__


class _MutableInt(int):
    pass


class _MutableFloat(float):
    pass


class _SingleTraversalMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.iterations = 0
        self.reads: dict[str, int] = {}

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        yield "safe" if self.iterations == 1 else "changed"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        self.reads[key] = self.reads.get(key, 0) + 1
        if key != "safe" or self.reads[key] != 1:
            raise RuntimeError("mapping value read more than once")
        return ["value"]


class _ExplodingMapping(Mapping[str, object]):
    def __init__(self, explode_on: str) -> None:
        self.explode_on = explode_on

    def __iter__(self) -> Iterator[str]:
        if self.explode_on == "iteration":
            raise RuntimeError("iteration exploded")
        yield "safe"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        if self.explode_on == "lookup":
            raise RuntimeError("lookup exploded")
        return "value"


@pytest.mark.parametrize(
    ("field_name", "changes"),
    (
        ("candidate_id", {"candidate_id": _EvilString("candidate")}),
        ("subject", {"subject": _EvilString("project")}),
        ("predicate", {"predicate": _EvilString("setting")}),
        ("source_ref", {"source_ref": _EvilString("synthetic:source")}),
        (
            "value.source",
            {
                "value": EvidenceValue.known(
                    "value",
                    source=_EvilString("synthetic:value"),
                )
            },
        ),
        (
            "value.reason",
            {
                "value": EvidenceValue.unknown(
                    reason=_EvilString("not known"),
                    source="synthetic:value",
                )
            },
        ),
        ("value", {"value": known(_EvilString("value"))}),
        (
            "observed_at",
            {"observed_at": known(_EvilString(CAPTURED_AT))},
        ),
        (
            "supersedes",
            {"supersedes": known((_EvilString("old"),))},
        ),
        ("metadata", {"metadata": {_EvilString("key"): "value"}}),
        ("metadata", {"metadata": {"key": _EvilString("value")}}),
    ),
    ids=(
        "candidate-id",
        "subject",
        "predicate",
        "source-ref",
        "evidence-source",
        "evidence-reason",
        "known-string-value",
        "timestamp-value",
        "supersedes-id",
        "metadata-key",
        "metadata-value",
    ),
)
def test_memory_candidate_rejects_string_subclasses(
    field_name: str,
    changes: dict[str, object],
) -> None:
    with pytest.raises(ReconciliationInputError, match=field_name):
        make_candidate(**changes)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("subject", _EvilString("project")),
        (
            "value",
            EvidenceValue.known(
                _EvilString("value"),
                source="synthetic:value",
            ),
        ),
        (
            "value",
            EvidenceValue.known(
                "value",
                source=_EvilString("synthetic:value"),
            ),
        ),
    ),
    ids=("identity", "known-value", "evidence-source"),
)
def test_candidate_group_rejects_tampered_string_subclass_before_dispatch(
    field_name: str,
    value: object,
) -> None:
    candidate = make_candidate(candidate_id="candidate")
    object.__setattr__(candidate, field_name, value)

    with pytest.raises(ReconciliationInputError, match="exact string"):
        group_candidates("synthetic-project", (candidate,))


def test_candidate_group_rejects_project_id_string_subclass() -> None:
    with pytest.raises(ReconciliationInputError, match="project_id"):
        group_candidates(
            _EvilString("synthetic-project"),
            (make_candidate(candidate_id="candidate"),),
        )


def test_memory_candidate_rejects_missing_source_ref() -> None:
    with pytest.raises(ReconciliationInputError, match="source_ref"):
        make_candidate(candidate_id="c1", source_ref="")


def test_candidate_defensively_freezes_nested_values() -> None:
    raw_value = {"steps": ["one"]}
    raw_metadata = {"labels": ["current"]}
    candidate = make_candidate(
        candidate_id="c1",
        value=raw_value,
        metadata=raw_metadata,
    )
    before = candidate.to_dict()

    raw_value["steps"].append("two")
    raw_metadata["labels"].append("changed")

    assert candidate.to_dict() == before


@pytest.mark.parametrize(
    "invalid",
    (
        _MutableInt(1),
        _MutableFloat(1.0),
        CandidateStatusHint.CURRENT_FACT,
        {"nested": _MutableInt(1)},
    ),
)
def test_memory_candidate_rejects_non_exact_json_scalars(invalid: object) -> None:
    with pytest.raises(ReconciliationInputError, match="value"):
        make_candidate(value=invalid)


def test_memory_candidate_rejects_float_subclass_confidence() -> None:
    with pytest.raises(ReconciliationInputError, match="confidence"):
        make_candidate(confidence=known(_MutableFloat(0.5)))


def test_memory_candidate_snapshots_mapping_items_once() -> None:
    value = _SingleTraversalMapping()

    candidate = make_candidate(value=value)

    assert candidate.value.value == {"safe": ("value",)}
    assert value.iterations == 1
    assert value.reads == {"safe": 1}


@pytest.mark.parametrize("explode_on", ("iteration", "lookup"))
def test_memory_candidate_wraps_mapping_snapshot_failures(
    explode_on: str,
) -> None:
    with pytest.raises(ReconciliationInputError, match="value"):
        make_candidate(value=_ExplodingMapping(explode_on))


def test_memory_candidate_rejects_top_level_tuple_json_value() -> None:
    with pytest.raises(ReconciliationInputError, match="value"):
        make_candidate(value=("x",))


def test_memory_candidate_rejects_nested_tuple_in_metadata() -> None:
    with pytest.raises(ReconciliationInputError, match="metadata"):
        make_candidate(metadata={"labels": ("current",)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, invalid)
        for field in ("candidate_id", "subject", "predicate", "source_ref")
        for invalid in ("", "  ", "invalid\x00value")
    ],
)
def test_memory_candidate_rejects_invalid_identity_or_provenance(
    field: str,
    value: str,
) -> None:
    values = {field: value}

    with pytest.raises(ReconciliationInputError, match=field):
        make_candidate(**values)


def test_memory_candidate_rejects_non_enum_source_type() -> None:
    with pytest.raises(ReconciliationInputError, match="source_type"):
        make_candidate(source_type="CURRENT_EVIDENCE")


def test_memory_candidate_is_frozen() -> None:
    candidate = make_candidate()

    with pytest.raises(FrozenInstanceError):
        candidate.subject = "changed"


@pytest.mark.parametrize(
    "field",
    [
        "value",
        "status_hint",
        "observed_at",
        "valid_from",
        "valid_until",
        "confidence",
        "explicit_user_instruction",
        "supersedes",
        "deprecated",
        "metadata",
    ],
)
def test_memory_candidate_requires_evidence_value_fields(field: str) -> None:
    values = make_candidate().__dict__.copy()
    values[field] = "not evidence"

    with pytest.raises(ReconciliationInputError, match=field):
        MemoryCandidate(**values)


def test_memory_candidate_rejects_malformed_evidence_status() -> None:
    malformed = EvidenceValue(
        status="unknown",
        reason="not known",
        source="synthetic:malformed",
    )

    with pytest.raises(ReconciliationInputError, match="value"):
        make_candidate(value=malformed)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_memory_candidate_rejects_confidence_outside_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ReconciliationInputError, match="confidence"):
        make_candidate(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_memory_candidate_accepts_confidence_unit_interval_boundaries(
    confidence: float,
) -> None:
    assert make_candidate(confidence=confidence).confidence.value == confidence


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        (
            "value",
            EvidenceValue.known("value", source="synthetic\x00value"),
        ),
        (
            "valid_until",
            EvidenceValue.unknown(
                reason="no expiry",
                source="synthetic\x00until",
            ),
        ),
        (
            "metadata",
            EvidenceValue.known({}, source="synthetic\x00metadata"),
        ),
    ],
)
def test_memory_candidate_rejects_nul_in_field_evidence_source(
    field: str,
    malformed: EvidenceValue,
) -> None:
    with pytest.raises(ReconciliationInputError, match=field):
        make_candidate(**{field: malformed})


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("value", 123),
        ("observed_at", "  "),
        ("metadata", "invalid\x00reason"),
    ],
)
def test_memory_candidate_rejects_invalid_non_known_evidence_reason(
    field: str,
    reason: object,
) -> None:
    malformed = EvidenceValue(
        status=EvidenceStatus.UNKNOWN,
        reason=reason,
        source="synthetic:malformed",
    )

    with pytest.raises(ReconciliationInputError, match=field):
        make_candidate(**{field: malformed})


def test_memory_candidate_preserves_valid_evidence_text_exactly() -> None:
    value = known("value", source="  exact:value-source  ")
    valid_until = EvidenceValue.unknown(
        reason="  no exact expiry  ",
        source="  exact:until-source  ",
    )

    candidate = make_candidate(value=value, valid_until=valid_until)

    assert candidate.value.source == "  exact:value-source  "
    assert candidate.valid_until.source == "  exact:until-source  "
    assert candidate.valid_until.reason == "  no exact expiry  "


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("value", {"unsupported"}),
        ("value", {1: "non-string key"}),
        ("value", float("inf")),
        ("status_hint", "CURRENT_FACT"),
        ("observed_at", 1),
        ("valid_from", False),
        ("valid_until", 1.0),
        ("confidence", 1),
        ("explicit_user_instruction", 1),
        ("supersedes", ("old", 1)),
        ("deprecated", 0),
        ("metadata", ["not", "a", "mapping"]),
        ("metadata", {"bad": {1: "non-string key"}}),
    ],
)
def test_memory_candidate_rejects_invalid_known_runtime_types(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(ReconciliationInputError, match=field):
        make_candidate(**{field: known(invalid)})


def test_candidate_frozen_containers_cannot_be_mutated() -> None:
    candidate = make_candidate(
        value={"outer": [{"inner": ["one"]}]},
        metadata={"labels": ["current"]},
    )

    assert isinstance(candidate.value.value, MappingProxyType)
    assert candidate.value.value["outer"][0]["inner"] == ("one",)
    assert isinstance(candidate.metadata.value, MappingProxyType)
    with pytest.raises(TypeError):
        candidate.value.value["added"] = "blocked"


def test_candidate_defensively_copies_supersedes_as_tuple() -> None:
    raw_supersedes = ["old"]
    candidate = make_candidate(supersedes=raw_supersedes)
    before = candidate.to_dict()

    raw_supersedes.append("older")

    assert candidate.supersedes.value == ("old",)
    assert candidate.to_dict() == before


def test_candidate_serialization_restores_json_containers() -> None:
    candidate = make_candidate(
        candidate_id="c1",
        value={"z": ["CURRENT_FACT"], "a": True},
        supersedes=["old"],
    )

    serialized = candidate.to_dict()

    assert serialized["source_type"] == "CURRENT_EVIDENCE"
    assert serialized["value"]["value"] == {
        "a": True,
        "z": ["CURRENT_FACT"],
    }
    assert serialized["supersedes"]["value"] == ["old"]


def test_candidate_preserves_evidence_sources_and_reasons() -> None:
    value = known("value", source="exact:value-source")
    valid_until = unknown("no exact expiry", source="exact:until-source")

    serialized = make_candidate(value=value, valid_until=valid_until).to_dict()

    assert serialized["value"]["source"] == "exact:value-source"
    assert serialized["valid_until"] == {
        "status": "unknown",
        "source": "exact:until-source",
        "reason": "no exact expiry",
    }


def test_candidate_helpers_apply_exact_default_semantics() -> None:
    current = make_current_candidate()
    historical = make_historical_candidate()
    source_of_truth = make_source_of_truth_candidate()

    assert current.observed_at.value == CAPTURED_AT
    assert current.valid_from.value == CAPTURED_AT
    assert current.valid_until.status is EvidenceStatus.UNKNOWN
    assert current.status_hint.value is CandidateStatusHint.CURRENT_FACT
    assert historical.source_type is SourceType.HISTORICAL_MEMORY
    assert historical.status_hint.value is CandidateStatusHint.HISTORICAL
    assert source_of_truth.source_type is SourceType.USER_EXPLICIT
    assert source_of_truth.status_hint.value is CandidateStatusHint.SOURCE_OF_TRUTH
    assert source_of_truth.explicit_user_instruction.value is True


def test_reconciliation_status_has_exact_wire_values() -> None:
    assert {item.value for item in ReconciliationStatus} == {
        "ACTIVE",
        "SUPERSEDED",
        "CONFLICTED",
        "PENDING",
        "DEPRECATED",
    }


@pytest.mark.parametrize(
    ("enum_type", "expected"),
    [
        (
            SourceType,
            {
                "USER_EXPLICIT",
                "CURRENT_EVIDENCE",
                "PROJECT_DOC",
                "PROJECT_CARD",
                "TEMPORAL_RECORD",
                "SESSION_LOG",
                "HISTORICAL_MEMORY",
            },
        ),
        (
            CandidateStatusHint,
            {
                "CURRENT_FACT",
                "HISTORICAL",
                "PLAN",
                "HYPOTHESIS",
                "SOURCE_OF_TRUTH",
                "DEPRECATED",
            },
        ),
        (RelationType, {"SUPERSEDES", "CONFLICTS"}),
        (
            ResolutionMethod,
            {
                "DIRECT_CURRENT",
                "SAME_VALUE_MERGE",
                "SAME_VALUE_REACTIVATION",
                "EXPLICIT_SOURCE_OF_TRUTH",
                "CURRENT_EVIDENCE_OVER_HISTORICAL",
                "EXPLICIT_SUPERSEDES",
                "UNRESOLVED_CONFLICT",
                "PENDING_SEMANTICS",
                "TEMPORAL_PENDING",
                "INSUFFICIENT_EVIDENCE",
                "EXPLICIT_DEPRECATION",
            },
        ),
        (
            WarningCode,
            {
                "EVIDENCE_UNKNOWN",
                "EVIDENCE_UNAVAILABLE",
                "INVALID_TEMPORAL_ORDER",
                "FUTURE_CLOCK_SKEW",
                "FUTURE_VALIDITY",
                "EXPIRED_WITHOUT_REPLACEMENT",
                "INSUFFICIENT_CONFIDENCE",
                "UNRESOLVED_TEMPORAL_COMPARISON",
                "CONTRADICTORY_SUPERSEDES",
            },
        ),
    ],
)
def test_phase_two_enums_have_exact_wire_values(enum_type, expected) -> None:
    assert {item.value for item in enum_type} == expected


def test_warning_is_frozen() -> None:
    warning = ReconciliationWarning(
        code=WarningCode.INSUFFICIENT_CONFIDENCE,
        message="below threshold",
        candidate_ids=("c1",),
        evidence_refs=("synthetic:confidence",),
        requires_human_review=False,
    )
    with pytest.raises(FrozenInstanceError):
        warning.message = "changed"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"message": " "}, "message"),
        ({"candidate_ids": ()}, "candidate_ids"),
        ({"candidate_ids": ("",)}, "candidate_ids"),
        ({"evidence_refs": ()}, "evidence_refs"),
        ({"evidence_refs": ("synthetic:b", "synthetic:a")}, "evidence_refs"),
        ({"candidate_ids": ("c1", "c1")}, "candidate_ids"),
    ],
)
def test_warning_rejects_invalid_message_or_provenance(overrides, field) -> None:
    values = {
        "code": WarningCode.INSUFFICIENT_CONFIDENCE,
        "message": "below threshold",
        "candidate_ids": ("c1",),
        "evidence_refs": ("synthetic:confidence",),
        "requires_human_review": False,
    }
    values.update(overrides)
    with pytest.raises(ReconciliationInputError, match=field):
        ReconciliationWarning(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", ""),
        ("subject", " "),
        ("predicate", ""),
        ("reason", ""),
        ("source_ref", ""),
        ("field_source", ""),
        ("related_fact_id", ""),
    ],
)
def test_unresolved_candidate_rejects_empty_identifiers(field, value) -> None:
    values = {
        "candidate_id": "c1",
        "subject": "project",
        "predicate": "setting",
        "evidence_status": EvidenceStatus.UNKNOWN,
        "reason": "value is unknown",
        "source_type": SourceType.CURRENT_EVIDENCE,
        "source_ref": "synthetic:c1",
        "field_source": "synthetic:c1:value",
        "related_fact_id": "unresolved:v1:abc",
    }
    values[field] = value
    with pytest.raises(ReconciliationInputError, match=field):
        UnresolvedCandidate(**values)


def test_unresolved_candidate_is_frozen() -> None:
    unresolved = UnresolvedCandidate(
        candidate_id="c1",
        subject="project",
        predicate="setting",
        evidence_status=EvidenceStatus.UNAVAILABLE,
        reason="source is unavailable",
        source_type=SourceType.CURRENT_EVIDENCE,
        source_ref="synthetic:c1",
        field_source="synthetic:c1:value",
        related_fact_id="unresolved:v1:abc",
    )
    with pytest.raises(FrozenInstanceError):
        unresolved.reason = "changed"


@pytest.mark.parametrize(
    "evidence_status",
    ["unknown", [], EvidenceStatus.KNOWN],
)
def test_unresolved_candidate_rejects_invalid_evidence_status(
    evidence_status,
) -> None:
    with pytest.raises(ReconciliationInputError, match="evidence_status"):
        UnresolvedCandidate(
            candidate_id="c1",
            subject="project",
            predicate="setting",
            evidence_status=evidence_status,
            reason="value is unknown",
            source_type=SourceType.CURRENT_EVIDENCE,
            source_ref="synthetic:c1",
            field_source="synthetic:c1:value",
            related_fact_id="unresolved:v1:abc",
        )


def test_reconciliation_snapshot_helper_is_complete_and_synthetic() -> None:
    snapshot = make_snapshot()

    assert snapshot.project_id.value == "synthetic-project"
    assert snapshot.repository.path.value == r"D:\synthetic\project"
    assert snapshot.captured_at.value == "2026-09-14T00:00:00+00:00"
    guard = snapshot.collector.shadow_guard.value
    assert guard is not None
    assert guard.head_unchanged.value is True
    assert guard.index_unchanged.value is True
    assert guard.status_unchanged.value is True
    assert guard.tracked_content_unchanged.value is True


def test_fact_identity_distinguishes_json_scalar_types() -> None:
    fact_ids = {
        make_fact_id("synthetic-project", "project", "setting", value)
        for value in ["1", 1, 1.0, True]
    }

    assert len(fact_ids) == 4


def test_canonical_value_preserves_generic_windows_string() -> None:
    windows_value = canonical_typed_value(r"C:\Foo\Bar")

    assert json.loads(windows_value)["value"] == r"C:\Foo\Bar"
    assert windows_value != canonical_typed_value("c:/foo/bar")


def test_canonical_value_sorts_keys_recursively() -> None:
    left = {"z": {"second": 2, "first": 1}, "a": True}
    right = {"a": True, "z": {"first": 1, "second": 2}}

    assert canonical_typed_value(left) == canonical_typed_value(right)


def test_canonical_value_preserves_list_order() -> None:
    assert canonical_typed_value(["first", "second"]) != canonical_typed_value(
        ["second", "first"]
    )


def test_canonical_value_preserves_unicode_without_ascii_escaping() -> None:
    canonical = canonical_typed_value("雪")

    assert "雪" in canonical
    assert "\\u96ea" not in canonical


def test_canonical_value_accepts_frozen_json_containers() -> None:
    frozen = MappingProxyType({"items": ("one", 2)})

    assert canonical_typed_value(frozen) == canonical_typed_value(
        {"items": ["one", 2]}
    )


@pytest.mark.parametrize(
    "invalid",
    (
        _MutableInt(1),
        _MutableFloat(1.0),
        _EvilString("value"),
        CandidateStatusHint.CURRENT_FACT,
    ),
)
def test_canonical_value_rejects_non_exact_json_scalars(invalid: object) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="unsupported KNOWN JSON value",
    ):
        canonical_typed_value(invalid)


def test_canonical_value_snapshots_mapping_items_once() -> None:
    value = _SingleTraversalMapping()

    canonical = canonical_typed_value(value)

    assert canonical == canonical_typed_value({"safe": ["value"]})
    assert value.iterations == 1
    assert value.reads == {"safe": 1}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param({"item"}, id="set"),
        pytest.param(b"item", id="bytes"),
        pytest.param(object(), id="custom-object"),
        pytest.param({1: "item"}, id="non-string-mapping-key"),
        pytest.param({"nested": {1: "item"}}, id="nested-non-string-mapping-key"),
    ],
)
def test_canonical_value_rejects_unsupported_known_json_values(value) -> None:
    with pytest.raises(
        ReconciliationInputError,
        match="unsupported KNOWN JSON value",
    ):
        canonical_typed_value(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_value_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ReconciliationInputError, match="float must be finite"):
        canonical_typed_value(value)


def test_canonical_json_bytes_are_sorted_compact_utf8() -> None:
    assert canonical_json_bytes({"z": "雪", "a": 1}) == (
        '{"a":1,"z":"雪"}'.encode("utf-8")
    )


def test_fact_identity_hashes_exact_versioned_payload() -> None:
    identity = {
        "namespace": "fact:v1",
        "project_id": "project",
        "subject": "service",
        "predicate": "port",
        "canonical_typed_value": {"type": "integer", "value": "8080"},
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert make_fact_id("project", "service", "port", 8080) == (
        f"fact:v1:{expected_digest}"
    )
