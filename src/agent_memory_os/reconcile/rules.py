"""Pure canonicalization and identity rules for reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import math
import re
import secrets
from typing import Callable, TypeAlias

from agent_memory_os.evidence.models import EvidenceStatus, EvidenceValue
from agent_memory_os.reconcile.precedence import (
    is_direct_current_evidence,
    is_strict_historical_memory,
    is_valid_source_of_truth,
)
from agent_memory_os.reconcile.models import (
    _FACT_DECISION_CONSTRUCTION_TOKEN,
    _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN,
    _ReconciliationCohort,
    _ReconciliationStage,
    _claim_canonical_stage_builders,
    _normalize_relation_provenance,
    _snapshot_json_mapping_items,
    _strict_evidence_signature,
    _validate_stage_authorization,
    _validate_reconciliation_cohort,
    _validate_required_text,
    CandidateStatusHint,
    FactDecision,
    LineageOutcome,
    MemoryCandidate,
    ReconciliationInputError,
    ReconciliationPolicy,
    ReconciliationStatus,
    ReconciliationWarning,
    RelationType,
    ResolutionMethod,
    SameValueLineage,
    SourceType,
    TemporalAssessment,
    UnresolvedCandidate,
    WarningCode,
)


(
    _group_stage_builder,
    _classified_stage_builder,
    _coordinated_stage_builder,
) = _claim_canonical_stage_builders(__name__)
del _claim_canonical_stage_builders


JsonValue: TypeAlias = (
    str | bool | int | float | list["JsonValue"] | dict[str, "JsonValue"]
)
FrozenJsonValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | tuple["FrozenJsonValue", ...]
    | Mapping[str, "FrozenJsonValue"]
)

_SUPPORTED_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?"
    r"(?:Z|[+-][0-9]{2}:[0-9]{2})"
)


def _parse_timestamp_text(raw: object, field_name: str) -> datetime:
    if type(raw) is not str:
        raise ReconciliationInputError(
            f"{field_name} must be an ISO-8601 string"
        )
    if _SUPPORTED_TIMESTAMP.fullmatch(raw) is None:
        raise ReconciliationInputError(
            f"{field_name} is not valid ISO-8601"
        )
    normalized = f"{raw[:-1]}+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ReconciliationInputError(
            f"{field_name} is not valid ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReconciliationInputError(
            f"{field_name} requires an explicit timezone"
        )
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError as error:
        raise ReconciliationInputError(
            f"{field_name} cannot be normalized to UTC"
        ) from error


def parse_known_timestamp(
    value: EvidenceValue[str],
    field_name: str,
) -> datetime | None:
    """Strictly parse a KNOWN timestamp and normalize it to UTC."""

    if not isinstance(value, EvidenceValue):
        raise ReconciliationInputError(
            f"{field_name} must be an EvidenceValue"
        )
    if not isinstance(value.status, EvidenceStatus):
        raise ReconciliationInputError(
            f"{field_name} must have a valid EvidenceStatus"
        )
    if value.status is not EvidenceStatus.KNOWN:
        return None
    return _parse_timestamp_text(value.value, field_name)


def parse_reconciliation_clock(raw: object) -> datetime:
    """Strictly parse the injected reconciliation clock and normalize it to UTC."""

    return _parse_timestamp_text(raw, "reconciliation clock")


def _declares_current_semantics(candidate: MemoryCandidate) -> bool:
    return (
        candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value
        in (
            CandidateStatusHint.CURRENT_FACT,
            CandidateStatusHint.SOURCE_OF_TRUTH,
        )
    )


def _make_warning(
    code: WarningCode,
    message: str,
    candidate_ids: Sequence[str],
    evidence_refs: Sequence[str],
    *,
    requires_human_review: bool,
) -> ReconciliationWarning:
    """Build every rule warning with canonical provenance ordering."""

    return ReconciliationWarning(
        code=code,
        message=message,
        candidate_ids=tuple(sorted(set(candidate_ids))),
        evidence_refs=tuple(sorted(set(evidence_refs))),
        requires_human_review=requires_human_review,
    )


def _temporal_warning(
    code: WarningCode,
    message: str,
    candidate: MemoryCandidate,
    evidence_refs: tuple[str, ...],
    *,
    requires_human_review: bool,
) -> ReconciliationWarning:
    return _make_warning(
        code,
        message,
        (candidate.candidate_id,),
        evidence_refs,
        requires_human_review=requires_human_review,
    )


def _temporal_warning_sort_key(
    warning: ReconciliationWarning,
) -> tuple[object, ...]:
    return (
        warning.code.value,
        warning.candidate_ids,
        warning.message,
        warning.evidence_refs,
        warning.requires_human_review,
    )


def _exceeds_future_skew(
    observed_at: datetime,
    reconciled_at: datetime,
    allowed_seconds: int,
) -> bool:
    difference = observed_at - reconciled_at
    if difference.days < 0:
        return False
    whole_seconds = difference.days * 86_400 + difference.seconds
    return whole_seconds > allowed_seconds or (
        whole_seconds == allowed_seconds and difference.microseconds > 0
    )


def assess_temporal(
    candidate: MemoryCandidate,
    now: datetime,
    policy: ReconciliationPolicy,
) -> TemporalAssessment:
    """Parse and assess one candidate against an injected reconciliation time."""

    if not isinstance(candidate, MemoryCandidate):
        raise ReconciliationInputError("candidate must be a MemoryCandidate")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ReconciliationInputError("now must be a timezone-aware datetime")
    if not isinstance(policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")
    try:
        reconciled_at = now.astimezone(timezone.utc)
    except OverflowError as error:
        raise ReconciliationInputError("now cannot be normalized to UTC") from error

    # Parse every KNOWN field before interpreting any semantic time condition.
    observed_at = parse_known_timestamp(candidate.observed_at, "observed_at")
    valid_from = parse_known_timestamp(candidate.valid_from, "valid_from")
    valid_until = parse_known_timestamp(candidate.valid_until, "valid_until")

    warnings: list[ReconciliationWarning] = []
    current_semantics = _declares_current_semantics(candidate)

    if (
        valid_from is not None
        and valid_until is not None
        and valid_until < valid_from
    ):
        warnings.append(
            _temporal_warning(
                WarningCode.INVALID_TEMPORAL_ORDER,
                "valid_until is earlier than valid_from",
                candidate,
                (candidate.valid_from.source, candidate.valid_until.source),
                requires_human_review=True,
            )
        )

    if observed_at is not None and _exceeds_future_skew(
        observed_at,
        reconciled_at,
        policy.max_future_clock_skew_seconds,
    ):
        warnings.append(
            _temporal_warning(
                WarningCode.FUTURE_CLOCK_SKEW,
                "observed_at exceeds the allowed future clock-skew boundary",
                candidate,
                (candidate.observed_at.source,),
                requires_human_review=True,
            )
        )

    if valid_from is not None and valid_from > reconciled_at:
        warnings.append(
            _temporal_warning(
                WarningCode.FUTURE_VALIDITY,
                "valid_from is later than the reconciliation time",
                candidate,
                (candidate.valid_from.source,),
                requires_human_review=current_semantics,
            )
        )

    if valid_until is not None and valid_until <= reconciled_at:
        warnings.append(
            _temporal_warning(
                WarningCode.EXPIRED_WITHOUT_REPLACEMENT,
                "valid_until has passed without an eligible replacement",
                candidate,
                (candidate.valid_until.source,),
                requires_human_review=current_semantics,
            )
        )

    sorted_warnings = tuple(sorted(warnings, key=_temporal_warning_sort_key))
    pending_priority = (
        WarningCode.INVALID_TEMPORAL_ORDER,
        WarningCode.FUTURE_CLOCK_SKEW,
        WarningCode.FUTURE_VALIDITY,
        WarningCode.EXPIRED_WITHOUT_REPLACEMENT,
    )
    emitted_codes = {warning.code for warning in sorted_warnings}
    pending_reason = next(
        (code.value for code in pending_priority if code in emitted_codes),
        None,
    )
    return TemporalAssessment(
        candidate_id=candidate.candidate_id,
        assessed_at=reconciled_at,
        max_future_clock_skew_seconds=(
            policy.max_future_clock_skew_seconds
        ),
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        eligible=not sorted_warnings,
        pending_reason=pending_reason,
        warnings=sorted_warnings,
    )


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to deterministic UTF-8 bytes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _typed_node(value: JsonValue | FrozenJsonValue) -> object:
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ReconciliationInputError("float must be finite")
        return {"type": "float", "value": value.hex()}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) in (list, tuple):
        try:
            items = tuple(value)
        except Exception as error:
            raise ReconciliationInputError(
                "unsupported KNOWN JSON value"
            ) from error
        return {"type": "list", "value": [_typed_node(item) for item in items]}
    if isinstance(value, Mapping):
        try:
            items = _snapshot_json_mapping_items(value)
        except ReconciliationInputError as error:
            raise ReconciliationInputError(
                "unsupported KNOWN JSON value"
            ) from error
        return {
            "type": "mapping",
            "value": {
                key: _typed_node(item_value)
                for key, item_value in sorted(items, key=lambda item: item[0])
            },
        }
    raise ReconciliationInputError("unsupported KNOWN JSON value")


def canonical_typed_value(value: JsonValue | FrozenJsonValue) -> str:
    """Return a canonical JSON string that preserves every JSON type."""
    return canonical_json_bytes(_typed_node(value)).decode("utf-8")


def _fact_id_from_canonical(
    project_id: str,
    subject: str,
    predicate: str,
    value: JsonValue | FrozenJsonValue,
) -> str:
    identity = {
        "namespace": "fact:v1",
        "project_id": project_id,
        "subject": subject,
        "predicate": predicate,
        "canonical_typed_value": _typed_node(value),
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return f"fact:v1:{digest}"


def make_fact_id(
    project_id: str,
    subject: str,
    predicate: str,
    value: JsonValue | FrozenJsonValue,
) -> str:
    """Build a stable fact identity from only its canonical semantic fields."""

    return _fact_id_from_canonical(project_id, subject, predicate, value)


def make_relation_id(
    relation_type: RelationType,
    from_fact_id: str,
    to_fact_id: str,
) -> str:
    """Build a stable relation identity from its type and endpoints."""
    if type(relation_type) is not RelationType:
        raise ReconciliationInputError(
            "relation_type must be a RelationType"
        )
    _validate_required_text(from_fact_id, "from_fact_id")
    _validate_required_text(to_fact_id, "to_fact_id")
    if from_fact_id == to_fact_id:
        raise ReconciliationInputError("relation cannot be a fact self-edge")
    identity = {
        "namespace": "relation:v1",
        "relation_type": relation_type.value,
        "from_fact_id": from_fact_id,
        "to_fact_id": to_fact_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return f"relation:v1:{digest}"


def _require_project_id(project_id: object) -> str:
    if (
        type(project_id) is not str
        or not project_id.strip()
        or "\x00" in project_id
    ):
        raise ReconciliationInputError(
            "project_id must be a non-empty, NUL-free string"
        )
    return project_id


def _make_unresolved_id(
    project_id: str,
    candidate: MemoryCandidate,
) -> str:
    identity = {
        "namespace": "unresolved:v1",
        "project_id": project_id,
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "candidate_id": candidate.candidate_id,
        "evidence_status": candidate.value.status.value,
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return f"unresolved:v1:{digest}"


def _json_has_only_exact_strings(value: object) -> bool:
    if isinstance(value, str):
        return type(value) is str
    if type(value) is tuple:
        return all(_json_has_only_exact_strings(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            type(key) is str and _json_has_only_exact_strings(item)
            for key, item in value.items()
        )
    return True


def _candidate_has_only_exact_strings(candidate: MemoryCandidate) -> bool:
    if any(
        type(getattr(candidate, field_name)) is not str
        for field_name in (
            "candidate_id",
            "subject",
            "predicate",
            "source_ref",
        )
    ):
        return False
    evidence_field_names = (
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
    )
    for field_name in evidence_field_names:
        evidence = getattr(candidate, field_name)
        if (
            type(evidence) is not EvidenceValue
            or type(evidence.source) is not str
            or (
                evidence.reason is not None
                and type(evidence.reason) is not str
            )
        ):
            return False
    if any(
        evidence.status is EvidenceStatus.KNOWN
        and type(evidence.value) is not str
        for evidence in (
            candidate.observed_at,
            candidate.valid_from,
            candidate.valid_until,
        )
    ):
        return False
    if (
        candidate.supersedes.status is EvidenceStatus.KNOWN
        and (
            type(candidate.supersedes.value) is not tuple
            or any(
                type(candidate_id) is not str
                for candidate_id in candidate.supersedes.value
            )
        )
    ):
        return False
    if any(
        evidence.status is EvidenceStatus.KNOWN
        and not _json_has_only_exact_strings(evidence.value)
        for evidence in (candidate.value, candidate.metadata)
    ):
        return False
    return True


def _require_exact_candidate_strings(candidate: MemoryCandidate) -> None:
    if not _candidate_has_only_exact_strings(candidate):
        raise ReconciliationInputError(
            "candidate identity and provenance must use exact strings"
        )


def _canonical_candidate_payload(candidate: MemoryCandidate) -> dict[str, object]:
    if type(candidate) is not MemoryCandidate:
        raise ReconciliationInputError(
            "cohort candidates must be exact MemoryCandidate records"
        )
    try:
        candidate_id = object.__getattribute__(candidate, "candidate_id")
        subject = object.__getattribute__(candidate, "subject")
        predicate = object.__getattribute__(candidate, "predicate")
        source_type = object.__getattribute__(candidate, "source_type")
        source_ref = object.__getattribute__(candidate, "source_ref")
        for field_name, value in (
            ("candidate_id", candidate_id),
            ("subject", subject),
            ("predicate", predicate),
            ("source_ref", source_ref),
        ):
            _validate_required_text(value, field_name)
        if type(source_type) is not SourceType:
            raise ReconciliationInputError(
                "candidate source_type must be an exact SourceType"
            )
        evidence_kinds = (
            ("value", "json"),
            ("status_hint", "candidate_status"),
            ("observed_at", "string"),
            ("valid_from", "string"),
            ("valid_until", "string"),
            ("confidence", "float"),
            ("explicit_user_instruction", "bool"),
            ("supersedes", "string_tuple"),
            ("deprecated", "bool"),
            ("metadata", "metadata"),
        )
        evidence_payload = {
            field_name: _strict_evidence_signature(
                object.__getattribute__(candidate, field_name),
                field_name,
                known_kind,
            ).hex()
            for field_name, known_kind in evidence_kinds
        }
        return {
            "candidate_id": candidate_id,
            "subject": subject,
            "predicate": predicate,
            "source_type": source_type.value,
            "source_ref": source_ref,
            "evidence": evidence_payload,
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ReconciliationInputError(
            "cohort candidate payload is invalid"
        ) from error


def _candidate_id_namespace(
    project_id: str,
    candidate_ids: tuple[str, ...],
) -> str:
    _validate_required_text(project_id, "project_id")
    if type(candidate_ids) is not tuple:
        raise ReconciliationInputError(
            "candidate ID namespace requires an exact tuple"
        )
    for candidate_id in candidate_ids:
        _validate_required_text(candidate_id, "candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        duplicates = sorted(
            candidate_id
            for candidate_id in set(candidate_ids)
            if candidate_ids.count(candidate_id) > 1
        )
        raise ReconciliationInputError(
            f"duplicate candidate_id: {duplicates[0]}"
        )
    namespace_payload = {
        "namespace": "candidate-id-namespace:v1",
        "project_id": project_id,
        "candidate_ids": sorted(candidate_ids),
    }
    return (
        "candidate-id-namespace:v1:"
        + hashlib.sha256(canonical_json_bytes(namespace_payload)).hexdigest()
    )


def _predicate_membership(
    project_id: str,
    candidates: tuple[MemoryCandidate, ...],
) -> tuple[
    tuple[
        str,
        str,
        str,
        tuple[str, ...],
        tuple[str, ...],
        str,
    ],
    ...,
]:
    grouped: dict[tuple[str, str], dict[str, set[str]]] = {}
    for candidate in candidates:
        payload = _canonical_candidate_payload(candidate)
        subject = payload["subject"]
        predicate = payload["predicate"]
        if type(subject) is not str or type(predicate) is not str:
            raise ReconciliationInputError(
                "predicate membership identity is invalid"
            )
        fact_id = (
            _fact_id_from_canonical(
                project_id,
                subject,
                predicate,
                candidate.value.value,
            )
            if candidate.value.status is EvidenceStatus.KNOWN
            else _make_unresolved_id(project_id, candidate)
        )
        membership = grouped.setdefault(
            (subject, predicate),
            {"fact_ids": set(), "candidate_ids": set()},
        )
        membership["fact_ids"].add(fact_id)
        membership["candidate_ids"].add(candidate.candidate_id)
    return tuple(
        (
            project_id,
            subject,
            predicate,
            tuple(sorted(values["fact_ids"])),
            tuple(sorted(values["candidate_ids"])),
            _candidate_id_namespace(
                project_id,
                tuple(sorted(values["candidate_ids"])),
            ),
        )
        for (subject, predicate), values in sorted(grouped.items())
    )


def _make_invocation_cohort(
    project_id: str,
    candidates: tuple[MemoryCandidate, ...],
    snapshot_id: str | None,
) -> _ReconciliationCohort:
    payloads = tuple(
        _canonical_candidate_payload(candidate)
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    )
    payload_digest = hashlib.sha256(canonical_json_bytes(payloads)).hexdigest()
    resolved_snapshot_id = (
        f"snapshot:v1:{payload_digest}" if snapshot_id is None else snapshot_id
    )
    _validate_required_text(resolved_snapshot_id, "snapshot_id")
    candidate_namespace = _candidate_id_namespace(
        project_id,
        tuple(candidate.candidate_id for candidate in candidates),
    )
    return _ReconciliationCohort._from_payload(
        _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN,
        project_id=project_id,
        snapshot_id=resolved_snapshot_id,
        policy_fingerprint=None,
        reconciled_at=None,
        candidate_id_namespace=candidate_namespace,
        predicate_membership=_predicate_membership(project_id, candidates),
        invocation_token=secrets.token_hex(32),
    )


def _policy_fingerprint(policy: ReconciliationPolicy) -> str:
    canonical = _canonicalize_reconciliation_policy(policy, "cohort policy")
    payload = {
        "namespace": "reconciliation-policy:v1",
        "source_precedence": [
            [source_type.value, rank]
            for source_type, rank in canonical.source_precedence
        ],
        "active_confidence_threshold": float(
            canonical.active_confidence_threshold
        ).hex(),
        "allow_explicit_source_of_truth_override": (
            canonical.allow_explicit_source_of_truth_override
        ),
        "allow_current_evidence_over_historical": (
            canonical.allow_current_evidence_over_historical
        ),
        "require_explicit_hint_for_user_override": (
            canonical.require_explicit_hint_for_user_override
        ),
        "conflict_on_equal_precedence_disagreement": (
            canonical.conflict_on_equal_precedence_disagreement
        ),
        "max_future_clock_skew_seconds": (
            canonical.max_future_clock_skew_seconds
        ),
    }
    return (
        "policy:v1:"
        + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    )


def _bind_decision_cohort(
    group_cohort: object,
    policy: ReconciliationPolicy,
    reconciled_at: datetime,
) -> _ReconciliationCohort:
    base = _validate_reconciliation_cohort(
        group_cohort,
        require_bound=False,
    )
    if base.policy_fingerprint is not None or base.reconciled_at is not None:
        raise ReconciliationInputError("candidate group cohort stage is invalid")
    return _ReconciliationCohort._from_payload(
        _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN,
        project_id=base.project_id,
        snapshot_id=base.snapshot_id,
        policy_fingerprint=_policy_fingerprint(policy),
        reconciled_at=reconciled_at.isoformat(),
        candidate_id_namespace=base.candidate_id_namespace,
        predicate_membership=base.predicate_membership,
        invocation_token=base.invocation_token,
    )


def _candidate_group_stage_payload(
    *,
    project_id: str,
    subject: str,
    predicate: str,
    fact_id: str,
    canonical_value_key: str,
    candidates: tuple[MemoryCandidate, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "project_id": project_id,
            "subject": subject,
            "predicate": predicate,
            "fact_id": fact_id,
            "canonical_value_key": canonical_value_key,
            "candidates": [
                _canonical_candidate_payload(candidate)
                for candidate in candidates
            ],
        }
    )


_CANDIDATE_GROUP_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class CandidateGroup:
    """An internally validated bucket for one exact typed candidate value."""

    project_id: str
    subject: str
    predicate: str
    fact_id: str
    canonical_value_key: str
    candidates: tuple[MemoryCandidate, ...]

    def __init__(
        self,
        project_id: str,
        candidates: tuple[MemoryCandidate, ...],
    ) -> None:
        raise TypeError(
            "CandidateGroup construction is internal; use group_candidates"
        )

    @classmethod
    def _from_invocation_cohort(
        cls,
        construction_token: object,
        project_id: str,
        candidates: tuple[MemoryCandidate, ...],
        cohort: _ReconciliationCohort,
    ) -> CandidateGroup:
        if (
            cls is not CandidateGroup
            or construction_token is not _CANDIDATE_GROUP_CONSTRUCTION_TOKEN
        ):
            raise ReconciliationInputError(
                "candidate group factory requires exact CandidateGroup"
            )
        instance = object.__new__(cls)
        instance._initialize(
            construction_token,
            project_id,
            candidates,
            cohort,
        )
        return instance

    def _initialize(
        self,
        construction_token: object,
        project_id: str,
        candidates: tuple[MemoryCandidate, ...],
        cohort: _ReconciliationCohort,
    ) -> None:
        if (
            type(self) is not CandidateGroup
            or construction_token is not _CANDIDATE_GROUP_CONSTRUCTION_TOKEN
        ):
            raise ReconciliationInputError(
                "candidate group initialization is internal"
            )
        checked_project_id = _require_project_id(project_id)
        if not isinstance(candidates, tuple) or not candidates:
            raise ReconciliationInputError(
                "candidates must be a non-empty tuple"
            )
        if not all(
            type(candidate) is MemoryCandidate for candidate in candidates
        ):
            raise ReconciliationInputError(
                "candidates must contain only MemoryCandidate records"
            )
        for candidate in candidates:
            _require_exact_candidate_strings(candidate)

        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ReconciliationInputError(
                "candidates must be sorted by candidate_id"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ReconciliationInputError(
                "candidates must not contain duplicate candidate_id values"
            )

        representative = candidates[0]
        if any(
            candidate.subject != representative.subject
            for candidate in candidates
        ):
            raise ReconciliationInputError(
                "candidates must have the same exact subject"
            )
        if any(
            candidate.predicate != representative.predicate
            for candidate in candidates
        ):
            raise ReconciliationInputError(
                "candidates must have the same exact predicate"
            )

        all_known = all(
            candidate.value.status is EvidenceStatus.KNOWN
            for candidate in candidates
        )
        all_unresolved = all(
            candidate.value.status
            in (EvidenceStatus.UNKNOWN, EvidenceStatus.UNAVAILABLE)
            for candidate in candidates
        )
        if not all_known and not all_unresolved:
            raise ReconciliationInputError(
                "candidates must have a consistent value evidence status"
            )

        if all_known:
            canonical_keys = tuple(
                canonical_typed_value(candidate.value.value)
                for candidate in candidates
            )
            canonical_key = canonical_keys[0]
            if any(key != canonical_key for key in canonical_keys[1:]):
                raise ReconciliationInputError(
                    "KNOWN candidates must share one canonical typed value"
                )
            fact_id = make_fact_id(
                checked_project_id,
                representative.subject,
                representative.predicate,
                representative.value.value,
            )
        else:
            if len(candidates) != 1:
                raise ReconciliationInputError(
                    "an unresolved group must contain exactly one candidate"
                )
            canonical_key = (
                f"unresolved:{representative.value.status.value}:"
                f"{representative.candidate_id}"
            )
            fact_id = _make_unresolved_id(
                checked_project_id,
                representative,
            )

        object.__setattr__(self, "project_id", checked_project_id)
        object.__setattr__(self, "subject", representative.subject)
        object.__setattr__(self, "predicate", representative.predicate)
        object.__setattr__(self, "fact_id", fact_id)
        object.__setattr__(self, "canonical_value_key", canonical_key)
        object.__setattr__(self, "candidates", candidates)
        checked_cohort = _validate_reconciliation_cohort(
            cohort,
            require_bound=False,
        )
        if (
            checked_cohort.project_id != checked_project_id
            or checked_cohort.policy_fingerprint is not None
            or checked_cohort.reconciled_at is not None
        ):
            raise ReconciliationInputError(
                "candidate group does not match its invocation cohort"
            )
        object.__setattr__(self, "_cohort", checked_cohort)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return exact candidate provenance in deterministic order."""

        return tuple(candidate.candidate_id for candidate in self.candidates)


def _make_candidate_group_builder(
    issue_stage: Callable[..., object],
) -> Callable[..., CandidateGroup]:
    def build_candidate_group(
        project_id: str,
        candidates: tuple[MemoryCandidate, ...],
        cohort: _ReconciliationCohort,
    ) -> CandidateGroup:
        group = CandidateGroup._from_invocation_cohort(
            _CANDIDATE_GROUP_CONSTRUCTION_TOKEN,
            project_id,
            candidates,
            cohort,
        )
        issue_stage(
            group,
            cohort=group._cohort,
            previous_stage=None,
            parent_fingerprint=None,
            payload=_candidate_group_stage_payload(
                project_id=group.project_id,
                subject=group.subject,
                predicate=group.predicate,
                fact_id=group.fact_id,
                canonical_value_key=group.canonical_value_key,
                candidates=group.candidates,
            ),
            graph_build_token=None,
        )
        return group

    return build_candidate_group


_build_candidate_group = _make_candidate_group_builder(
    _group_stage_builder
)
del _make_candidate_group_builder
del _group_stage_builder


def _validate_candidate_group_stage(group: CandidateGroup) -> None:
    try:
        cohort = object.__getattribute__(group, "_cohort")
        payload = _candidate_group_stage_payload(
            project_id=object.__getattribute__(group, "project_id"),
            subject=object.__getattribute__(group, "subject"),
            predicate=object.__getattribute__(group, "predicate"),
            fact_id=object.__getattribute__(group, "fact_id"),
            canonical_value_key=object.__getattribute__(
                group,
                "canonical_value_key",
            ),
            candidates=object.__getattribute__(group, "candidates"),
        )
        _validate_stage_authorization(
            group,
            cohort=cohort,
            expected_stage=_ReconciliationStage.GROUPED,
            expected_previous_stage=None,
            payload=payload,
            graph_build_token=None,
        )
    except (AttributeError, TypeError, ReconciliationInputError) as error:
        raise ReconciliationInputError(
            "candidate group stage authentication is invalid"
        ) from error


@dataclass(frozen=True)
class _DecisionCandidateContext:
    """Immutable source context retained for predicate-level rules."""

    group: CandidateGroup
    policy: ReconciliationPolicy
    reconciled_at: datetime


def _normalize_relation_factory_provenance(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    return _normalize_relation_provenance(
        value,
        field_name,
        allow_list=False,
    )


@dataclass(frozen=True)
class RelationRequest:
    """A provisional, provenance-bearing fact relationship request."""

    relation_type: RelationType
    from_fact_id: str
    to_fact_id: str
    reason: str
    candidate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    method: ResolutionMethod

    @classmethod
    def supersedes(
        cls,
        from_fact_id: str,
        to_fact_id: str,
        reason: str,
        candidate_ids: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        method: ResolutionMethod = ResolutionMethod.EXPLICIT_SUPERSEDES,
    ) -> RelationRequest:
        if cls is not RelationRequest:
            raise ReconciliationInputError(
                "relation request factories require exact RelationRequest"
            )
        if type(method) is not ResolutionMethod or method not in (
            ResolutionMethod.EXPLICIT_SUPERSEDES,
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
            ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
        ):
            raise ReconciliationInputError(
                "method is not coherent with a SUPERSEDES request"
            )
        return cls(
            relation_type=RelationType.SUPERSEDES,
            from_fact_id=from_fact_id,
            to_fact_id=to_fact_id,
            reason=reason,
            candidate_ids=_normalize_relation_factory_provenance(
                candidate_ids,
                "candidate_ids",
            ),
            evidence_refs=_normalize_relation_factory_provenance(
                evidence_refs,
                "evidence_refs",
            ),
            method=method,
        )

    @classmethod
    def conflicts(
        cls,
        from_fact_id: str,
        to_fact_id: str,
        reason: str,
        candidate_ids: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        method: ResolutionMethod = ResolutionMethod.UNRESOLVED_CONFLICT,
    ) -> RelationRequest:
        if cls is not RelationRequest:
            raise ReconciliationInputError(
                "relation request factories require exact RelationRequest"
            )
        if (
            type(method) is not ResolutionMethod
            or method is not ResolutionMethod.UNRESOLVED_CONFLICT
        ):
            raise ReconciliationInputError(
                "method is not coherent with a CONFLICTS request"
            )
        return cls(
            relation_type=RelationType.CONFLICTS,
            from_fact_id=from_fact_id,
            to_fact_id=to_fact_id,
            reason=reason,
            candidate_ids=_normalize_relation_factory_provenance(
                candidate_ids,
                "candidate_ids",
            ),
            evidence_refs=_normalize_relation_factory_provenance(
                evidence_refs,
                "evidence_refs",
            ),
            method=method,
        )

    def __post_init__(self) -> None:
        if type(self.relation_type) is not RelationType:
            raise ReconciliationInputError(
                "relation_type must be a RelationType"
            )
        for field_name in ("from_fact_id", "to_fact_id", "reason"):
            _validate_required_text(getattr(self, field_name), field_name)
        if type(self.method) is not ResolutionMethod:
            raise ReconciliationInputError(
                "method must be a ResolutionMethod"
            )
        coherent = (
            self.relation_type is RelationType.SUPERSEDES
            and self.method
            in (
                ResolutionMethod.EXPLICIT_SUPERSEDES,
                ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
                ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
            )
        ) or (
            self.relation_type is RelationType.CONFLICTS
            and self.method is ResolutionMethod.UNRESOLVED_CONFLICT
        )
        if not coherent:
            raise ReconciliationInputError(
                "relation_type and method must be coherent"
            )
        for field_name in ("candidate_ids", "evidence_refs"):
            values = getattr(self, field_name)
            if (
                type(values) is not tuple
                or not values
                or any(
                    type(value) is not str
                    or not value.strip()
                    or "\x00" in value
                    for value in values
                )
                or values != tuple(sorted(set(values)))
            ):
                raise ReconciliationInputError(
                    f"{field_name} must be a sorted non-empty tuple of "
                    "unique strings"
                )
        if self.from_fact_id == self.to_fact_id:
            raise ReconciliationInputError(
                "relation request cannot be a fact self-edge"
            )


_PREDICATE_OUTCOME_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True)
class PredicateOutcome:
    """Immutable predicate-level decisions and provisional relations."""

    decisions: tuple[FactDecision, ...]
    relation_requests: tuple[RelationRequest, ...]
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if type(self) is not PredicateOutcome:
            raise ReconciliationInputError(
                "predicate outcomes require exact PredicateOutcome"
            )
        if type(self.decisions) is not tuple or any(
            type(decision) is not FactDecision for decision in self.decisions
        ):
            raise ReconciliationInputError(
                "decisions must be an exact tuple of FactDecision records"
            )
        if type(self.relation_requests) is not tuple or any(
            type(request) is not RelationRequest
            for request in self.relation_requests
        ):
            raise ReconciliationInputError(
                "relation_requests must be an exact tuple of RelationRequest "
                "records"
            )
        canonical_base_decisions = tuple(
            _validate_decision_candidate_context(
                decision,
                require_stage_auth=False,
                validate_identity=False,
            )
            for decision in self.decisions
        )
        for decision, canonical in zip(
            self.decisions,
            canonical_base_decisions,
            strict=True,
        ):
            if (
                decision.fact_id != canonical.fact_id
                or decision.subject != canonical.subject
                or decision.predicate != canonical.predicate
                or _evidence_payload_signature(
                    decision.selected_value,
                    "selected_value",
                    "json",
                )
                != _evidence_payload_signature(
                    canonical.selected_value,
                    "selected_value",
                    "json",
                )
                or decision.candidate_ids != canonical.candidate_ids
                or decision.evidence_refs != canonical.evidence_refs
            ):
                raise ReconciliationInputError(
                    "predicate outcome requires canonical FactDecision "
                    "identity"
                )
        predicate_identities = {
            (decision.subject, decision.predicate)
            for decision in self.decisions
        }
        if len(predicate_identities) > 1:
            raise ReconciliationInputError(
                "predicate outcome decisions must share one exact subject "
                "and predicate"
            )
        fact_ids = tuple(decision.fact_id for decision in self.decisions)
        if fact_ids != tuple(sorted(fact_ids)):
            raise ReconciliationInputError(
                "predicate outcome decisions must be sorted by fact_id"
            )
        if len(set(fact_ids)) != len(fact_ids):
            raise ReconciliationInputError(
                "predicate outcome decisions must have unique fact_id values"
            )
        relation_keys = tuple(
            (
                request.relation_type.value,
                request.from_fact_id,
                request.to_fact_id,
            )
            for request in self.relation_requests
        )
        if relation_keys != tuple(sorted(set(relation_keys))):
            raise ReconciliationInputError(
                "predicate outcome relation request keys must be sorted and "
                "unique"
            )
        incoming_supersedes = {
            request.to_fact_id
            for request in self.relation_requests
            if request.relation_type is RelationType.SUPERSEDES
        }
        decisions_by_fact_id = {
            decision.fact_id: decision for decision in self.decisions
        }
        for request in self.relation_requests:
            source = decisions_by_fact_id.get(request.from_fact_id)
            target = decisions_by_fact_id.get(request.to_fact_id)
            if source is None or target is None:
                raise ReconciliationInputError(
                    "a relation request requires both endpoint decisions"
                )
            if request.relation_type is RelationType.CONFLICTS:
                if request.reason not in (
                    _UNRESOLVED_CURRENT_CONFLICT_REASON,
                    _CONTRADICTORY_SUPERSEDES_REASON,
                ):
                    raise ReconciliationInputError(
                        "a CONFLICTS request requires a canonical reason"
                    )
                expected_candidate_ids = tuple(
                    sorted(set((*source.candidate_ids, *target.candidate_ids)))
                )
                expected_evidence_refs = tuple(
                    sorted(set((*source.evidence_refs, *target.evidence_refs)))
                )
                if (
                    source.status is not ReconciliationStatus.CONFLICTED
                    or target.status is not ReconciliationStatus.CONFLICTED
                ):
                    raise ReconciliationInputError(
                        "a CONFLICTS request requires CONFLICTED endpoints"
                    )
                if (
                    source.reason != request.reason
                    or target.reason != request.reason
                ):
                    raise ReconciliationInputError(
                        "a CONFLICTS request reason must match its endpoints"
                    )
                if (
                    request.candidate_ids != expected_candidate_ids
                    or request.evidence_refs != expected_evidence_refs
                ):
                    raise ReconciliationInputError(
                        "a CONFLICTS request requires exact endpoint provenance"
                )
                continue
            if not _is_canonical_supersedes_request(
                request,
                source,
                target,
                canonical_base_decisions,
            ):
                raise ReconciliationInputError(
                    "SUPERSEDES request payload is not canonical"
                )
            if target.status is not ReconciliationStatus.SUPERSEDED:
                raise ReconciliationInputError(
                    "a SUPERSEDES request target must be SUPERSEDED"
                )
            if (
                source.status is not ReconciliationStatus.ACTIVE
                and source.fact_id not in incoming_supersedes
            ):
                raise ReconciliationInputError(
                    "a SUPERSEDES request source must be ACTIVE or itself "
                    "superseded by an incoming request"
                )
        if not _outgoing_replacement_primary_methods_hold(
            self.decisions,
            self.relation_requests,
        ):
            raise ReconciliationInputError(
                "a replacement source must record its highest-priority "
                "outgoing rule"
            )
        if not _coordinator_preconditions_hold(
            self.decisions,
            self.relation_requests,
            canonical_base_decisions,
        ):
            raise ReconciliationInputError(
                "predicate outcome violates global coordinator preconditions"
            )
        if any(
            decision.status is ReconciliationStatus.SUPERSEDED
            and decision.fact_id not in incoming_supersedes
            for decision in self.decisions
        ):
                raise ReconciliationInputError(
                    "a SUPERSEDED decision requires an incoming request"
                )
        conflict_requests = {
            (request.from_fact_id, request.to_fact_id): request
            for request in self.relation_requests
            if request.relation_type is RelationType.CONFLICTS
        }
        for direction, request in conflict_requests.items():
            reverse = conflict_requests.get((direction[1], direction[0]))
            if reverse is None or (
                reverse.reason,
                reverse.candidate_ids,
                reverse.evidence_refs,
                reverse.method,
            ) != (
                request.reason,
                request.candidate_ids,
                request.evidence_refs,
                request.method,
            ):
                raise ReconciliationInputError(
                    "CONFLICTS requests must be exactly symmetric"
                )
        conflicted_ids = {
            decision.fact_id
            for decision in self.decisions
            if decision.status is ReconciliationStatus.CONFLICTED
        }
        if conflicted_ids:
            if any(
                decision.status is ReconciliationStatus.ACTIVE
                for decision in self.decisions
            ):
                raise ReconciliationInputError(
                    "a conflicted predicate cannot retain an ACTIVE winner"
                )
            if any(
                not decision.requires_human_review
                for decision in self.decisions
                if decision.fact_id in conflicted_ids
            ):
                raise ReconciliationInputError(
                    "CONFLICTED decisions require human review"
                )
            request_endpoints = {
                fact_id
                for direction in conflict_requests
                for fact_id in direction
            }
            if request_endpoints != conflicted_ids:
                raise ReconciliationInputError(
                    "CONFLICTED decisions and request endpoints must match"
                )
            expected_directions = {
                (left_id, right_id)
                for left_id in conflicted_ids
                for right_id in conflicted_ids
                if left_id != right_id
            }
            if set(conflict_requests) != expected_directions:
                raise ReconciliationInputError(
                    "CONFLICTED decisions require every directed pair"
                )
        if _construction_token is not _PREDICATE_OUTCOME_CONSTRUCTION_TOKEN:
            raise ReconciliationInputError(
                "PredicateOutcome must be created by its canonical factory"
            )

    @classmethod
    def _from_canonical_resolution(
        cls,
        construction_token: object,
        decisions: tuple[FactDecision, ...],
        relation_requests: tuple[RelationRequest, ...],
    ) -> PredicateOutcome:
        if (
            cls is not PredicateOutcome
            or construction_token
            is not _PREDICATE_OUTCOME_CONSTRUCTION_TOKEN
        ):
            raise ReconciliationInputError(
                "PredicateOutcome must be created by its canonical factory"
            )
        return cls(
            decisions,
            relation_requests,
            _construction_token=construction_token,
        )

    @property
    def statuses(self) -> dict[str, ReconciliationStatus]:
        """Return a fresh deterministic fact-status index."""

        return {
            decision.fact_id: decision.status
            for decision in self.decisions
        }

    @property
    def active_fact_ids(self) -> tuple[str, ...]:
        """Return ACTIVE fact identifiers in canonical order."""

        return tuple(
            decision.fact_id
            for decision in self.decisions
            if decision.status is ReconciliationStatus.ACTIVE
        )

    def status_for(self, fact_id: str) -> ReconciliationStatus:
        """Return the resolved status for one exact fact identifier."""

        if type(fact_id) is not str or not fact_id:
            raise ReconciliationInputError(
                "fact_id must be a non-empty string"
            )
        for decision in self.decisions:
            if decision.fact_id == fact_id:
                return decision.status
        raise ReconciliationInputError(f"unknown fact_id: {fact_id}")


def _canonical_predicate_outcome_payload(
    decisions: tuple[FactDecision, ...],
    relation_requests: tuple[RelationRequest, ...],
) -> tuple[tuple[FactDecision, ...], tuple[RelationRequest, ...]]:
    """Re-derive one exact canonical predicate outcome payload."""

    if type(decisions) is not tuple or any(
        type(decision) is not FactDecision for decision in decisions
    ):
        raise ReconciliationInputError(
            "decisions must be an exact tuple of FactDecision records"
        )
    if type(relation_requests) is not tuple or any(
        type(request) is not RelationRequest
        for request in relation_requests
    ):
        raise ReconciliationInputError(
            "relation_requests must be an exact tuple of RelationRequest "
            "records"
        )
    if not decisions:
        if relation_requests:
            raise ReconciliationInputError(
                "empty decisions cannot have relation requests"
            )
        return (), ()

    try:
        canonical_base_decisions = tuple(
            _validate_decision_candidate_context(
                decision,
                validate_identity=False,
            )
            for decision in decisions
        )
    except ReconciliationInputError as error:
        raise ReconciliationInputError(
            "predicate outcome does not match canonical coordinator payload"
        ) from error
    policy = canonical_base_decisions[0]._candidate_context.policy
    expected_decisions, expected_requests = _resolve_predicate_payload(
        canonical_base_decisions,
        policy,
    )
    if (
        _decision_payload_signatures(decisions)
        != _decision_payload_signatures(expected_decisions)
        or _request_payload_signatures(relation_requests)
        != _request_payload_signatures(expected_requests)
    ):
        raise ReconciliationInputError(
            "predicate outcome does not match canonical coordinator payload"
        )

    return expected_decisions, expected_requests


def _evidence_payload_signature(
    value: object,
    field_name: str,
    known_kind: str,
) -> bytes:
    try:
        return _strict_evidence_signature(value, field_name, known_kind)
    except ReconciliationInputError as error:
        raise ReconciliationInputError(
            "predicate outcome has a non-canonical evidence value: "
            f"{field_name}"
        ) from error


def _warning_payload_signature(
    warning: ReconciliationWarning,
) -> tuple[object, ...]:
    if type(warning) is not ReconciliationWarning:
        raise ReconciliationInputError(
            "predicate outcome has a non-canonical warning"
        )
    return (
        warning.code.value,
        warning.message,
        warning.candidate_ids,
        warning.evidence_refs,
        warning.requires_human_review,
    )


def _decision_payload_signatures(
    decisions: tuple[FactDecision, ...],
) -> tuple[tuple[object, ...], ...]:
    signatures: list[tuple[object, ...]] = []
    for decision in decisions:
        if (
            type(decision) is not FactDecision
            or type(decision.status) is not ReconciliationStatus
            or type(decision.resolution_method) is not ResolutionMethod
            or type(decision.requires_human_review) is not bool
            or type(decision.warnings) is not tuple
            or type(decision.relation_requests) is not tuple
        ):
            raise ReconciliationInputError(
                "predicate outcome has a non-canonical decision payload"
            )
        signatures.append(
            (
                decision.fact_id,
                decision.subject,
                decision.predicate,
                _evidence_payload_signature(
                    decision.selected_value,
                    "selected_value",
                    "json",
                ),
                decision.candidate_ids,
                decision.evidence_refs,
                decision.activation_witness_candidate_ids,
                decision.superseded_candidate_ids,
                decision.status.value,
                _evidence_payload_signature(
                    decision.confidence,
                    "confidence",
                    "float",
                ),
                _evidence_payload_signature(
                    decision.valid_from,
                    "valid_from",
                    "string",
                ),
                decision.resolution_method.value,
                decision.reason,
                decision.requires_human_review,
                tuple(
                    _warning_payload_signature(warning)
                    for warning in decision.warnings
                ),
                decision.relation_requests,
            )
        )
    return tuple(signatures)


def _signature_json_node(value: object) -> object:
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is tuple:
        return [_signature_json_node(item) for item in value]
    if type(value) in (str, bool, int, float) or value is None:
        return value
    raise ReconciliationInputError(
        "stage signature contains a non-canonical value"
    )


def _decision_stage_payload(decision: FactDecision) -> bytes:
    return canonical_json_bytes(
        _signature_json_node(_decision_payload_signatures((decision,))[0])
    )


def _request_payload_signatures(
    requests: tuple[RelationRequest, ...],
) -> tuple[tuple[object, ...], ...]:
    signatures: list[tuple[object, ...]] = []
    for request in requests:
        if (
            type(request) is not RelationRequest
            or type(request.relation_type) is not RelationType
            or type(request.method) is not ResolutionMethod
        ):
            raise ReconciliationInputError(
                "predicate outcome has a non-canonical request payload"
            )
        signatures.append(
            (
                request.relation_type.value,
                request.from_fact_id,
                request.to_fact_id,
                request.reason,
                request.candidate_ids,
                request.evidence_refs,
                request.method.value,
            )
        )
    return tuple(signatures)


def _candidate_evidence_refs(
    *candidates: MemoryCandidate,
) -> tuple[str, ...]:
    refs = {candidate.source_ref for candidate in candidates}
    refs.update(
        getattr(candidate, field_name).source
        for candidate in candidates
        for field_name in _CANDIDATE_EVIDENCE_FIELDS
    )
    return tuple(sorted(refs))


_RESOLUTION_METHOD_PRIORITY = {
    method: index
    for index, method in enumerate(
        (
            ResolutionMethod.SAME_VALUE_REACTIVATION,
            ResolutionMethod.EXPLICIT_DEPRECATION,
            ResolutionMethod.TEMPORAL_PENDING,
            ResolutionMethod.INSUFFICIENT_EVIDENCE,
            ResolutionMethod.PENDING_SEMANTICS,
            ResolutionMethod.UNRESOLVED_CONFLICT,
            ResolutionMethod.EXPLICIT_SUPERSEDES,
            ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
            ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
            ResolutionMethod.SAME_VALUE_MERGE,
            ResolutionMethod.DIRECT_CURRENT,
        )
    )
}


def _relation_request_priority(
    request: RelationRequest,
) -> tuple[int, str]:
    return (_RESOLUTION_METHOD_PRIORITY[request.method], request.reason)


@dataclass
class _RelationRequestAccumulator:
    relation_type: RelationType
    from_fact_id: str
    to_fact_id: str
    primary: RelationRequest
    candidate_ids: set[str]
    evidence_refs: set[str]

    def add(self, request: RelationRequest) -> None:
        if _relation_request_priority(request) < _relation_request_priority(
            self.primary
        ):
            self.primary = request
        self.candidate_ids.update(request.candidate_ids)
        self.evidence_refs.update(request.evidence_refs)

    def materialize(self) -> RelationRequest:
        factory = (
            RelationRequest.supersedes
            if self.relation_type is RelationType.SUPERSEDES
            else RelationRequest.conflicts
        )
        return factory(
            self.from_fact_id,
            self.to_fact_id,
            self.primary.reason,
            tuple(self.candidate_ids),
            tuple(self.evidence_refs),
            method=self.primary.method,
        )


def _merge_relation_requests(
    requests: Sequence[RelationRequest],
) -> tuple[RelationRequest, ...]:
    accumulators: dict[
        tuple[str, str, str], _RelationRequestAccumulator
    ] = {}
    for request in requests:
        key = (
            request.relation_type.value,
            request.from_fact_id,
            request.to_fact_id,
        )
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = _RelationRequestAccumulator(
                relation_type=request.relation_type,
                from_fact_id=request.from_fact_id,
                to_fact_id=request.to_fact_id,
                primary=request,
                candidate_ids=set(),
                evidence_refs=set(),
            )
            accumulators[key] = accumulator
        accumulator.add(request)
    return tuple(
        accumulators[key].materialize() for key in sorted(accumulators)
    )


def _drop_mutual_supersedes(
    requests: Sequence[RelationRequest],
) -> tuple[RelationRequest, ...]:
    supersedes_directions = {
        (request.from_fact_id, request.to_fact_id): request
        for request in requests
        if request.relation_type is RelationType.SUPERSEDES
    }
    return tuple(
        request
        for request in requests
        if request.relation_type is not RelationType.SUPERSEDES
        or (
            request.to_fact_id,
            request.from_fact_id,
        )
        not in supersedes_directions
    )


def _guarded_sequence_snapshot(
    value: object,
    field_name: str,
    item_description: str,
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ReconciliationInputError(
            f"{field_name} must be a sequence of {item_description}"
        )
    try:
        return tuple(value)
    except Exception as error:
        raise ReconciliationInputError(
            f"{field_name} must be a stable finite sequence of {item_description}"
        ) from error


def group_candidates(
    project_id: str,
    candidates: Sequence[MemoryCandidate],
    *,
    snapshot_id: str | None = None,
) -> tuple[CandidateGroup, ...]:
    """Group exact typed values without interpreting candidate semantics."""

    checked_project_id = _require_project_id(project_id)
    copied = _guarded_sequence_snapshot(
        candidates,
        "candidates",
        "MemoryCandidate records",
    )
    if not all(type(candidate) is MemoryCandidate for candidate in copied):
        raise ReconciliationInputError(
            "candidates must contain only MemoryCandidate records"
        )
    for candidate in copied:
        _require_exact_candidate_strings(candidate)

    candidate_id_counts: dict[str, int] = {}
    for candidate in copied:
        candidate_id_counts[candidate.candidate_id] = (
            candidate_id_counts.get(candidate.candidate_id, 0) + 1
        )
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in candidate_id_counts.items()
        if count > 1
    )
    if duplicate_ids:
        raise ReconciliationInputError(
            f"duplicate candidate_id: {duplicate_ids[0]}"
        )

    cohort = _make_invocation_cohort(
        checked_project_id,
        copied,
        snapshot_id,
    )

    buckets: dict[tuple[str, ...], list[MemoryCandidate]] = {}
    for candidate in sorted(copied, key=lambda item: item.candidate_id):
        if candidate.value.status is EvidenceStatus.KNOWN:
            canonical_key = canonical_typed_value(candidate.value.value)
            bucket_key = (
                candidate.subject,
                candidate.predicate,
                "known",
                canonical_key,
            )
        else:
            canonical_key = (
                f"unresolved:{candidate.value.status.value}:"
                f"{candidate.candidate_id}"
            )
            bucket_key = (
                candidate.subject,
                candidate.predicate,
                "unresolved",
                candidate.candidate_id,
                candidate.value.status.value,
            )
        buckets.setdefault(bucket_key, []).append(candidate)

    groups: list[CandidateGroup] = []
    for _, members in sorted(buckets.items()):
        groups.append(
            _build_candidate_group(
                checked_project_id,
                tuple(members),
                cohort,
            )
        )
    return tuple(groups)


def _is_current_semantic(
    candidate: MemoryCandidate,
    policy: ReconciliationPolicy,
) -> bool:
    return (
        candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.CURRENT_FACT
    ) or is_valid_source_of_truth(candidate, policy)


def _is_deprecated_semantic(candidate: MemoryCandidate) -> bool:
    return (
        candidate.deprecated.status is EvidenceStatus.KNOWN
        and candidate.deprecated.value is True
    ) or (
        candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.DEPRECATED
    )


def _validate_temporal_assessment(
    candidate: MemoryCandidate,
    temporal: TemporalAssessment,
    policy: ReconciliationPolicy,
) -> TemporalAssessment:
    if not isinstance(candidate, MemoryCandidate):
        raise ReconciliationInputError("candidate must be a MemoryCandidate")
    if type(temporal) is not TemporalAssessment:
        raise ReconciliationInputError("temporal must be a TemporalAssessment")
    if not isinstance(policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")
    if temporal.candidate_id != candidate.candidate_id:
        raise ReconciliationInputError(
            "temporal assessment candidate_id does not match candidate"
        )
    if (
        temporal.max_future_clock_skew_seconds
        != policy.max_future_clock_skew_seconds
    ):
        raise ReconciliationInputError(
            "temporal assessment does not match the supplied policy"
        )
    expected = assess_temporal(candidate, temporal.assessed_at, policy)
    if temporal != expected:
        raise ReconciliationInputError(
            "temporal assessment does not match its canonical derivation"
        )
    return expected


def _passes_activation_witness_gates(
    candidate: MemoryCandidate,
    temporal: TemporalAssessment,
    policy: ReconciliationPolicy,
) -> bool:
    return (
        candidate.value.status is EvidenceStatus.KNOWN
        and _is_current_semantic(candidate, policy)
        and temporal.eligible
        and candidate.confidence.status is EvidenceStatus.KNOWN
        and candidate.confidence.value >= policy.active_confidence_threshold
    )


def is_activation_witness(
    candidate: MemoryCandidate,
    temporal: TemporalAssessment,
    policy: ReconciliationPolicy,
) -> bool:
    """Return whether one candidate independently passes every activation gate."""

    canonical = _validate_temporal_assessment(candidate, temporal, policy)
    return _passes_activation_witness_gates(candidate, canonical, policy)


def find_activation_witnesses(
    group: CandidateGroup,
    assessments: Mapping[str, TemporalAssessment],
    policy: ReconciliationPolicy,
) -> tuple[MemoryCandidate, ...]:
    """Return candidate-local activation witnesses in deterministic ID order."""

    if not isinstance(group, CandidateGroup):
        raise ReconciliationInputError("group must be a CandidateGroup")
    if not isinstance(assessments, Mapping):
        raise ReconciliationInputError("assessments must be a mapping")
    if not isinstance(policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")

    assessment_items = tuple(assessments.items())
    expected_ids = set(group.candidate_ids)
    assessment_keys = tuple(key for key, _ in assessment_items)
    if (
        len(assessment_keys) != len(expected_ids)
        or set(assessment_keys) != expected_ids
    ):
        raise ReconciliationInputError(
            "assessments must contain exactly the group candidate IDs"
        )
    if not all(
        type(assessment) is TemporalAssessment
        for _, assessment in assessment_items
    ):
        raise ReconciliationInputError(
            "assessments must contain only TemporalAssessment records"
        )

    assessment_snapshot = dict(assessment_items)

    assessment_clocks = {
        assessment.assessed_at
        for assessment in assessment_snapshot.values()
    }
    if len(assessment_clocks) != 1:
        raise ReconciliationInputError(
            "temporal assessments must use one reconciliation clock"
        )

    ordered_candidates = tuple(
        sorted(group.candidates, key=lambda item: item.candidate_id)
    )
    canonical_assessments: dict[str, TemporalAssessment] = {}
    for candidate in ordered_candidates:
        canonical_assessments[candidate.candidate_id] = (
            _validate_temporal_assessment(
                candidate,
                assessment_snapshot[candidate.candidate_id],
                policy,
            )
        )

    return tuple(
        candidate
        for candidate in ordered_candidates
        if _passes_activation_witness_gates(
            candidate,
            canonical_assessments[candidate.candidate_id],
            policy,
        )
    )


def _is_activation_capable(
    candidate: MemoryCandidate,
    assessment: TemporalAssessment,
    policy: ReconciliationPolicy,
) -> bool:
    """Apply only the candidate-local gates needed by same-value lineage."""

    # Task 8 creates this assessment immediately above the lineage analysis.
    return _passes_activation_witness_gates(
        candidate,
        assessment,
        policy,
    ) and not _is_deprecated_semantic(candidate)


def _lineage_warning(
    candidate_ids: set[str],
    evidence_refs: set[str],
    message: str,
) -> ReconciliationWarning:
    return _make_warning(
        WarningCode.CONTRADICTORY_SUPERSEDES,
        message,
        tuple(candidate_ids),
        tuple(evidence_refs),
        requires_human_review=True,
    )


@dataclass(frozen=True)
class _LineageEdge:
    explicit_evidence_refs: tuple[str, ...] = ()
    interval_evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LineageGraph:
    successors: Mapping[str, set[str]]
    edges: Mapping[tuple[str, str], _LineageEdge]


@dataclass(frozen=True)
class _LineageGraphAnalysis:
    components: tuple[tuple[str, ...], ...]
    component_by_node: Mapping[str, int]
    cyclic_component_ids: tuple[int, ...]
    topological_order: tuple[str, ...]


def _add_lineage_edge(
    successors: dict[str, set[str]],
    edges: dict[tuple[str, str], _LineageEdge],
    predecessor_id: str,
    successor_id: str,
    *,
    explicit_evidence_refs: tuple[str, ...] = (),
    interval_evidence_refs: tuple[str, ...] = (),
) -> None:
    key = (predecessor_id, successor_id)
    existing = edges.get(key, _LineageEdge())
    edges[key] = _LineageEdge(
        explicit_evidence_refs=tuple(
            sorted(
                set(existing.explicit_evidence_refs)
                | set(explicit_evidence_refs)
            )
        ),
        interval_evidence_refs=tuple(
            sorted(
                set(existing.interval_evidence_refs)
                | set(interval_evidence_refs)
            )
        ),
    )
    successors[predecessor_id].add(successor_id)


def _build_same_value_graph(
    group: CandidateGroup,
    now: datetime,
    assessments: Mapping[str, TemporalAssessment],
    policy: ReconciliationPolicy,
) -> tuple[_LineageGraph, ReconciliationWarning | None]:
    candidates = {
        candidate.candidate_id: candidate for candidate in group.candidates
    }
    successors = {candidate_id: set() for candidate_id in candidates}
    edges: dict[tuple[str, str], _LineageEdge] = {}
    self_references: set[str] = set()
    self_reference_sources: set[str] = set()

    for successor in group.candidates:
        if successor.supersedes.status is not EvidenceStatus.KNOWN:
            continue
        for predecessor_id in successor.supersedes.value:
            if predecessor_id == successor.candidate_id:
                self_references.add(successor.candidate_id)
                self_reference_sources.add(successor.supersedes.source)
                continue
            if predecessor_id not in candidates:
                continue
            _add_lineage_edge(
                successors,
                edges,
                predecessor_id,
                successor.candidate_id,
                explicit_evidence_refs=(successor.supersedes.source,),
            )

    if self_references:
        return _LineageGraph(successors, edges), _lineage_warning(
            self_references,
            self_reference_sources,
            "same-value supersedes reference is self-referential",
        )

    deprecated_intervals: list[tuple[datetime, str, MemoryCandidate]] = []
    activation_starts: list[tuple[datetime, str, MemoryCandidate]] = []
    for candidate in group.candidates:
        assessment = assessments[candidate.candidate_id]
        if _is_deprecated_semantic(candidate):
            old_until = assessment.valid_until
            old_from = assessment.valid_from
            if old_until is not None and (
                old_from is None or old_from <= old_until
            ):
                deprecated_intervals.append(
                    (old_until, candidate.candidate_id, candidate)
                )
        if (
            _is_activation_capable(candidate, assessment, policy)
            and assessment.valid_from is not None
            and assessment.valid_from <= now
        ):
            activation_starts.append(
                (
                    assessment.valid_from,
                    candidate.candidate_id,
                    candidate,
                )
            )

    deprecated_intervals.sort(key=lambda item: (item[0], item[1]))
    activation_starts.sort(key=lambda item: (item[0], item[1]))
    eligible_deprecated: list[tuple[datetime, str, MemoryCandidate]] = []
    deprecated_index = 0
    for new_from, _, new in activation_starts:
        while (
            deprecated_index < len(deprecated_intervals)
            and deprecated_intervals[deprecated_index][0] <= new_from
        ):
            eligible_deprecated.append(
                deprecated_intervals[deprecated_index]
            )
            deprecated_index += 1
        for _, _, old in eligible_deprecated:
            if old.candidate_id == new.candidate_id:
                continue
            _add_lineage_edge(
                successors,
                edges,
                old.candidate_id,
                new.candidate_id,
                interval_evidence_refs=(
                    old.valid_until.source,
                    new.valid_from.source,
                ),
            )

    return _LineageGraph(successors, edges), None


def _analyze_lineage_graph(
    successors: Mapping[str, set[str]],
) -> _LineageGraphAnalysis:
    adjacency = {
        candidate_id: tuple(successors[candidate_id])
        for candidate_id in successors
    }
    reverse_adjacency = {candidate_id: [] for candidate_id in adjacency}
    indegree = {candidate_id: 0 for candidate_id in adjacency}
    for predecessor_id, successor_ids in adjacency.items():
        for successor_id in successor_ids:
            reverse_adjacency[successor_id].append(predecessor_id)
            indegree[successor_id] += 1

    finished: list[str] = []
    visited: set[str] = set()
    for start_id in adjacency:
        if start_id in visited:
            continue
        pending: list[tuple[str, bool]] = [(start_id, False)]
        while pending:
            candidate_id, expanded = pending.pop()
            if expanded:
                finished.append(candidate_id)
                continue
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            pending.append((candidate_id, True))
            pending.extend(
                (successor_id, False)
                for successor_id in adjacency[candidate_id]
                if successor_id not in visited
            )

    assigned: set[str] = set()
    components: list[tuple[str, ...]] = []
    component_by_node: dict[str, int] = {}
    cyclic_component_ids: list[int] = []
    for start_id in reversed(finished):
        if start_id in assigned:
            continue
        component: set[str] = set()
        pending = [start_id]
        assigned.add(start_id)
        while pending:
            candidate_id = pending.pop()
            component.add(candidate_id)
            for predecessor_id in reverse_adjacency[candidate_id]:
                if predecessor_id not in assigned:
                    assigned.add(predecessor_id)
                    pending.append(predecessor_id)
        component_id = len(components)
        normalized_component = tuple(sorted(component))
        components.append(normalized_component)
        for candidate_id in component:
            component_by_node[candidate_id] = component_id
        if len(component) > 1 or any(
            candidate_id in adjacency[candidate_id]
            for candidate_id in component
        ):
            cyclic_component_ids.append(component_id)

    if cyclic_component_ids:
        return _LineageGraphAnalysis(
            components=tuple(components),
            component_by_node=component_by_node,
            cyclic_component_ids=tuple(cyclic_component_ids),
            topological_order=(),
        )

    ready = [
        candidate_id
        for candidate_id, count in indegree.items()
        if count == 0
    ]
    topological_order: list[str] = []
    while ready:
        candidate_id = ready.pop()
        topological_order.append(candidate_id)
        for successor_id in adjacency[candidate_id]:
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)
    return _LineageGraphAnalysis(
        components=tuple(components),
        component_by_node=component_by_node,
        cyclic_component_ids=(),
        topological_order=tuple(topological_order),
    )


def _cycle_nodes(successors: Mapping[str, set[str]]) -> tuple[str, ...]:
    analysis = _analyze_lineage_graph(successors)
    return tuple(
        sorted(
            candidate_id
            for component_id in analysis.cyclic_component_ids
            for candidate_id in analysis.components[component_id]
        )
    )


def _lineage_predecessors(
    graph: _LineageGraph,
) -> dict[str, set[str]]:
    predecessors = {
        candidate_id: set() for candidate_id in graph.successors
    }
    for predecessor_id, successor_id in graph.edges:
        predecessors[successor_id].add(predecessor_id)
    return predecessors


def _semantic_ancestor_flags(
    predecessors: Mapping[str, set[str]],
    topological_order: tuple[str, ...],
    candidates: Mapping[str, MemoryCandidate],
    is_current_semantic: Callable[[MemoryCandidate], bool],
) -> tuple[dict[str, bool], dict[str, bool]]:
    current_flags: dict[str, bool] = {}
    deprecated_flags: dict[str, bool] = {}
    for candidate_id in topological_order:
        current_flags[candidate_id] = any(
            is_current_semantic(candidates[predecessor_id])
            or current_flags[predecessor_id]
            for predecessor_id in predecessors[candidate_id]
        )
        deprecated_flags[candidate_id] = any(
            _is_deprecated_semantic(candidates[predecessor_id])
            or deprecated_flags[predecessor_id]
            for predecessor_id in predecessors[candidate_id]
        )
    return current_flags, deprecated_flags


def _collect_winning_ancestors(
    predecessors: Mapping[str, set[str]],
    winning_tip_ids: tuple[str, ...],
) -> tuple[str, ...]:
    ancestors: set[str] = set()
    pending = [
        predecessor_id
        for tip_id in winning_tip_ids
        for predecessor_id in predecessors[tip_id]
    ]
    while pending:
        candidate_id = pending.pop()
        if candidate_id in ancestors:
            continue
        ancestors.add(candidate_id)
        pending.extend(predecessors[candidate_id])
    return tuple(sorted(ancestors))


def _explicit_incompatible_fork_warning(
    graph: _LineageGraph,
    topological_order: tuple[str, ...],
    current_tip_ids: tuple[str, ...],
    deprecated_tip_ids: tuple[str, ...],
) -> ReconciliationWarning | None:
    current_targets = set(current_tip_ids) - set(deprecated_tip_ids)
    deprecated_targets = set(deprecated_tip_ids) - set(current_tip_ids)
    reaches_current = {
        candidate_id: candidate_id in current_targets
        for candidate_id in graph.successors
    }
    reaches_deprecated = {
        candidate_id: candidate_id in deprecated_targets
        for candidate_id in graph.successors
    }
    for predecessor_id in reversed(topological_order):
        for successor_id in graph.successors[predecessor_id]:
            edge = graph.edges[(predecessor_id, successor_id)]
            if not edge.explicit_evidence_refs:
                continue
            reaches_current[predecessor_id] = (
                reaches_current[predecessor_id]
                or reaches_current[successor_id]
            )
            reaches_deprecated[predecessor_id] = (
                reaches_deprecated[predecessor_id]
                or reaches_deprecated[successor_id]
            )

    fork_nodes = {
        candidate_id
        for candidate_id in graph.successors
        if reaches_current[candidate_id] and reaches_deprecated[candidate_id]
    }
    if not fork_nodes:
        return None

    descends_from_fork = {
        candidate_id: candidate_id in fork_nodes
        for candidate_id in graph.successors
    }
    for predecessor_id in topological_order:
        if not descends_from_fork[predecessor_id]:
            continue
        for successor_id in graph.successors[predecessor_id]:
            edge = graph.edges[(predecessor_id, successor_id)]
            if edge.explicit_evidence_refs:
                descends_from_fork[successor_id] = True

    participants = {
        candidate_id
        for candidate_id in graph.successors
        if descends_from_fork[candidate_id]
        and (reaches_current[candidate_id] or reaches_deprecated[candidate_id])
    }
    evidence_refs: set[str] = set()
    for (predecessor_id, successor_id), edge in graph.edges.items():
        if predecessor_id in participants and successor_id in participants:
            evidence_refs.update(edge.explicit_evidence_refs)
    return _lineage_warning(
        participants,
        evidence_refs,
        "explicit same-value lineage forks to incompatible semantic tips",
    )


def resolve_same_value_lineage(
    group: CandidateGroup,
    now: datetime,
    policy: ReconciliationPolicy | None = None,
) -> SameValueLineage:
    """Resolve same-value candidate edges without creating a fact relation."""

    if not isinstance(group, CandidateGroup):
        raise ReconciliationInputError("group must be a CandidateGroup")
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise ReconciliationInputError("now must be a timezone-aware datetime")
    try:
        reconciled_at = now.astimezone(timezone.utc)
    except OverflowError as error:
        raise ReconciliationInputError("now cannot be normalized to UTC") from error
    selected_policy = ReconciliationPolicy() if policy is None else policy
    if not isinstance(selected_policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")

    candidates = {
        candidate.candidate_id: candidate for candidate in group.candidates
    }

    def is_current_semantic(candidate: MemoryCandidate) -> bool:
        return _is_current_semantic(
            candidate,
            selected_policy,
        ) and not _is_deprecated_semantic(candidate)

    assessments = {
        candidate.candidate_id: assess_temporal(
            candidate,
            reconciled_at,
            selected_policy,
        )
        for candidate in group.candidates
    }
    invalid_temporal_warnings = tuple(
        warning
        for candidate_id in sorted(assessments)
        for warning in assessments[candidate_id].warnings
        if warning.code is WarningCode.INVALID_TEMPORAL_ORDER
    )
    graph, invalid_warning = _build_same_value_graph(
        group,
        reconciled_at,
        assessments,
        selected_policy,
    )
    if invalid_warning is not None:
        warnings = tuple(
            sorted(
                (*invalid_temporal_warnings, invalid_warning),
                key=_temporal_warning_sort_key,
            )
        )
        return SameValueLineage(
            outcome=LineageOutcome.UNRESOLVED,
            surviving_candidate_ids=group.candidate_ids,
            superseded_candidate_ids=(),
            requires_human_review=True,
            warnings=warnings,
        )

    graph_analysis = _analyze_lineage_graph(graph.successors)
    if graph_analysis.cyclic_component_ids:
        cycle_candidate_ids = {
            candidate_id
            for component_id in graph_analysis.cyclic_component_ids
            for candidate_id in graph_analysis.components[component_id]
        }
        cycle_evidence_refs: set[str] = set()
        cyclic_component_ids = set(graph_analysis.cyclic_component_ids)
        for (predecessor_id, successor_id), edge in graph.edges.items():
            component_id = graph_analysis.component_by_node[predecessor_id]
            if (
                component_id in cyclic_component_ids
                and graph_analysis.component_by_node[successor_id] == component_id
            ):
                cycle_evidence_refs.update(edge.explicit_evidence_refs)
                cycle_evidence_refs.update(edge.interval_evidence_refs)
        warning = _lineage_warning(
            cycle_candidate_ids,
            cycle_evidence_refs,
            "same-value supersedes lineage contains a cycle",
        )
        warnings = tuple(
            sorted(
                (*invalid_temporal_warnings, warning),
                key=_temporal_warning_sort_key,
            )
        )
        return SameValueLineage(
            outcome=LineageOutcome.UNRESOLVED,
            surviving_candidate_ids=(),
            superseded_candidate_ids=(),
            requires_human_review=True,
            warnings=warnings,
        )

    predecessors = _lineage_predecessors(graph)
    current_ancestor_flags, deprecated_ancestor_flags = (
        _semantic_ancestor_flags(
            predecessors,
            graph_analysis.topological_order,
            candidates,
            is_current_semantic,
        )
    )
    survivors = tuple(
        sorted(
            candidate_id
            for candidate_id, successor_ids in graph.successors.items()
            if not successor_ids
        )
    )
    current_tips = tuple(
        tip_id
        for tip_id in survivors
        if is_current_semantic(candidates[tip_id])
    )
    deprecated_tips = tuple(
        tip_id
        for tip_id in survivors
        if _is_deprecated_semantic(candidates[tip_id])
    )
    neutral_tips = tuple(
        tip_id
        for tip_id in survivors
        if tip_id not in set(current_tips) | set(deprecated_tips)
    )

    incompatible_tips = bool(set(current_tips) & set(deprecated_tips)) or bool(
        current_tips and deprecated_tips
    )
    neutral_replaces_semantic = any(
        current_ancestor_flags[tip_id] or deprecated_ancestor_flags[tip_id]
        for tip_id in neutral_tips
    )
    unresolved = incompatible_tips or neutral_replaces_semantic
    if invalid_temporal_warnings and (
        current_tips or not deprecated_tips
    ):
        unresolved = True
    warning_items = list(invalid_temporal_warnings)
    if incompatible_tips:
        fork_warning = _explicit_incompatible_fork_warning(
            graph,
            graph_analysis.topological_order,
            current_tips,
            deprecated_tips,
        )
        if fork_warning is not None:
            warning_items.append(fork_warning)
    unresolved_warnings = tuple(
        sorted(warning_items, key=_temporal_warning_sort_key)
    )
    outcome = LineageOutcome.NEUTRAL
    if not unresolved and current_tips:
        unsupported_current_reactivation = any(
            deprecated_ancestor_flags[tip_id]
            and not _is_activation_capable(
                candidates[tip_id],
                assessments[tip_id],
                selected_policy,
            )
            for tip_id in current_tips
        )
        unresolved = unsupported_current_reactivation
        if not unresolved:
            reactivated = any(
                deprecated_ancestor_flags[tip_id]
                for tip_id in current_tips
            )
            outcome = (
                LineageOutcome.REACTIVATED
                if reactivated
                else LineageOutcome.CURRENT
            )
    elif not unresolved and deprecated_tips:
        outcome = LineageOutcome.DEPRECATED
    if unresolved:
        return SameValueLineage(
            outcome=LineageOutcome.UNRESOLVED,
            surviving_candidate_ids=survivors,
            superseded_candidate_ids=(),
            requires_human_review=True,
            warnings=unresolved_warnings,
        )

    if outcome in (LineageOutcome.CURRENT, LineageOutcome.REACTIVATED):
        winning_tip_ids = current_tips
    elif outcome is LineageOutcome.DEPRECATED:
        winning_tip_ids = deprecated_tips
    else:
        winning_tip_ids = survivors
    winning_ancestors = _collect_winning_ancestors(
        predecessors,
        winning_tip_ids,
    )
    return SameValueLineage(
        outcome=outcome,
        surviving_candidate_ids=survivors,
        superseded_candidate_ids=winning_ancestors,
        requires_human_review=False,
    )


_CANDIDATE_EVIDENCE_FIELDS = (
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
)


def _group_evidence_refs(group: CandidateGroup) -> tuple[str, ...]:
    refs = {
        candidate.source_ref
        for candidate in group.candidates
    }
    refs.update(
        getattr(candidate, field_name).source
        for candidate in group.candidates
        for field_name in _CANDIDATE_EVIDENCE_FIELDS
    )
    return tuple(sorted(refs))


def _decision_warnings(
    assessments: Mapping[str, TemporalAssessment],
    lineage: SameValueLineage,
    additional: Sequence[ReconciliationWarning] = (),
) -> tuple[ReconciliationWarning, ...]:
    warnings = {
        _temporal_warning_sort_key(warning): warning
        for assessment in assessments.values()
        for warning in assessment.warnings
    }
    warnings.update(
        {
            _temporal_warning_sort_key(warning): warning
            for warning in lineage.warnings
        }
    )
    warnings.update(
        {
            _temporal_warning_sort_key(warning): warning
            for warning in additional
        }
    )
    return tuple(warnings[key] for key in sorted(warnings))


_TEMPORAL_PENDING_PRIORITY = (
    WarningCode.INVALID_TEMPORAL_ORDER,
    WarningCode.FUTURE_CLOCK_SKEW,
    WarningCode.FUTURE_VALIDITY,
    WarningCode.EXPIRED_WITHOUT_REPLACEMENT,
)

_TEMPORAL_PENDING_REASONS = {
    WarningCode.INVALID_TEMPORAL_ORDER: "candidate interval is invalid",
    WarningCode.FUTURE_CLOCK_SKEW: (
        "candidate observation exceeds allowed future clock skew"
    ),
    WarningCode.FUTURE_VALIDITY: "candidate is not yet valid",
    WarningCode.EXPIRED_WITHOUT_REPLACEMENT: (
        "candidate expired without an eligible replacement"
    ),
}


def _nonknown_warning_code(status: EvidenceStatus) -> WarningCode:
    if status is EvidenceStatus.UNKNOWN:
        return WarningCode.EVIDENCE_UNKNOWN
    if status is EvidenceStatus.UNAVAILABLE:
        return WarningCode.EVIDENCE_UNAVAILABLE
    raise ReconciliationInputError("non-known warning requires non-known evidence")


def _evidence_state_warning(
    candidate: MemoryCandidate,
    field_name: str,
    *,
    requires_human_review: bool,
) -> ReconciliationWarning:
    evidence = getattr(candidate, field_name)
    message = (
        f"{field_name} is {evidence.status.value} and remains optional provenance"
        if field_name == "metadata"
        else (
            f"{field_name} is {evidence.status.value} and prevents current "
            "classification"
        )
    )
    return _make_warning(
        _nonknown_warning_code(evidence.status),
        message,
        (candidate.candidate_id,),
        (evidence.source,),
        requires_human_review=requires_human_review,
    )


def _status_hint_could_be_current(
    candidate: MemoryCandidate,
    policy: ReconciliationPolicy,
) -> bool:
    return candidate.source_type is SourceType.CURRENT_EVIDENCE or (
        policy.allow_explicit_source_of_truth_override
        and candidate.source_type is SourceType.USER_EXPLICIT
        and candidate.explicit_user_instruction.status is EvidenceStatus.KNOWN
        and candidate.explicit_user_instruction.value is True
    )


def _nonknown_semantic_warnings(
    group: CandidateGroup,
    policy: ReconciliationPolicy,
) -> tuple[ReconciliationWarning, ...]:
    warnings: list[ReconciliationWarning] = []
    for candidate in group.candidates:
        if candidate.value.status is not EvidenceStatus.KNOWN:
            warnings.append(
                _evidence_state_warning(
                    candidate,
                    "value",
                    requires_human_review=_is_current_semantic(
                        candidate,
                        policy,
                    ),
                )
            )
        if candidate.metadata.status is not EvidenceStatus.KNOWN:
            warnings.append(
                _evidence_state_warning(
                    candidate,
                    "metadata",
                    requires_human_review=False,
                )
            )
        if candidate.status_hint.status is not EvidenceStatus.KNOWN:
            warnings.append(
                _evidence_state_warning(
                    candidate,
                    "status_hint",
                    requires_human_review=_status_hint_could_be_current(
                        candidate,
                        policy,
                    ),
                )
            )
        if candidate.confidence.status is not EvidenceStatus.KNOWN:
            warnings.append(
                _evidence_state_warning(
                    candidate,
                    "confidence",
                    requires_human_review=_is_current_semantic(
                        candidate,
                        policy,
                    ),
                )
            )
        if (
            policy.allow_explicit_source_of_truth_override
            and candidate.source_type is SourceType.USER_EXPLICIT
            and candidate.status_hint.status is EvidenceStatus.KNOWN
            and candidate.status_hint.value
            is CandidateStatusHint.SOURCE_OF_TRUTH
            and candidate.explicit_user_instruction.status
            is not EvidenceStatus.KNOWN
        ):
            warnings.append(
                _evidence_state_warning(
                    candidate,
                    "explicit_user_instruction",
                    requires_human_review=True,
                )
            )
    return tuple(sorted(warnings, key=_temporal_warning_sort_key))


def _optional_metadata_warnings(
    group: CandidateGroup,
) -> tuple[ReconciliationWarning, ...]:
    return tuple(
        _evidence_state_warning(
            candidate,
            "metadata",
            requires_human_review=False,
        )
        for candidate in group.candidates
        if candidate.metadata.status is not EvidenceStatus.KNOWN
    )


def _unresolved_candidate_sort_key(
    record: UnresolvedCandidate,
) -> tuple[str, ...]:
    return (
        record.candidate_id,
        record.evidence_status.value,
        record.subject,
        record.predicate,
        record.related_fact_id,
        record.source_type.value,
        record.source_ref,
        record.field_source,
        record.reason,
    )


def _unresolved_value_record(
    group: CandidateGroup,
    candidate: MemoryCandidate,
) -> UnresolvedCandidate:
    evidence = candidate.value
    if evidence.status is EvidenceStatus.KNOWN:
        raise ReconciliationInputError(
            "unresolved value record requires UNKNOWN or UNAVAILABLE evidence"
        )
    return UnresolvedCandidate(
        candidate_id=candidate.candidate_id,
        subject=candidate.subject,
        predicate=candidate.predicate,
        evidence_status=evidence.status,
        reason=evidence.reason,
        source_type=candidate.source_type,
        source_ref=candidate.source_ref,
        field_source=evidence.source,
        related_fact_id=group.fact_id,
    )


def _derive_unresolved_candidates(
    decisions: Sequence[FactDecision],
) -> tuple[UnresolvedCandidate, ...]:
    """Derive unresolved value records only from authenticated decisions."""

    decision_snapshot = _guarded_sequence_snapshot(
        decisions,
        "decisions",
        "FactDecision records",
    )
    if any(type(decision) is not FactDecision for decision in decision_snapshot):
        raise ReconciliationInputError(
            "decisions must contain only exact FactDecision records"
        )
    records: list[UnresolvedCandidate] = []
    for decision in decision_snapshot:
        canonical = _validate_decision_candidate_context(decision)
        group = canonical._candidate_context.group
        non_known = tuple(
            candidate
            for candidate in group.candidates
            if candidate.value.status is not EvidenceStatus.KNOWN
        )
        if non_known and canonical.status is not ReconciliationStatus.PENDING:
            raise ReconciliationInputError(
                "non-known value candidates must produce a PENDING fact"
            )
        records.extend(
            _unresolved_value_record(group, candidate)
            for candidate in non_known
        )
    return tuple(sorted(records, key=_unresolved_candidate_sort_key))


def resolve_non_known_groups(
    groups: Sequence[CandidateGroup],
    now: datetime,
    policy: ReconciliationPolicy,
) -> tuple[
    tuple[FactDecision, ...],
    tuple[UnresolvedCandidate, ...],
    tuple[ReconciliationWarning, ...],
]:
    """Classify non-known value groups without losing their provenance."""

    group_snapshot = _guarded_sequence_snapshot(
        groups,
        "groups",
        "CandidateGroup records",
    )
    if not group_snapshot:
        return (), (), ()
    if any(type(group) is not CandidateGroup for group in group_snapshot):
        raise ReconciliationInputError(
            "groups must contain only exact CandidateGroup records"
        )

    cohorts: list[_ReconciliationCohort] = []
    for group in group_snapshot:
        _validate_candidate_group_stage(group)
        cohorts.append(
            _validate_reconciliation_cohort(
                object.__getattribute__(group, "_cohort"),
                require_bound=False,
            )
        )
    cohort = cohorts[0]
    if any(item.fingerprint != cohort.fingerprint for item in cohorts[1:]):
        raise ReconciliationInputError(
            "groups must belong to one reconciliation cohort"
        )

    decisions: list[FactDecision] = []
    warning_index: dict[
        tuple[object, ...], ReconciliationWarning
    ] = {}
    for group in group_snapshot:
        if any(
            candidate.value.status is EvidenceStatus.KNOWN
            for candidate in group.candidates
        ):
            raise ReconciliationInputError(
                "resolve_non_known_groups accepts only non-known value groups"
        )
        decision = classify_group(group, now, policy)
        decisions.append(decision)
        for warning in decision.warnings:
            warning_index[_temporal_warning_sort_key(warning)] = warning

    checked_decisions = tuple(
        sorted(decisions, key=lambda decision: decision.fact_id)
    )
    return (
        checked_decisions,
        _derive_unresolved_candidates(checked_decisions),
        tuple(warning_index[key] for key in sorted(warning_index)),
    )


def _insufficient_confidence_warnings(
    candidates: Sequence[MemoryCandidate],
) -> tuple[ReconciliationWarning, ...]:
    warnings = tuple(
        _make_warning(
            WarningCode.INSUFFICIENT_CONFIDENCE,
            "confidence is below the active threshold",
            (candidate.candidate_id,),
            (candidate.confidence.source,),
            requires_human_review=False,
        )
        for candidate in candidates
    )
    return tuple(sorted(warnings, key=_temporal_warning_sort_key))


def _intent_hint(group: CandidateGroup) -> CandidateStatusHint | None:
    for candidate in group.candidates:
        if (
            candidate.status_hint.status is EvidenceStatus.KNOWN
            and candidate.status_hint.value
            in (CandidateStatusHint.PLAN, CandidateStatusHint.HYPOTHESIS)
        ):
            return candidate.status_hint.value
    return None


def _critical_evidence_reason(
    warnings: Sequence[ReconciliationWarning],
) -> str | None:
    for warning in warnings:
        if (
            warning.code
            in (WarningCode.EVIDENCE_UNKNOWN, WarningCode.EVIDENCE_UNAVAILABLE)
            and warning.requires_human_review
        ):
            return warning.message
    return None


def _select_maximum_confidence(
    candidates: Sequence[MemoryCandidate],
) -> EvidenceValue[float]:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    known_candidates = tuple(
        candidate
        for candidate in ordered
        if candidate.confidence.status is EvidenceStatus.KNOWN
    )
    if known_candidates:
        return max(
            known_candidates,
            key=lambda item: item.confidence.value,
        ).confidence
    unknown_candidates = tuple(
        candidate
        for candidate in ordered
        if candidate.confidence.status is EvidenceStatus.UNKNOWN
    )
    if unknown_candidates:
        return unknown_candidates[0].confidence
    return ordered[0].confidence


def _select_latest_valid_from(
    candidates: Sequence[MemoryCandidate],
) -> EvidenceValue[str]:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    known_candidates = tuple(
        candidate
        for candidate in ordered
        if candidate.valid_from.status is EvidenceStatus.KNOWN
    )
    if known_candidates:
        return max(
            known_candidates,
            key=lambda item: parse_known_timestamp(
                item.valid_from,
                "valid_from",
            ),
        ).valid_from
    unknown_candidates = tuple(
        candidate
        for candidate in ordered
        if candidate.valid_from.status is EvidenceStatus.UNKNOWN
    )
    if unknown_candidates:
        return unknown_candidates[0].valid_from
    return ordered[0].valid_from


def _active_resolution_method(
    group: CandidateGroup,
    lineage: SameValueLineage,
) -> ResolutionMethod:
    if lineage.outcome is LineageOutcome.REACTIVATED:
        return ResolutionMethod.SAME_VALUE_REACTIVATION
    if len(group.candidates) > 1:
        return ResolutionMethod.SAME_VALUE_MERGE
    return ResolutionMethod.DIRECT_CURRENT


def _active_reason(method: ResolutionMethod) -> str:
    if method is ResolutionMethod.SAME_VALUE_REACTIVATION:
        return "same-value lineage was reactivated by an activation witness"
    if method is ResolutionMethod.SAME_VALUE_MERGE:
        return "same typed value merged with an activation witness"
    return "candidate independently satisfies every ACTIVE gate"


def _derive_fact_decision_from_group(
    group: CandidateGroup,
    now: datetime,
    policy: ReconciliationPolicy,
) -> FactDecision:
    if type(group) is not CandidateGroup:
        raise ReconciliationInputError("group must be a CandidateGroup")
    _validate_candidate_group_stage(group)
    if not isinstance(policy, ReconciliationPolicy):
        raise ReconciliationInputError("policy must be a ReconciliationPolicy")
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise ReconciliationInputError("now must be a timezone-aware datetime")
    try:
        reconciled_at = now.astimezone(timezone.utc)
    except OverflowError as error:
        raise ReconciliationInputError(
            "now cannot be normalized to UTC"
        ) from error

    assessments = {
        candidate.candidate_id: assess_temporal(
            candidate,
            reconciled_at,
            policy,
        )
        for candidate in group.candidates
    }
    lineage = resolve_same_value_lineage(group, reconciled_at, policy)
    surviving_candidate_ids = set(lineage.surviving_candidate_ids)
    local_witnesses = tuple(
        candidate
        for candidate in find_activation_witnesses(
            group,
            assessments,
            policy,
        )
        if candidate.candidate_id in surviving_candidate_ids
    )
    warnings = _decision_warnings(
        assessments,
        lineage,
        _optional_metadata_warnings(group),
    )

    if (
        local_witnesses
        and lineage.outcome
        not in (LineageOutcome.DEPRECATED, LineageOutcome.UNRESOLVED)
    ):
        method = _active_resolution_method(group, lineage)
        status = ReconciliationStatus.ACTIVE
        reason = _active_reason(method)
        requires_human_review = False
        witnesses = local_witnesses
    elif (
        lineage.outcome is not LineageOutcome.UNRESOLVED
        and (
            lineage.outcome is LineageOutcome.DEPRECATED
            or any(
                candidate.candidate_id in lineage.surviving_candidate_ids
                and _is_deprecated_semantic(candidate)
                for candidate in group.candidates
            )
        )
    ):
        status = ReconciliationStatus.DEPRECATED
        method = ResolutionMethod.EXPLICIT_DEPRECATION
        reason = "explicit deprecation"
        requires_human_review = any(
            warning.requires_human_review for warning in warnings
        ) or lineage.requires_human_review
        witnesses = ()
    else:
        status = ReconciliationStatus.PENDING
        witnesses = ()
        semantic_warnings = _nonknown_semantic_warnings(group, policy)
        low_confidence_candidates = tuple(
            candidate
            for candidate in group.candidates
            if _is_current_semantic(candidate, policy)
            and candidate.confidence.status is EvidenceStatus.KNOWN
            and candidate.confidence.value
            < policy.active_confidence_threshold
        )
        warnings = _decision_warnings(
            assessments,
            lineage,
            (
                *semantic_warnings,
                *_insufficient_confidence_warnings(
                    low_confidence_candidates
                ),
            ),
        )
        intent_hint = _intent_hint(group)
        temporal_codes = {warning.code for warning in warnings}
        temporal_code = next(
            (
                code
                for code in _TEMPORAL_PENDING_PRIORITY
                if code in temporal_codes
            ),
            None,
        )

        if temporal_code is not None:
            method = ResolutionMethod.TEMPORAL_PENDING
            reason = _TEMPORAL_PENDING_REASONS[temporal_code]
        elif low_confidence_candidates:
            method = ResolutionMethod.INSUFFICIENT_EVIDENCE
            reason = (
                "current evidence confidence is below the activation threshold"
            )
        elif intent_hint is not None:
            method = ResolutionMethod.PENDING_SEMANTICS
            reason = (
                f"{intent_hint.value} is intent, not current fact evidence"
            )
        elif (critical_reason := _critical_evidence_reason(warnings)) is not None:
            method = ResolutionMethod.PENDING_SEMANTICS
            reason = critical_reason
        elif lineage.outcome is LineageOutcome.UNRESOLVED:
            method = ResolutionMethod.PENDING_SEMANTICS
            reason = "same-value candidate lineage is unresolved"
        elif not any(
            _is_current_semantic(candidate, policy)
            for candidate in group.candidates
        ):
            method = ResolutionMethod.PENDING_SEMANTICS
            reason = "candidate group does not satisfy ACTIVE gates"
        else:
            method = ResolutionMethod.PENDING_SEMANTICS
            reason = "candidate group does not satisfy ACTIVE gates"
        requires_human_review = any(
            warning.requires_human_review for warning in warnings
        ) or lineage.requires_human_review

    selected_candidate = witnesses[0] if witnesses else group.candidates[0]
    confidence_candidates = witnesses if witnesses else group.candidates
    valid_from_candidates = witnesses if witnesses else group.candidates
    decision = FactDecision._from_canonical_group(
        _FACT_DECISION_CONSTRUCTION_TOKEN,
        fact_id=group.fact_id,
        subject=group.subject,
        predicate=group.predicate,
        selected_value=selected_candidate.value,
        candidate_ids=group.candidate_ids,
        evidence_refs=_group_evidence_refs(group),
        activation_witness_candidate_ids=tuple(
            candidate.candidate_id for candidate in witnesses
        ),
        superseded_candidate_ids=lineage.superseded_candidate_ids,
        status=status,
        confidence=_select_maximum_confidence(confidence_candidates),
        valid_from=_select_latest_valid_from(valid_from_candidates),
        resolution_method=method,
        reason=reason,
        requires_human_review=requires_human_review,
        warnings=warnings,
    )
    object.__setattr__(
        decision,
        "_candidate_context",
        _DecisionCandidateContext(group, policy, reconciled_at),
    )
    object.__setattr__(
        decision,
        "_cohort",
        _bind_decision_cohort(group._cohort, policy, reconciled_at),
    )
    return decision


def _make_classified_decision_builder(
    issue_stage: Callable[..., object],
) -> Callable[..., FactDecision]:
    def build_classified_decision(
        group: CandidateGroup,
        now: datetime,
        policy: ReconciliationPolicy,
    ) -> FactDecision:
        decision = _derive_fact_decision_from_group(group, now, policy)
        issue_stage(
            decision,
            cohort=decision._cohort,
            previous_stage=_ReconciliationStage.GROUPED,
            parent_fingerprint=group._stage_auth.fingerprint,
            payload=_decision_stage_payload(decision),
            graph_build_token=None,
        )
        return decision

    return build_classified_decision


_fact_decision_from_group = _make_classified_decision_builder(
    _classified_stage_builder
)
del _make_classified_decision_builder
del _classified_stage_builder


def classify_group(
    group: CandidateGroup,
    now: datetime,
    policy: ReconciliationPolicy,
) -> FactDecision:
    """Classify one same-value group through canonical decision construction."""

    return _fact_decision_from_group(group, now, policy)


def _copy_trusted_utc_clock(value: object) -> datetime:
    if type(value) is not datetime:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        )
    try:
        offset = value.utcoffset()
        if type(offset) is not timedelta or offset != timedelta(0):
            raise ValueError("reconciliation clock must be UTC-aware")
        return datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=timezone.utc,
            fold=value.fold,
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as error:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context and stage authentication"
        ) from error


def _validate_decision_stage_auth(
    decision: FactDecision,
    cohort: _ReconciliationCohort,
) -> None:
    try:
        authorization = object.__getattribute__(decision, "_stage_auth")
        stage = object.__getattribute__(authorization, "stage")
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "decision is not stage authenticated"
        ) from error
    previous_by_stage = {
        _ReconciliationStage.CLASSIFIED: _ReconciliationStage.GROUPED,
        _ReconciliationStage.COORDINATED: _ReconciliationStage.CLASSIFIED,
    }
    if type(stage) is not _ReconciliationStage or stage not in previous_by_stage:
        raise ReconciliationInputError("decision stage is invalid")
    _validate_stage_authorization(
        decision,
        cohort=cohort,
        expected_stage=stage,
        expected_previous_stage=previous_by_stage[stage],
        payload=_decision_stage_payload(decision),
        graph_build_token=None,
    )


def _validate_decision_candidate_context(
    decision: FactDecision,
    *,
    require_stage_auth: bool = True,
    validate_identity: bool = True,
) -> FactDecision:
    try:
        context = object.__getattribute__(decision, "_candidate_context")
        decision_cohort = object.__getattribute__(decision, "_cohort")
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context for exact "
            "canonical FactDecision records"
        ) from error
    try:
        fact_id = object.__getattribute__(decision, "fact_id")
        subject = object.__getattribute__(decision, "subject")
        predicate = object.__getattribute__(decision, "predicate")
        selected_value = object.__getattribute__(decision, "selected_value")
        candidate_ids = object.__getattribute__(decision, "candidate_ids")
        evidence_refs = object.__getattribute__(decision, "evidence_refs")
        activation_witness_ids = object.__getattribute__(
            decision,
            "activation_witness_candidate_ids",
        )
        superseded_candidate_ids = object.__getattribute__(
            decision,
            "superseded_candidate_ids",
        )
        status = object.__getattribute__(decision, "status")
        confidence = object.__getattribute__(decision, "confidence")
        valid_from = object.__getattribute__(decision, "valid_from")
        resolution_method = object.__getattribute__(
            decision,
            "resolution_method",
        )
        reason = object.__getattribute__(decision, "reason")
        requires_human_review = object.__getattribute__(
            decision,
            "requires_human_review",
        )
        warnings = object.__getattribute__(decision, "warnings")
        relation_requests = object.__getattribute__(
            decision,
            "relation_requests",
        )
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "decisions must be complete canonical FactDecision records"
        ) from error
    for field_name, value in (
        ("fact_id", fact_id),
        ("subject", subject),
        ("predicate", predicate),
        ("reason", reason),
    ):
        if type(value) is not str:
            raise ReconciliationInputError(
                f"canonical FactDecision {field_name} has an invalid type"
            )
    for field_name, value in (
        ("candidate_ids", candidate_ids),
        ("evidence_refs", evidence_refs),
        ("activation_witness_candidate_ids", activation_witness_ids),
        ("superseded_candidate_ids", superseded_candidate_ids),
    ):
        if type(value) is not tuple or any(
            type(item) is not str for item in value
        ):
            raise ReconciliationInputError(
                f"canonical FactDecision {field_name} has an invalid type"
            )
    for field_name, value, expected_type in (
        ("selected_value", selected_value, EvidenceValue),
        ("status", status, ReconciliationStatus),
        ("confidence", confidence, EvidenceValue),
        ("valid_from", valid_from, EvidenceValue),
        ("resolution_method", resolution_method, ResolutionMethod),
        ("requires_human_review", requires_human_review, bool),
    ):
        if type(value) is not expected_type:
            raise ReconciliationInputError(
                f"canonical FactDecision {field_name} has an invalid type"
            )
    if type(warnings) is not tuple or any(
        type(warning) is not ReconciliationWarning for warning in warnings
    ):
        raise ReconciliationInputError(
            "canonical FactDecision warnings has an invalid type"
        )
    if type(relation_requests) is not tuple:
        raise ReconciliationInputError(
            "canonical FactDecision relation_requests has an invalid type"
        )
    if type(context) is not _DecisionCandidateContext:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        )
    try:
        group = context.group
        retained_policy = context.policy
        reconciled_at = context.reconciled_at
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        ) from error
    if (
        type(group) is not CandidateGroup
        or type(retained_policy) is not ReconciliationPolicy
    ):
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        )
    trusted_reconciled_at = _copy_trusted_utc_clock(reconciled_at)
    canonical_policy = _canonicalize_reconciliation_policy(
        retained_policy,
        "candidate context policy",
    )
    try:
        project_id = group.project_id
        group_subject = group.subject
        group_predicate = group.predicate
        group_fact_id = group.fact_id
        canonical_value_key = group.canonical_value_key
        candidates = group.candidates
        group_cohort = object.__getattribute__(group, "_cohort")
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        ) from error
    if (
        type(project_id) is not str
        or type(group_subject) is not str
        or type(group_predicate) is not str
        or type(group_fact_id) is not str
        or type(canonical_value_key) is not str
        or type(candidates) is not tuple
        or not candidates
        or any(
            type(candidate) is not MemoryCandidate
            for candidate in candidates
        )
    ):
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context"
        )
    try:
        checked_group_cohort = _validate_reconciliation_cohort(
            group_cohort,
            require_bound=False,
        )
        checked_decision_cohort = _validate_reconciliation_cohort(
            decision_cohort,
            require_bound=True,
        )
        if require_stage_auth:
            _validate_decision_stage_auth(decision, checked_decision_cohort)
        _validate_candidate_group_stage(group)
        expected_decision_cohort = _bind_decision_cohort(
            checked_group_cohort,
            canonical_policy,
            trusted_reconciled_at,
        )
        if (
            checked_group_cohort.project_id != project_id
            or checked_decision_cohort.fingerprint
            != expected_decision_cohort.fingerprint
        ):
            raise ReconciliationInputError(
                "decision and candidate group cohort do not match"
            )
        canonical_group = _build_candidate_group(
            project_id,
            candidates,
            checked_group_cohort,
        )
        if (
            canonical_group.subject != group_subject
            or canonical_group.predicate != group_predicate
            or canonical_group.fact_id != group_fact_id
            or canonical_group.canonical_value_key != canonical_value_key
        ):
            raise ReconciliationInputError(
                "candidate group identity does not match its candidate context"
            )
        canonical = _fact_decision_from_group(
            canonical_group,
            trusted_reconciled_at,
            canonical_policy,
        )
        if validate_identity and (
            fact_id != canonical.fact_id
            or subject != canonical.subject
            or predicate != canonical.predicate
            or candidate_ids != canonical.candidate_ids
            or evidence_refs != canonical.evidence_refs
            or _evidence_payload_signature(
                selected_value,
                "selected_value",
                "json",
            )
            != _evidence_payload_signature(
                canonical.selected_value,
                "selected_value",
                "json",
            )
        ):
            raise ReconciliationInputError(
                "decision identity does not match its candidate context"
            )
    except (ReconciliationInputError, AttributeError, TypeError, ValueError) as error:
        raise ReconciliationInputError(
            "decisions must retain canonical candidate context and stage authentication"
        ) from error
    return canonical


def _canonicalize_reconciliation_policy(
    policy: object,
    field_name: str,
) -> ReconciliationPolicy:
    if type(policy) is not ReconciliationPolicy:
        raise ReconciliationInputError(
            f"{field_name} must be an exact ReconciliationPolicy"
        )
    try:
        precedence = object.__getattribute__(policy, "source_precedence")
        threshold = object.__getattribute__(
            policy,
            "active_confidence_threshold",
        )
        allow_source_of_truth = object.__getattribute__(
            policy,
            "allow_explicit_source_of_truth_override",
        )
        allow_current = object.__getattribute__(
            policy,
            "allow_current_evidence_over_historical",
        )
        require_explicit = object.__getattribute__(
            policy,
            "require_explicit_hint_for_user_override",
        )
        conflict_on_equal = object.__getattribute__(
            policy,
            "conflict_on_equal_precedence_disagreement",
        )
        skew = object.__getattribute__(
            policy,
            "max_future_clock_skew_seconds",
        )
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(f"{field_name} is invalid") from error
    if type(precedence) is not tuple:
        raise ReconciliationInputError(f"{field_name} is invalid")
    normalized_precedence: list[tuple[SourceType, int]] = []
    for entry in precedence:
        if type(entry) is not tuple or len(entry) != 2:
            raise ReconciliationInputError(f"{field_name} is invalid")
        source_type, rank = entry
        if type(source_type) is not SourceType or type(rank) is not int:
            raise ReconciliationInputError(f"{field_name} is invalid")
        normalized_precedence.append((source_type, rank))

    boolean_values = (
        allow_source_of_truth,
        allow_current,
        require_explicit,
        conflict_on_equal,
    )
    if (
        type(threshold) not in (int, float)
        or type(skew) is not int
        or any(
            type(boolean_value) is not bool
            for boolean_value in boolean_values
        )
    ):
        raise ReconciliationInputError(f"{field_name} is invalid")

    try:
        return ReconciliationPolicy(
            source_precedence=tuple(normalized_precedence),
            active_confidence_threshold=threshold,
            allow_explicit_source_of_truth_override=allow_source_of_truth,
            allow_current_evidence_over_historical=allow_current,
            require_explicit_hint_for_user_override=require_explicit,
            conflict_on_equal_precedence_disagreement=conflict_on_equal,
            max_future_clock_skew_seconds=skew,
        )
    except (ReconciliationInputError, TypeError, ValueError) as error:
        raise ReconciliationInputError(f"{field_name} is invalid") from error


def _require_coherent_decision_cohort(
    decisions: tuple[FactDecision, ...],
    policy: ReconciliationPolicy,
) -> _ReconciliationCohort | None:
    if not decisions:
        return None
    expected_policy_fingerprint = _policy_fingerprint(policy)
    cohorts: list[_ReconciliationCohort] = []
    for decision in decisions:
        try:
            cohort = object.__getattribute__(decision, "_cohort")
        except (AttributeError, TypeError) as error:
            raise ReconciliationInputError(
                "decisions must retain their reconciliation cohort"
            ) from error
        checked = _validate_reconciliation_cohort(
            cohort,
            require_bound=True,
        )
        if checked.policy_fingerprint != expected_policy_fingerprint:
            raise ReconciliationInputError(
                "decisions must use the resolver policy cohort"
            )
        cohorts.append(checked)
    first = cohorts[0]
    if any(cohort.fingerprint != first.fingerprint for cohort in cohorts[1:]):
        raise ReconciliationInputError(
            "decisions must belong to one reconciliation cohort"
        )
    return first


def _validate_complete_decision_cohort(
    decisions: tuple[FactDecision, ...],
) -> _ReconciliationCohort:
    """Verify that decisions are the complete, duplicate-free invocation set."""

    if type(decisions) is not tuple or not decisions:
        raise ReconciliationInputError(
            "decisions must retain one complete reconciliation cohort"
        )
    cohorts: list[_ReconciliationCohort] = []
    candidate_ids: list[str] = []
    for decision in decisions:
        if type(decision) is not FactDecision:
            raise ReconciliationInputError(
                "decisions must contain only exact FactDecision records"
            )
        try:
            cohort = _validate_reconciliation_cohort(
                object.__getattribute__(decision, "_cohort"),
                require_bound=True,
            )
            context = object.__getattribute__(decision, "_candidate_context")
            group = object.__getattribute__(context, "group")
            if type(group) is not CandidateGroup:
                raise ReconciliationInputError(
                    "decision candidate group is invalid"
                )
            _validate_candidate_group_stage(group)
            group_candidates_snapshot = object.__getattribute__(
                group,
                "candidates",
            )
            if type(group_candidates_snapshot) is not tuple:
                raise ReconciliationInputError(
                    "decision candidate group is invalid"
                )
            for candidate in group_candidates_snapshot:
                if type(candidate) is not MemoryCandidate:
                    raise ReconciliationInputError(
                        "decision candidate group is invalid"
                    )
                candidate_id = object.__getattribute__(
                    candidate,
                    "candidate_id",
                )
                _validate_required_text(candidate_id, "candidate_id")
                candidate_ids.append(candidate_id)
        except (AttributeError, TypeError, ValueError) as error:
            raise ReconciliationInputError(
                "decisions must retain one complete reconciliation cohort"
            ) from error
        cohorts.append(cohort)
    first = cohorts[0]
    if any(cohort.fingerprint != first.fingerprint for cohort in cohorts[1:]):
        raise ReconciliationInputError(
            "decisions must belong to one reconciliation cohort"
        )
    expected_namespace = _candidate_id_namespace(
        first.project_id,
        tuple(candidate_ids),
    )
    if expected_namespace != first.candidate_id_namespace:
        raise ReconciliationInputError(
            "decisions must retain the complete cohort candidate ID namespace"
        )
    expected_fact_ids = tuple(
        sorted(
            fact_id
            for entry in first.predicate_membership
            for fact_id in entry[3]
        )
    )
    actual_fact_ids = tuple(sorted(decision.fact_id for decision in decisions))
    if actual_fact_ids != expected_fact_ids:
        raise ReconciliationInputError(
            "decisions must retain the complete cohort fact membership"
        )
    return first


def _validate_complete_predicate_membership(
    decisions: tuple[FactDecision, ...],
) -> None:
    if not decisions:
        raise ReconciliationInputError(
            "predicate resolution requires non-empty complete membership"
        )
    cohorts = tuple(
        _validate_reconciliation_cohort(
            object.__getattribute__(decision, "_cohort"),
            require_bound=True,
        )
        for decision in decisions
    )
    cohort = cohorts[0]
    if any(
        item.reconciled_at != cohort.reconciled_at for item in cohorts[1:]
    ):
        raise ReconciliationInputError(
            "predicate decisions must use one reconciliation clock"
        )
    if any(
        item.policy_fingerprint != cohort.policy_fingerprint
        for item in cohorts[1:]
    ):
        raise ReconciliationInputError(
            "predicate decisions must use one reconciliation policy"
        )
    if any(item.fingerprint != cohort.fingerprint for item in cohorts[1:]):
        raise ReconciliationInputError(
            "predicate decisions must belong to one reconciliation cohort"
        )
    contexts = tuple(
        object.__getattribute__(decision, "_candidate_context")
        for decision in decisions
    )
    groups = tuple(object.__getattribute__(context, "group") for context in contexts)
    identities = {
        (group.project_id, group.subject, group.predicate)
        for group in groups
    }
    if len(identities) != 1:
        raise ReconciliationInputError(
            "predicate cohort membership must use one exact subject and predicate"
        )
    identity = next(iter(identities))
    expected = next(
        (
            entry
            for entry in cohort.predicate_membership
            if entry[:3] == identity
        ),
        None,
    )
    if expected is None:
        raise ReconciliationInputError(
            "predicate membership is absent from its reconciliation cohort"
        )
    actual_fact_ids = tuple(sorted(decision.fact_id for decision in decisions))
    actual_candidate_ids = tuple(
        sorted(
            candidate.candidate_id
            for group in groups
            for candidate in group.candidates
        )
    )
    if (
        actual_fact_ids != expected[3]
        or actual_candidate_ids != expected[4]
        or _candidate_id_namespace(identity[0], actual_candidate_ids)
        != expected[5]
    ):
        raise ReconciliationInputError(
            "predicate resolution requires exact complete cohort membership"
        )


def resolve_cross_value_replacements(
    decisions: Sequence[FactDecision],
    policy: ReconciliationPolicy,
) -> tuple[RelationRequest, ...]:
    """Collect explicit cross-value supersession requests."""

    canonical_policy = _canonicalize_reconciliation_policy(
        policy,
        "resolver policy",
    )
    decision_snapshot = _guarded_sequence_snapshot(
        decisions,
        "decisions",
        "FactDecision records",
    )
    if not decision_snapshot:
        raise ReconciliationInputError(
            "cross-value resolution requires non-empty complete membership"
        )
    if not all(
        type(decision) is FactDecision for decision in decision_snapshot
    ):
        raise ReconciliationInputError(
            "decisions must contain only exact FactDecision records"
        )
    canonical_decisions: list[FactDecision] = []
    for decision in decision_snapshot:
        canonical_decisions.append(
            _validate_decision_candidate_context(
                decision,
                validate_identity=False,
            )
        )
    _validate_complete_predicate_membership(tuple(canonical_decisions))
    candidate_index: dict[str, tuple[FactDecision, MemoryCandidate]] = {}
    fact_ids: set[str] = set()
    reconciliation_clocks: set[datetime] = set()
    for decision in canonical_decisions:
        if decision.fact_id in fact_ids:
            raise ReconciliationInputError(
                f"duplicate fact_id: {decision.fact_id}"
            )
        fact_ids.add(decision.fact_id)
        context = decision._candidate_context
        if _policy_fingerprint(context.policy) != _policy_fingerprint(
            canonical_policy
        ):
            raise ReconciliationInputError(
                "decisions must use the resolver classification policy"
            )
        reconciliation_clocks.add(context.reconciled_at)
        for candidate in context.group.candidates:
            if candidate.candidate_id in candidate_index:
                raise ReconciliationInputError(
                    f"duplicate candidate_id: {candidate.candidate_id}"
                )
            candidate_index[candidate.candidate_id] = (decision, candidate)
    if len(reconciliation_clocks) > 1:
        raise ReconciliationInputError(
            "decisions must use one reconciliation clock"
        )
    _require_coherent_decision_cohort(
        tuple(canonical_decisions),
        canonical_policy,
    )

    for source_id in sorted(candidate_index):
        _, source = candidate_index[source_id]
        if source.supersedes.status is not EvidenceStatus.KNOWN:
            continue
        for target_id in sorted(source.supersedes.value):
            if target_id not in candidate_index:
                raise ReconciliationInputError(
                    f"dangling supersedes reference: {source_id} -> {target_id}"
                )
            _, target = candidate_index[target_id]
            if (
                source.subject != target.subject
                or source.predicate != target.predicate
            ):
                raise ReconciliationInputError(
                    "supersedes reference must preserve exact subject and "
                    f"predicate: {source_id} -> {target_id}"
                )

    return _drop_mutual_supersedes(
        _explicit_requests_from_canonical_decisions(canonical_decisions)
    )


_EXPLICIT_SOURCE_OF_TRUTH_REASON = (
    "explicit current source-of-truth designation"
)
_CURRENT_EVIDENCE_OVER_HISTORICAL_REASON = (
    "current direct evidence supersedes historical memory"
)
_UNRESOLVED_CURRENT_CONFLICT_REASON = "unresolved competing current facts"
_CONTRADICTORY_SUPERSEDES_REASON = (
    "contradictory explicit supersedes declarations"
)


def _with_predicate_resolution_status(
    decision: FactDecision,
    status: ReconciliationStatus,
    method: ResolutionMethod,
    reason: str,
) -> FactDecision:
    """Rebuild one canonical decision with an authorized cross-value rule."""

    if status not in (
        ReconciliationStatus.ACTIVE,
        ReconciliationStatus.SUPERSEDED,
    ):
        raise ReconciliationInputError(
            "predicate resolution status must be ACTIVE or SUPERSEDED"
        )
    expected_reasons = {
        ResolutionMethod.EXPLICIT_SUPERSEDES: (
            "explicit candidate supersedes relation"
        ),
        ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH: (
            _EXPLICIT_SOURCE_OF_TRUTH_REASON
        ),
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL: (
            _CURRENT_EVIDENCE_OVER_HISTORICAL_REASON
        ),
    }
    if method not in expected_reasons or reason != expected_reasons[method]:
        raise ReconciliationInputError(
            "predicate resolution method and reason must be coherent"
        )
    canonical = _validate_decision_candidate_context(decision)
    resolved = FactDecision._from_canonical_group(
        _FACT_DECISION_CONSTRUCTION_TOKEN,
        fact_id=canonical.fact_id,
        subject=canonical.subject,
        predicate=canonical.predicate,
        selected_value=canonical.selected_value,
        candidate_ids=canonical.candidate_ids,
        evidence_refs=canonical.evidence_refs,
        activation_witness_candidate_ids=(
            canonical.activation_witness_candidate_ids
        ),
        superseded_candidate_ids=canonical.superseded_candidate_ids,
        status=status,
        confidence=canonical.confidence,
        valid_from=canonical.valid_from,
        resolution_method=method,
        reason=reason,
        requires_human_review=canonical.requires_human_review,
        warnings=canonical.warnings,
    )
    object.__setattr__(
        resolved,
        "_candidate_context",
        canonical._candidate_context,
    )
    object.__setattr__(resolved, "_cohort", canonical._cohort)
    object.__setattr__(resolved, "_classification_parent", canonical)
    return resolved


def _with_conflicted_status(
    decision: FactDecision,
    reason: str,
    warning: ReconciliationWarning | None = None,
) -> FactDecision:
    """Rebuild one eligible decision as a review-requiring conflict."""

    if reason not in (
        _UNRESOLVED_CURRENT_CONFLICT_REASON,
        _CONTRADICTORY_SUPERSEDES_REASON,
    ):
        raise ReconciliationInputError("conflict reason is not canonical")
    canonical = _validate_decision_candidate_context(decision)
    warnings = canonical.warnings
    if warning is not None:
        warnings = tuple(
            sorted(set((*warnings, warning)), key=_temporal_warning_sort_key)
        )
    conflicted = FactDecision._from_canonical_group(
        _FACT_DECISION_CONSTRUCTION_TOKEN,
        fact_id=canonical.fact_id,
        subject=canonical.subject,
        predicate=canonical.predicate,
        selected_value=canonical.selected_value,
        candidate_ids=canonical.candidate_ids,
        evidence_refs=canonical.evidence_refs,
        activation_witness_candidate_ids=(),
        superseded_candidate_ids=canonical.superseded_candidate_ids,
        status=ReconciliationStatus.CONFLICTED,
        confidence=canonical.confidence,
        valid_from=canonical.valid_from,
        resolution_method=ResolutionMethod.UNRESOLVED_CONFLICT,
        reason=reason,
        requires_human_review=True,
        warnings=warnings,
    )
    object.__setattr__(
        conflicted,
        "_candidate_context",
        canonical._candidate_context,
    )
    object.__setattr__(conflicted, "_cohort", canonical._cohort)
    object.__setattr__(conflicted, "_classification_parent", canonical)
    return conflicted


def _finalize_predicate_payload(
    base_decisions_by_fact_id: Mapping[str, FactDecision],
    resolved_decisions_by_fact_id: Mapping[str, FactDecision],
    existing_requests: Sequence[RelationRequest],
    *,
    contradictory_fact_ids: set[str] | None = None,
) -> tuple[tuple[FactDecision, ...], tuple[RelationRequest, ...]]:
    """Atomically finalize replacements or roll them back into conflicts."""

    contradiction_ids = (
        set() if contradictory_fact_ids is None else contradictory_fact_ids
    )
    provisional_active_ids = {
        fact_id
        for fact_id, decision in resolved_decisions_by_fact_id.items()
        if decision.status is ReconciliationStatus.ACTIVE
        and decision.selected_value.status is EvidenceStatus.KNOWN
    }
    if (
        not contradiction_ids
        and len(provisional_active_ids) < 2
    ):
        return (
            tuple(
                resolved_decisions_by_fact_id[fact_id]
                for fact_id in sorted(resolved_decisions_by_fact_id)
            ),
            tuple(existing_requests),
        )

    conflict_ids = provisional_active_ids | contradiction_ids
    resolved = dict(resolved_decisions_by_fact_id)
    retained_requests = list(existing_requests)
    while True:
        invalidated = tuple(
            request
            for request in retained_requests
            if request.relation_type is RelationType.SUPERSEDES
            and request.from_fact_id in conflict_ids
        )
        if not invalidated:
            break
        invalidated_set = set(invalidated)
        retained_requests = [
            request
            for request in retained_requests
            if request not in invalidated_set
        ]
        for request in invalidated:
            restored = base_decisions_by_fact_id[request.to_fact_id]
            resolved[request.to_fact_id] = restored
            if (
                restored.status is ReconciliationStatus.ACTIVE
                and restored.selected_value.status is EvidenceStatus.KNOWN
            ):
                conflict_ids.add(restored.fact_id)

    if len(conflict_ids) < 2:
        return (
            tuple(resolved[fact_id] for fact_id in sorted(resolved)),
            tuple(retained_requests),
        )

    reason = (
        _CONTRADICTORY_SUPERSEDES_REASON
        if contradiction_ids
        else _UNRESOLVED_CURRENT_CONFLICT_REASON
    )
    ordered = tuple(
        base_decisions_by_fact_id[fact_id]
        for fact_id in sorted(conflict_ids)
    )
    warning = None
    if contradiction_ids:
        causal_decisions = tuple(
            base_decisions_by_fact_id[fact_id]
            for fact_id in sorted(contradiction_ids)
        )
        warning = _make_warning(
            WarningCode.CONTRADICTORY_SUPERSEDES,
            reason,
            tuple(
                candidate_id
                for decision in causal_decisions
                for candidate_id in decision.candidate_ids
            ),
            tuple(
                evidence_ref
                for decision in causal_decisions
                for evidence_ref in decision.evidence_refs
            ),
            requires_human_review=True,
        )

    for decision in ordered:
        resolved[decision.fact_id] = _with_conflicted_status(
            decision,
            reason,
            warning,
        )

    conflict_requests: list[RelationRequest] = []
    for left, right in combinations(ordered, 2):
        candidate_ids = tuple((*left.candidate_ids, *right.candidate_ids))
        evidence_refs = tuple((*left.evidence_refs, *right.evidence_refs))
        conflict_requests.extend(
            (
                RelationRequest.conflicts(
                    left.fact_id,
                    right.fact_id,
                    reason,
                    candidate_ids,
                    evidence_refs,
                ),
                RelationRequest.conflicts(
                    right.fact_id,
                    left.fact_id,
                    reason,
                    candidate_ids,
                    evidence_refs,
                ),
            )
        )
    return (
        tuple(resolved[fact_id] for fact_id in sorted(resolved)),
        _merge_relation_requests((*retained_requests, *conflict_requests)),
    )


def _has_atomic_source_of_truth_witness(
    decision: FactDecision,
    policy: ReconciliationPolicy,
) -> bool:
    witness_ids = set(decision.activation_witness_candidate_ids)
    return any(
        candidate.candidate_id in witness_ids
        and is_valid_source_of_truth(candidate, policy)
        for candidate in decision._candidate_context.group.candidates
    )


def _has_atomic_current_evidence_witness(decision: FactDecision) -> bool:
    witness_ids = set(decision.activation_witness_candidate_ids)
    return any(
        candidate.candidate_id in witness_ids
        and is_direct_current_evidence(candidate)
        for candidate in decision._candidate_context.group.candidates
    )


def _is_strict_historical_group(decision: FactDecision) -> bool:
    return all(
        is_strict_historical_memory(candidate)
        for candidate in decision._candidate_context.group.candidates
    )


def _is_plausible_competing_fact(decision: FactDecision) -> bool:
    if decision.selected_value.status is not EvidenceStatus.KNOWN:
        return False
    if decision.status is ReconciliationStatus.ACTIVE:
        return True
    if (
        decision.status is not ReconciliationStatus.PENDING
        or decision.requires_human_review
    ):
        return False

    context = decision._candidate_context
    return any(
        candidate.source_type is SourceType.HISTORICAL_MEMORY
        and candidate.status_hint.status is EvidenceStatus.KNOWN
        and candidate.status_hint.value is CandidateStatusHint.HISTORICAL
        and candidate.confidence.status is EvidenceStatus.KNOWN
        and not _is_deprecated_semantic(candidate)
        and assess_temporal(
            candidate,
            context.reconciled_at,
            context.policy,
        ).eligible
        for candidate in context.group.candidates
    )


def _explicit_requests_from_canonical_decisions(
    decisions: Sequence[FactDecision],
) -> tuple[RelationRequest, ...]:
    candidate_index = {
        candidate.candidate_id: (decision, candidate)
        for decision in decisions
        for candidate in decision._candidate_context.group.candidates
    }
    requests: list[RelationRequest] = []
    for source_id in sorted(candidate_index):
        source_decision, source = candidate_index[source_id]
        if (
            source.supersedes.status is not EvidenceStatus.KNOWN
            or source_id
            not in source_decision.activation_witness_candidate_ids
        ):
            continue
        for target_id in source.supersedes.value:
            target_decision, target = candidate_index[target_id]
            if source_decision.fact_id == target_decision.fact_id:
                continue
            requests.append(
                RelationRequest.supersedes(
                    source_decision.fact_id,
                    target_decision.fact_id,
                    "explicit candidate supersedes relation",
                    (source_id, target_id),
                    _candidate_evidence_refs(source, target),
                )
            )
    return _merge_relation_requests(requests)


def _is_canonical_supersedes_request(
    request: RelationRequest,
    source: FactDecision,
    target: FactDecision,
    canonical_base_decisions: tuple[FactDecision, ...],
) -> bool:
    """Check one provisional replacement against its producing rule."""

    expected_reasons = {
        ResolutionMethod.EXPLICIT_SUPERSEDES: (
            "explicit candidate supersedes relation"
        ),
        ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH: (
            _EXPLICIT_SOURCE_OF_TRUTH_REASON
        ),
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL: (
            _CURRENT_EVIDENCE_OVER_HISTORICAL_REASON
        ),
    }
    if (
        request.reason != expected_reasons.get(request.method)
        or target.resolution_method is not request.method
        or target.reason != request.reason
    ):
        return False

    if request.method is ResolutionMethod.EXPLICIT_SUPERSEDES:
        base_target = next(
            decision
            for decision in canonical_base_decisions
            if decision.fact_id == target.fact_id
        )
        if not _is_plausible_competing_fact(base_target):
            return False
        expected = next(
            (
                candidate
                for candidate in _explicit_requests_from_canonical_decisions(
                    canonical_base_decisions
                )
                if candidate.from_fact_id == request.from_fact_id
                and candidate.to_fact_id == request.to_fact_id
            ),
            None,
        )
        return request == expected

    base_by_fact_id = {
        decision.fact_id: decision
        for decision in canonical_base_decisions
    }
    base_source = base_by_fact_id[source.fact_id]
    base_target = base_by_fact_id[target.fact_id]
    policy = base_source._candidate_context.policy
    if request.method is ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH:
        if (
            not policy.allow_explicit_source_of_truth_override
            or not _has_atomic_source_of_truth_witness(
                base_source,
                policy,
            )
            or not _is_plausible_competing_fact(base_target)
        ):
            return False
    elif request.method is ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL:
        if (
            not policy.allow_current_evidence_over_historical
            or not _has_atomic_current_evidence_witness(base_source)
            or not _is_plausible_competing_fact(base_target)
            or not _is_strict_historical_group(base_target)
        ):
            return False

    expected = RelationRequest.supersedes(
        source.fact_id,
        target.fact_id,
        expected_reasons[request.method],
        tuple((*source.candidate_ids, *target.candidate_ids)),
        tuple((*source.evidence_refs, *target.evidence_refs)),
        method=request.method,
    )
    return request == expected


def _outgoing_replacement_primary_methods_hold(
    decisions: tuple[FactDecision, ...],
    requests: tuple[RelationRequest, ...],
) -> bool:
    """Require each source to record its strongest applicable rule."""

    decisions_by_fact_id = {
        decision.fact_id: decision for decision in decisions
    }
    outgoing_by_fact_id: dict[str, list[RelationRequest]] = {}
    for request in requests:
        if request.relation_type is not RelationType.SUPERSEDES:
            continue
        outgoing_by_fact_id.setdefault(request.from_fact_id, []).append(
            request
        )
    for fact_id, outgoing in outgoing_by_fact_id.items():
        primary = min(outgoing, key=_relation_request_priority)
        source = decisions_by_fact_id[fact_id]
        if (
            source.resolution_method is not primary.method
            or source.reason != primary.reason
        ):
            return False
    return True


def _coordinator_preconditions_hold(
    resolved_decisions: tuple[FactDecision, ...],
    requests: tuple[RelationRequest, ...],
    canonical_base_decisions: tuple[FactDecision, ...],
) -> bool:
    """Validate predicate-wide prerequisites for coordinator winner rules."""

    if not canonical_base_decisions:
        return True

    policies = tuple(
        decision._candidate_context.policy
        for decision in canonical_base_decisions
    )
    policy = policies[0]
    if any(candidate_policy != policy for candidate_policy in policies[1:]):
        return False

    active_known = tuple(
        decision
        for decision in resolved_decisions
        if decision.status is ReconciliationStatus.ACTIVE
        and decision.selected_value.status is EvidenceStatus.KNOWN
    )
    if len(active_known) > 1:
        return False

    source_of_truth_requests = tuple(
        request
        for request in requests
        if request.method is ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH
    )
    current_history_requests = tuple(
        request
        for request in requests
        if request.method
        is ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL
    )
    if not source_of_truth_requests and not current_history_requests:
        return True
    if source_of_truth_requests and current_history_requests:
        return False

    raw_explicit_requests = _explicit_requests_from_canonical_decisions(
        canonical_base_decisions
    )
    explicit_directions = {
        (request.from_fact_id, request.to_fact_id)
        for request in raw_explicit_requests
    }
    mutual_explicit_fact_ids = {
        fact_id
        for from_fact_id, to_fact_id in explicit_directions
        if (to_fact_id, from_fact_id) in explicit_directions
        for fact_id in (from_fact_id, to_fact_id)
    }
    base_by_fact_id = {
        decision.fact_id: decision
        for decision in canonical_base_decisions
    }
    explicit_requests = tuple(
        request
        for request in _drop_mutual_supersedes(raw_explicit_requests)
        if _is_plausible_competing_fact(
            base_by_fact_id[request.to_fact_id]
        )
    )
    explicit_source_ids = {
        request.from_fact_id for request in explicit_requests
    }
    explicit_target_ids = {
        request.to_fact_id for request in explicit_requests
    }
    explicit_governed_ids = (
        explicit_source_ids
        | explicit_target_ids
        | mutual_explicit_fact_ids
    )

    if source_of_truth_requests:
        source_of_truth_groups = tuple(
            decision
            for decision in canonical_base_decisions
            if _has_atomic_source_of_truth_witness(decision, policy)
        )
        if len(source_of_truth_groups) != 1:
            return False
        winner = source_of_truth_groups[0]
        if winner.fact_id in (
            explicit_target_ids | mutual_explicit_fact_ids
        ):
            return False
        competitors = tuple(
            decision
            for decision in canonical_base_decisions
            if decision.fact_id != winner.fact_id
            and decision.fact_id not in explicit_governed_ids
            and _is_plausible_competing_fact(decision)
        )
        expected_target_ids = {
            decision.fact_id for decision in competitors
        }
        return (
            bool(expected_target_ids)
            and len(source_of_truth_requests) == len(expected_target_ids)
            and {
                request.from_fact_id
                for request in source_of_truth_requests
            }
            == {winner.fact_id}
            and {
                request.to_fact_id
                for request in source_of_truth_requests
            }
            == expected_target_ids
        )

    remaining_decisions = tuple(
        decision
        for decision in canonical_base_decisions
        if decision.fact_id not in explicit_governed_ids
    )
    current_groups = tuple(
        decision
        for decision in remaining_decisions
        if _has_atomic_current_evidence_witness(decision)
    )
    if (
        not policy.allow_current_evidence_over_historical
        or len(current_groups) != 1
    ):
        return False
    winner = current_groups[0]
    competitors = tuple(
        decision
        for decision in remaining_decisions
        if decision.fact_id != winner.fact_id
    )
    expected_target_ids = {decision.fact_id for decision in competitors}
    return (
        bool(expected_target_ids)
        and all(
            _is_plausible_competing_fact(competitor)
            and _is_strict_historical_group(competitor)
            for competitor in competitors
        )
        and len(current_history_requests) == len(expected_target_ids)
        and {
            request.from_fact_id for request in current_history_requests
        }
        == {winner.fact_id}
        and {
            request.to_fact_id for request in current_history_requests
        }
        == expected_target_ids
    )


def _resolve_predicate_payload(
    canonical_decisions: tuple[FactDecision, ...],
    canonical_policy: ReconciliationPolicy,
) -> tuple[tuple[FactDecision, ...], tuple[RelationRequest, ...]]:
    """Return the canonical coordinator payload without constructing it."""

    declared_explicit_requests = resolve_cross_value_replacements(
        canonical_decisions,
        canonical_policy,
    )
    predicate_identities = {
        (decision.subject, decision.predicate)
        for decision in canonical_decisions
    }
    if len(predicate_identities) > 1:
        raise ReconciliationInputError(
            "predicate decisions must share one exact subject and predicate"
        )
    decisions_by_fact_id = {
        decision.fact_id: decision for decision in canonical_decisions
    }
    explicit_requests = tuple(
        request
        for request in declared_explicit_requests
        if _is_plausible_competing_fact(
            decisions_by_fact_id[request.to_fact_id]
        )
    )
    raw_explicit_requests = _explicit_requests_from_canonical_decisions(
        canonical_decisions
    )
    explicit_directions = {
        (request.from_fact_id, request.to_fact_id)
        for request in raw_explicit_requests
    }
    mutual_explicit_fact_ids = {
        fact_id
        for from_fact_id, to_fact_id in explicit_directions
        if (to_fact_id, from_fact_id) in explicit_directions
        for fact_id in (from_fact_id, to_fact_id)
    }
    contradictory_fact_ids = {
        fact_id
        for fact_id in mutual_explicit_fact_ids
        if decisions_by_fact_id[fact_id].status
        is ReconciliationStatus.ACTIVE
    }
    explicit_source_ids = {
        request.from_fact_id for request in explicit_requests
    }
    explicit_target_ids = {
        request.to_fact_id for request in explicit_requests
    }
    explicit_governed_ids = (
        explicit_source_ids
        | explicit_target_ids
        | mutual_explicit_fact_ids
    )
    resolved_by_fact_id = {
        decision.fact_id: decision for decision in canonical_decisions
    }
    for decision in canonical_decisions:
        if decision.fact_id not in (
            explicit_source_ids | explicit_target_ids
        ):
            continue
        if (
            decision.fact_id in explicit_target_ids
            and not _is_plausible_competing_fact(decision)
        ):
            continue
        resolved_by_fact_id[decision.fact_id] = (
            _with_predicate_resolution_status(
                decision,
                (
                    ReconciliationStatus.SUPERSEDED
                    if decision.fact_id in explicit_target_ids
                    else ReconciliationStatus.ACTIVE
                ),
                ResolutionMethod.EXPLICIT_SUPERSEDES,
                "explicit candidate supersedes relation",
            )
        )
    source_of_truth_groups = tuple(
        decision
        for decision in canonical_decisions
        if _has_atomic_source_of_truth_witness(decision, canonical_policy)
    )

    if len(source_of_truth_groups) > 1:
        return _finalize_predicate_payload(
            decisions_by_fact_id,
            resolved_by_fact_id,
            explicit_requests,
            contradictory_fact_ids=contradictory_fact_ids,
        )

    if len(source_of_truth_groups) == 1:
        winner = source_of_truth_groups[0]
        if winner.fact_id in (
            explicit_target_ids | mutual_explicit_fact_ids
        ):
            return _finalize_predicate_payload(
                decisions_by_fact_id,
                resolved_by_fact_id,
                explicit_requests,
                contradictory_fact_ids=contradictory_fact_ids,
            )
        competitors = tuple(
            decision
            for decision in canonical_decisions
            if decision.fact_id != winner.fact_id
            and decision.subject == winner.subject
            and decision.predicate == winner.predicate
            and decision.fact_id not in explicit_governed_ids
            and _is_plausible_competing_fact(decision)
        )
        if not competitors:
            return _finalize_predicate_payload(
                decisions_by_fact_id,
                resolved_by_fact_id,
                explicit_requests,
                contradictory_fact_ids=contradictory_fact_ids,
            )
        source_of_truth_requests = tuple(
            RelationRequest.supersedes(
                winner.fact_id,
                competitor.fact_id,
                _EXPLICIT_SOURCE_OF_TRUTH_REASON,
                tuple((*winner.candidate_ids, *competitor.candidate_ids)),
                tuple((*winner.evidence_refs, *competitor.evidence_refs)),
                method=ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
            )
            for competitor in competitors
        )
        if winner.fact_id not in explicit_source_ids:
            resolved_by_fact_id[winner.fact_id] = (
                _with_predicate_resolution_status(
                    winner,
                    ReconciliationStatus.ACTIVE,
                    ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
                    _EXPLICIT_SOURCE_OF_TRUTH_REASON,
                )
            )
        for competitor in competitors:
            resolved_by_fact_id[competitor.fact_id] = (
                _with_predicate_resolution_status(
                    competitor,
                    ReconciliationStatus.SUPERSEDED,
                    ResolutionMethod.EXPLICIT_SOURCE_OF_TRUTH,
                    _EXPLICIT_SOURCE_OF_TRUTH_REASON,
                )
            )
        requests = _drop_mutual_supersedes(
            _merge_relation_requests(
                (*explicit_requests, *source_of_truth_requests)
            )
        )
        return _finalize_predicate_payload(
            decisions_by_fact_id,
            resolved_by_fact_id,
            requests,
            contradictory_fact_ids=contradictory_fact_ids,
        )

    remaining_decisions = tuple(
        decision
        for decision in canonical_decisions
        if decision.fact_id not in explicit_governed_ids
    )
    current_groups = tuple(
        decision
        for decision in remaining_decisions
        if _has_atomic_current_evidence_witness(decision)
    )
    if (
        not canonical_policy.allow_current_evidence_over_historical
        or len(current_groups) != 1
    ):
        return _finalize_predicate_payload(
            decisions_by_fact_id,
            resolved_by_fact_id,
            explicit_requests,
            contradictory_fact_ids=contradictory_fact_ids,
        )

    winner = current_groups[0]
    competing_groups = tuple(
        decision
        for decision in remaining_decisions
        if decision.fact_id != winner.fact_id
    )
    if not competing_groups or not all(
        _is_plausible_competing_fact(competitor)
        and _is_strict_historical_group(competitor)
        for competitor in competing_groups
    ):
        return _finalize_predicate_payload(
            decisions_by_fact_id,
            resolved_by_fact_id,
            explicit_requests,
            contradictory_fact_ids=contradictory_fact_ids,
        )

    current_requests = tuple(
        RelationRequest.supersedes(
            winner.fact_id,
            competitor.fact_id,
            _CURRENT_EVIDENCE_OVER_HISTORICAL_REASON,
            tuple((*winner.candidate_ids, *competitor.candidate_ids)),
            tuple((*winner.evidence_refs, *competitor.evidence_refs)),
            method=ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
        )
        for competitor in competing_groups
    )
    resolved_by_fact_id[winner.fact_id] = _with_predicate_resolution_status(
        winner,
        ReconciliationStatus.ACTIVE,
        ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
        _CURRENT_EVIDENCE_OVER_HISTORICAL_REASON,
    )
    for competitor in competing_groups:
        resolved_by_fact_id[competitor.fact_id] = (
            _with_predicate_resolution_status(
                competitor,
                ReconciliationStatus.SUPERSEDED,
                ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
                _CURRENT_EVIDENCE_OVER_HISTORICAL_REASON,
            )
        )
    return _finalize_predicate_payload(
        decisions_by_fact_id,
        resolved_by_fact_id,
        _drop_mutual_supersedes(
            _merge_relation_requests((*explicit_requests, *current_requests))
        ),
        contradictory_fact_ids=contradictory_fact_ids,
    )


def _make_predicate_coordinator_builders(
    issue_stage: Callable[..., object],
) -> tuple[Callable[..., PredicateOutcome], Callable[..., PredicateOutcome]]:
    def construct_canonical_outcome(
        decisions: tuple[FactDecision, ...],
        relation_requests: tuple[RelationRequest, ...],
    ) -> PredicateOutcome:
        outcome = PredicateOutcome._from_canonical_resolution(
            _PREDICATE_OUTCOME_CONSTRUCTION_TOKEN,
            decisions,
            relation_requests,
        )
        for decision in outcome.decisions:
            try:
                object.__getattribute__(decision, "_stage_auth")
            except AttributeError:
                try:
                    parent = object.__getattribute__(
                        decision,
                        "_classification_parent",
                    )
                    cohort = object.__getattribute__(decision, "_cohort")
                    parent_authorization = object.__getattribute__(
                        parent,
                        "_stage_auth",
                    )
                except (AttributeError, TypeError) as error:
                    raise ReconciliationInputError(
                        "coordinated decision lacks its canonical parent"
                    ) from error
                _validate_decision_candidate_context(parent)
                issue_stage(
                    decision,
                    cohort=cohort,
                    previous_stage=_ReconciliationStage.CLASSIFIED,
                    parent_fingerprint=object.__getattribute__(
                        parent_authorization,
                        "fingerprint",
                    ),
                    payload=_decision_stage_payload(decision),
                    graph_build_token=None,
                )
                object.__delattr__(decision, "_classification_parent")
        return outcome

    def make_predicate_outcome(
        decisions: tuple[FactDecision, ...],
        relation_requests: tuple[RelationRequest, ...],
    ) -> PredicateOutcome:
        expected_decisions, expected_requests = (
            _canonical_predicate_outcome_payload(
                decisions,
                relation_requests,
            )
        )
        return construct_canonical_outcome(
            expected_decisions,
            expected_requests,
        )

    def resolve_predicate(
        decisions: Sequence[FactDecision],
        policy: ReconciliationPolicy,
    ) -> PredicateOutcome:
        """Apply authorized cross-value rules for one predicate."""

        canonical_policy = _canonicalize_reconciliation_policy(
            policy,
            "resolver policy",
        )
        decision_snapshot = _guarded_sequence_snapshot(
            decisions,
            "decisions",
            "FactDecision records",
        )
        if not decision_snapshot:
            raise ReconciliationInputError(
                "predicate resolution requires non-empty complete membership"
            )
        if not all(
            type(decision) is FactDecision for decision in decision_snapshot
        ):
            raise ReconciliationInputError(
                "decisions must contain only exact FactDecision records"
            )
        canonical_decisions = tuple(
            _validate_decision_candidate_context(decision)
            for decision in decision_snapshot
        )
        _validate_complete_predicate_membership(canonical_decisions)
        resolved_decisions, requests = _resolve_predicate_payload(
            canonical_decisions,
            canonical_policy,
        )
        return construct_canonical_outcome(resolved_decisions, requests)

    return make_predicate_outcome, resolve_predicate


_make_predicate_outcome, resolve_predicate = (
    _make_predicate_coordinator_builders(_coordinated_stage_builder)
)
del _make_predicate_coordinator_builders
del _coordinated_stage_builder


__all__ = [
    "FrozenJsonValue",
    "JsonValue",
    "canonical_json_bytes",
    "canonical_typed_value",
    "classify_group",
    "find_activation_witnesses",
    "group_candidates",
    "is_activation_witness",
    "make_fact_id",
    "make_relation_id",
    "PredicateOutcome",
    "RelationRequest",
    "resolve_cross_value_replacements",
    "resolve_non_known_groups",
    "resolve_predicate",
    "resolve_same_value_lineage",
]
