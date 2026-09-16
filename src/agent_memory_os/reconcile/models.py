"""Fundamental immutable types for Phase 2 reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import json
import math
import secrets
import sys
from types import MappingProxyType
from typing import Callable, TypeAlias
import weakref

from agent_memory_os.evidence.models import EvidenceStatus, EvidenceValue


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


class ReconciliationStatus(str, Enum):
    """The five externally visible reconciliation states."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONFLICTED = "CONFLICTED"
    PENDING = "PENDING"
    DEPRECATED = "DEPRECATED"


class SourceType(str, Enum):
    """The attributable source category of a memory candidate."""

    USER_EXPLICIT = "USER_EXPLICIT"
    CURRENT_EVIDENCE = "CURRENT_EVIDENCE"
    PROJECT_DOC = "PROJECT_DOC"
    PROJECT_CARD = "PROJECT_CARD"
    TEMPORAL_RECORD = "TEMPORAL_RECORD"
    SESSION_LOG = "SESSION_LOG"
    HISTORICAL_MEMORY = "HISTORICAL_MEMORY"


class CandidateStatusHint(str, Enum):
    """The source-provided semantics of a candidate claim."""

    CURRENT_FACT = "CURRENT_FACT"
    HISTORICAL = "HISTORICAL"
    PLAN = "PLAN"
    HYPOTHESIS = "HYPOTHESIS"
    SOURCE_OF_TRUTH = "SOURCE_OF_TRUTH"
    DEPRECATED = "DEPRECATED"


class RelationType(str, Enum):
    """Canonical relationship types between reconciled facts."""

    SUPERSEDES = "SUPERSEDES"
    CONFLICTS = "CONFLICTS"


_RECONCILIATION_COHORT_CONSTRUCTION_TOKEN = object()


def _cohort_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"cohort:v1:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, init=False)
class _ReconciliationCohort:
    """Authenticated invocation metadata carried between internal stages."""

    project_id: str
    snapshot_id: str
    policy_fingerprint: str | None
    reconciled_at: str | None
    candidate_id_namespace: str
    predicate_membership: tuple[
        tuple[
            str,
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            str,
        ],
        ...,
    ]
    invocation_token: str
    fingerprint: str

    def __new__(cls, *args: object, **kwargs: object) -> _ReconciliationCohort:
        raise TypeError("reconciliation cohorts are internal")

    @classmethod
    def _from_payload(
        cls,
        construction_token: object,
        *,
        project_id: str,
        snapshot_id: str,
        policy_fingerprint: str | None,
        reconciled_at: str | None,
        candidate_id_namespace: str,
        predicate_membership: tuple[
            tuple[
                str,
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
                str,
            ],
            ...,
        ],
        invocation_token: str,
    ) -> _ReconciliationCohort:
        if (
            cls is not _ReconciliationCohort
            or construction_token
            is not _RECONCILIATION_COHORT_CONSTRUCTION_TOKEN
        ):
            raise TypeError("reconciliation cohorts are internal")
        payload = {
            "namespace": "reconciliation-cohort:v1",
            "project_id": project_id,
            "snapshot_id": snapshot_id,
            "policy_fingerprint": policy_fingerprint,
            "reconciled_at": reconciled_at,
            "candidate_id_namespace": candidate_id_namespace,
            "predicate_membership": predicate_membership,
            "invocation_token": invocation_token,
        }
        instance = object.__new__(cls)
        for field_name, value in payload.items():
            if field_name != "namespace":
                object.__setattr__(instance, field_name, value)
        object.__setattr__(instance, "fingerprint", _cohort_digest(payload))
        return instance


def _validate_reconciliation_cohort(
    cohort: object,
    *,
    require_bound: bool,
) -> _ReconciliationCohort:
    if type(cohort) is not _ReconciliationCohort:
        raise ReconciliationInputError("reconciliation cohort is invalid")
    try:
        values = {
            field_name: object.__getattribute__(cohort, field_name)
            for field_name in (
                "project_id",
                "snapshot_id",
                "policy_fingerprint",
                "reconciled_at",
                "candidate_id_namespace",
                "predicate_membership",
                "invocation_token",
                "fingerprint",
            )
        }
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            "reconciliation cohort is incomplete"
        ) from error
    for field_name in (
        "project_id",
        "snapshot_id",
        "candidate_id_namespace",
        "invocation_token",
        "fingerprint",
    ):
        _validate_required_text(values[field_name], f"cohort {field_name}")
    for field_name in ("policy_fingerprint", "reconciled_at"):
        value = values[field_name]
        if value is not None:
            _validate_required_text(value, f"cohort {field_name}")
    membership = values["predicate_membership"]
    if type(membership) is not tuple:
        raise ReconciliationInputError(
            "cohort predicate membership must be an exact tuple"
        )
    if not membership and (
        values["policy_fingerprint"] is None
        or values["reconciled_at"] is None
    ):
        raise ReconciliationInputError(
            "an empty reconciliation cohort must bind policy and clock"
        )
    previous_key: tuple[str, str, str] | None = None
    for entry in membership:
        if type(entry) is not tuple or len(entry) != 6:
            raise ReconciliationInputError(
                "cohort predicate membership entry is invalid"
            )
        project_id, subject, predicate, fact_ids, candidate_ids, namespace = entry
        for field_name, value in (
            ("project_id", project_id),
            ("subject", subject),
            ("predicate", predicate),
            ("candidate namespace", namespace),
        ):
            _validate_required_text(value, f"cohort membership {field_name}")
        if project_id != values["project_id"]:
            raise ReconciliationInputError(
                "cohort predicate membership project is invalid"
            )
        for field_name, field_value in (
            ("fact IDs", fact_ids),
            ("candidate IDs", candidate_ids),
        ):
            if type(field_value) is not tuple or not field_value:
                raise ReconciliationInputError(
                    f"cohort predicate membership {field_name} is invalid"
                )
            for item in field_value:
                _validate_required_text(
                    item,
                    f"cohort predicate membership {field_name}",
                )
            if field_value != tuple(sorted(set(field_value))):
                raise ReconciliationInputError(
                    f"cohort predicate membership {field_name} is invalid"
                )
        key = (project_id, subject, predicate)
        if previous_key is not None and key <= previous_key:
            raise ReconciliationInputError(
                "cohort predicate membership must be sorted and unique"
            )
        previous_key = key
    if require_bound and (
        values["policy_fingerprint"] is None
        or values["reconciled_at"] is None
    ):
        raise ReconciliationInputError(
            "reconciliation cohort must bind policy and clock"
        )
    if values["reconciled_at"] is not None:
        try:
            reconciled_at = datetime.fromisoformat(values["reconciled_at"])
            if (
                reconciled_at.tzinfo is None
                or reconciled_at.utcoffset() != timedelta(0)
                or reconciled_at.isoformat() != values["reconciled_at"]
            ):
                raise ValueError("cohort clock is not canonical UTC")
        except (TypeError, ValueError, OverflowError) as error:
            raise ReconciliationInputError(
                "reconciliation cohort clock must be canonical UTC"
            ) from error
    payload = {
        "namespace": "reconciliation-cohort:v1",
        "project_id": values["project_id"],
        "snapshot_id": values["snapshot_id"],
        "policy_fingerprint": values["policy_fingerprint"],
        "reconciled_at": values["reconciled_at"],
        "candidate_id_namespace": values["candidate_id_namespace"],
        "predicate_membership": values["predicate_membership"],
        "invocation_token": values["invocation_token"],
    }
    if values["fingerprint"] != _cohort_digest(payload):
        raise ReconciliationInputError(
            "reconciliation cohort fingerprint is invalid"
        )
    return cohort


def _strict_frozen_json_node(value: object) -> dict[str, object]:
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ReconciliationInputError("KNOWN float must be finite")
        return {"type": "float", "value": value.hex()}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is tuple:
        return {
            "type": "list",
            "value": [_strict_frozen_json_node(item) for item in value],
        }
    if type(value) is MappingProxyType:
        try:
            items = tuple(value.items())
        except Exception as error:
            raise ReconciliationInputError(
                "KNOWN mapping could not be inspected safely"
            ) from error
        normalized: list[tuple[str, dict[str, object]]] = []
        for item in items:
            if type(item) is not tuple or len(item) != 2:
                raise ReconciliationInputError(
                    "KNOWN mapping items must be exact pairs"
                )
            key, item_value = item
            _validate_required_text(key, "KNOWN mapping key")
            normalized.append((key, _strict_frozen_json_node(item_value)))
        keys = tuple(key for key, _ in normalized)
        if len(set(keys)) != len(keys):
            raise ReconciliationInputError(
                "KNOWN mapping keys must be unique"
            )
        return {
            "type": "mapping",
            "value": {
                key: item_value for key, item_value in sorted(normalized)
            },
        }
    raise ReconciliationInputError(
        "KNOWN value must use exact frozen JSON types"
    )


def _strict_known_evidence_node(value: object, known_kind: str) -> object:
    if known_kind == "json":
        return _strict_frozen_json_node(value)
    if known_kind == "metadata":
        if type(value) is not MappingProxyType:
            raise ReconciliationInputError(
                "KNOWN metadata must be an exact frozen mapping"
            )
        return _strict_frozen_json_node(value)
    if known_kind == "float":
        if type(value) is not float or not math.isfinite(value):
            raise ReconciliationInputError("KNOWN confidence must be a finite float")
        return {"type": "float", "value": value.hex()}
    if known_kind == "string":
        _validate_required_text(value, "KNOWN string")
        return {"type": "string", "value": value}
    if known_kind == "bool":
        if type(value) is not bool:
            raise ReconciliationInputError("KNOWN boolean must be exact bool")
        return {"type": "boolean", "value": value}
    if known_kind == "candidate_status":
        if type(value) is not CandidateStatusHint:
            raise ReconciliationInputError(
                "KNOWN status hint must be an exact CandidateStatusHint"
            )
        return {"type": "candidate-status", "value": value.value}
    if known_kind == "string_tuple":
        if type(value) is not tuple:
            raise ReconciliationInputError(
                "KNOWN string sequence must be an exact tuple"
            )
        for item in value:
            _validate_required_text(item, "KNOWN string sequence")
        return {"type": "string-list", "value": list(value)}
    raise ReconciliationInputError("unsupported evidence signature kind")


def _strict_evidence_signature(
    value: object,
    field_name: str,
    known_kind: str,
) -> bytes:
    if type(value) is not EvidenceValue:
        raise ReconciliationInputError(
            f"{field_name} must be an exact EvidenceValue"
        )
    try:
        status = object.__getattribute__(value, "status")
        known_value = object.__getattribute__(value, "value")
        reason = object.__getattribute__(value, "reason")
        source = object.__getattribute__(value, "source")
    except (AttributeError, TypeError) as error:
        raise ReconciliationInputError(
            f"{field_name} must be a complete EvidenceValue"
        ) from error
    if type(status) is not EvidenceStatus:
        raise ReconciliationInputError(
            f"{field_name}.status must be an exact EvidenceStatus"
        )
    _validate_required_text(source, f"{field_name}.source")
    if status is EvidenceStatus.KNOWN:
        if known_value is None or reason is not None:
            raise ReconciliationInputError(
                f"{field_name} violates KNOWN EvidenceValue invariants"
            )
        payload_value = _strict_known_evidence_node(known_value, known_kind)
        payload_reason = None
    else:
        if known_value is not None:
            raise ReconciliationInputError(
                f"{field_name} non-known evidence forbids value"
            )
        _validate_required_text(reason, f"{field_name}.reason")
        payload_value = None
        payload_reason = reason
    payload = {
        "status": status.value,
        "value": payload_value,
        "reason": payload_reason,
        "source": source,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _ReconciliationStage(str, Enum):
    GROUPED = "GROUPED"
    CLASSIFIED = "CLASSIFIED"
    COORDINATED = "COORDINATED"
    BASE_FACT = "BASE_FACT"
    MATERIALIZED_FACT = "MATERIALIZED_FACT"
    MATERIALIZED_RELATION = "MATERIALIZED_RELATION"
    MATERIALIZED_GRAPH = "MATERIALIZED_GRAPH"


@dataclass(frozen=True, init=False)
class _StageAuthorization:
    stage: _ReconciliationStage
    previous_stage: _ReconciliationStage | None
    cohort_fingerprint: str
    parent_fingerprint: str | None
    payload_fingerprint: str
    graph_build_token: str | None
    member_token: str
    fingerprint: str

    def __new__(cls, *args: object, **kwargs: object) -> _StageAuthorization:
        raise TypeError("reconciliation stage authorization is internal")


def _build_stage_authority() -> tuple[Callable[..., _StageAuthorization], ...]:
    """Create a process-local keyed authority with weak identity registration."""

    signing_key = secrets.token_bytes(32)
    members: dict[str, weakref.ReferenceType[object]] = {}
    member_tokens: dict[int, tuple[weakref.ReferenceType[object], str]] = {}

    def sign(payload: dict[str, object]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "stage-auth:v2:" + hmac.new(
            signing_key,
            encoded,
            hashlib.sha256,
        ).hexdigest()

    def discard(
        member_id: int,
        member_token: str,
        expired: weakref.ReferenceType[object],
    ) -> None:
        if members.get(member_token) is expired:
            members.pop(member_token, None)
        retained = member_tokens.get(member_id)
        if retained is not None and retained[0] is expired:
            member_tokens.pop(member_id, None)

    def issue(
        member: object,
        *,
        cohort: _ReconciliationCohort,
        stage: _ReconciliationStage,
        previous_stage: _ReconciliationStage | None,
        parent_fingerprint: str | None,
        payload: bytes,
        graph_build_token: str | None,
    ) -> _StageAuthorization:
        checked_cohort = _validate_reconciliation_cohort(
            cohort,
            require_bound=stage is not _ReconciliationStage.GROUPED,
        )
        if type(payload) is not bytes:
            raise ReconciliationInputError("stage payload must be canonical bytes")
        if parent_fingerprint is not None:
            _validate_required_text(parent_fingerprint, "parent stage fingerprint")
        if graph_build_token is not None:
            _validate_required_text(graph_build_token, "graph build token")
        retained = member_tokens.get(id(member))
        if retained is not None and retained[0]() is member:
            raise ReconciliationInputError(
                "reconciliation member is already stage authenticated"
            )
        payload_fingerprint = hashlib.sha256(payload).hexdigest()
        member_token = secrets.token_hex(32)
        signature_payload = {
            "namespace": "stage-authorization:v2",
            "stage": stage.value,
            "previous_stage": (
                None if previous_stage is None else previous_stage.value
            ),
            "cohort_fingerprint": checked_cohort.fingerprint,
            "parent_fingerprint": parent_fingerprint,
            "payload_fingerprint": payload_fingerprint,
            "graph_build_token": graph_build_token,
            "member_token": member_token,
        }
        instance = object.__new__(_StageAuthorization)
        for field_name, value in signature_payload.items():
            if field_name == "namespace":
                continue
            if field_name == "stage":
                value = stage
            elif field_name == "previous_stage":
                value = previous_stage
            object.__setattr__(instance, field_name, value)
        object.__setattr__(instance, "fingerprint", sign(signature_payload))
        member_id = id(member)
        reference = weakref.ref(
            member,
            lambda expired: discard(member_id, member_token, expired),
        )
        members[member_token] = reference
        member_tokens[member_id] = (reference, member_token)
        object.__setattr__(member, "_stage_auth", instance)
        return instance

    def validate(
        member: object,
        *,
        cohort: object,
        expected_stage: _ReconciliationStage,
        expected_previous_stage: _ReconciliationStage | None,
        payload: bytes,
        graph_build_token: str | None,
    ) -> _StageAuthorization:
        checked_cohort = _validate_reconciliation_cohort(
            cohort,
            require_bound=expected_stage is not _ReconciliationStage.GROUPED,
        )
        try:
            authorization = object.__getattribute__(member, "_stage_auth")
            values = {
                name: object.__getattribute__(authorization, name)
                for name in (
                    "stage",
                    "previous_stage",
                    "cohort_fingerprint",
                    "parent_fingerprint",
                    "payload_fingerprint",
                    "graph_build_token",
                    "member_token",
                    "fingerprint",
                )
            }
        except (AttributeError, TypeError) as error:
            raise ReconciliationInputError(
                "reconciliation member is not stage authenticated"
            ) from error
        if type(authorization) is not _StageAuthorization:
            raise ReconciliationInputError("stage authorization is invalid")
        if (
            values["stage"] is not expected_stage
            or values["previous_stage"] is not expected_previous_stage
        ):
            raise ReconciliationInputError(
                "reconciliation member is at the wrong stage"
            )
        signature_payload = {
            "namespace": "stage-authorization:v2",
            "stage": expected_stage.value,
            "previous_stage": (
                None
                if expected_previous_stage is None
                else expected_previous_stage.value
            ),
            "cohort_fingerprint": values["cohort_fingerprint"],
            "parent_fingerprint": values["parent_fingerprint"],
            "payload_fingerprint": values["payload_fingerprint"],
            "graph_build_token": values["graph_build_token"],
            "member_token": values["member_token"],
        }
        member_token = values["member_token"]
        if (
            type(values["cohort_fingerprint"]) is not str
            or values["cohort_fingerprint"] != checked_cohort.fingerprint
            or type(values["payload_fingerprint"]) is not str
            or values["payload_fingerprint"] != hashlib.sha256(payload).hexdigest()
            or values["graph_build_token"] != graph_build_token
            or type(member_token) is not str
            or members.get(member_token) is None
            or members[member_token]() is not member
            or type(values["fingerprint"]) is not str
            or not hmac.compare_digest(values["fingerprint"], sign(signature_payload))
        ):
            raise ReconciliationInputError(
                "reconciliation stage authorization payload is invalid"
            )
        return authorization

    def fixed_authorizer(
        stage: _ReconciliationStage,
        previous_stages: tuple[_ReconciliationStage | None, ...],
    ) -> Callable[..., _StageAuthorization]:
        def authorize(
            member: object,
            *,
            cohort: _ReconciliationCohort,
            previous_stage: _ReconciliationStage | None,
            parent_fingerprint: str | None,
            payload: bytes,
            graph_build_token: str | None,
        ) -> _StageAuthorization:
            if previous_stage not in previous_stages:
                raise ReconciliationInputError(
                    "reconciliation stage transition is invalid"
                )
            return issue(
                member,
                cohort=cohort,
                stage=stage,
                previous_stage=previous_stage,
                parent_fingerprint=parent_fingerprint,
                payload=payload,
                graph_build_token=graph_build_token,
            )

        return authorize

    return (
        fixed_authorizer(_ReconciliationStage.GROUPED, (None,)),
        fixed_authorizer(
            _ReconciliationStage.CLASSIFIED,
            (_ReconciliationStage.GROUPED,),
        ),
        fixed_authorizer(
            _ReconciliationStage.COORDINATED,
            (_ReconciliationStage.CLASSIFIED,),
        ),
        fixed_authorizer(
            _ReconciliationStage.BASE_FACT,
            (_ReconciliationStage.CLASSIFIED, _ReconciliationStage.COORDINATED),
        ),
        fixed_authorizer(
            _ReconciliationStage.MATERIALIZED_FACT,
            (_ReconciliationStage.BASE_FACT,),
        ),
        fixed_authorizer(
            _ReconciliationStage.MATERIALIZED_RELATION,
            (_ReconciliationStage.BASE_FACT,),
        ),
        fixed_authorizer(
            _ReconciliationStage.MATERIALIZED_GRAPH,
            (
                _ReconciliationStage.MATERIALIZED_FACT,
                _ReconciliationStage.MATERIALIZED_RELATION,
            ),
        ),
        validate,
    )


def _make_stage_authority_claim(
    authorities: tuple[Callable[..., _StageAuthorization], ...],
) -> tuple[
    Callable[[str], tuple[Callable[..., _StageAuthorization], ...]],
    Callable[..., _StageAuthorization],
]:
    issued = authorities[:-1]
    verifier = authorities[-1]
    claimed: set[str] = set()
    expected_callers = {
        "agent_memory_os.reconcile.rules": issued[:3],
        "agent_memory_os.reconcile.reconciler": issued[3:],
    }

    def claim(
        client_module: str,
    ) -> tuple[Callable[..., _StageAuthorization], ...]:
        try:
            caller_frame = sys._getframe(1)
            caller_module = sys.modules.get(client_module)
            caller_spec = object.__getattribute__(caller_module, "__spec__")
            caller_globals = caller_frame.f_globals
            caller_is_initializing = object.__getattribute__(
                caller_spec,
                "_initializing",
            )
            caller_origin = object.__getattribute__(caller_spec, "origin")
        except (AttributeError, TypeError, ValueError):
            caller_module = None
            caller_globals = None
            caller_is_initializing = False
            caller_origin = None
        if (
            type(client_module) is not str
            or client_module not in expected_callers
            or client_module in claimed
            or caller_module is None
            or caller_globals is not vars(caller_module)
            or caller_globals.get("__name__") != client_module
            or caller_globals.get("__spec__") is not caller_spec
            or caller_is_initializing is not True
            or type(caller_origin) is not str
            or caller_frame.f_code.co_filename != caller_origin
        ):
            raise TypeError("stage builder authority is internal")
        claimed.add(client_module)
        selected = expected_callers[client_module]
        if claimed == set(expected_callers):
            globals().pop("_claim_canonical_stage_builders", None)
        return selected

    return claim, verifier


_claim_canonical_stage_builders, _validate_stage_authorization = (
    _make_stage_authority_claim(_build_stage_authority())
)
del _build_stage_authority
del _make_stage_authority_claim


_RELATION_RECORD_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class RelationRecord:
    """One immutable canonical relationship between reconciled facts."""

    relation_id: str
    relation_type: RelationType
    from_fact_id: str
    to_fact_id: str
    relation_reason: str
    candidate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __new__(cls, *args: object, **kwargs: object) -> RelationRecord:
        raise TypeError("RelationRecord instances must be created by create")

    @classmethod
    def create(
        cls,
        relation_type: RelationType,
        from_fact_id: str,
        to_fact_id: str,
        relation_reason: str,
        candidate_ids: tuple[str, ...] | list[str],
        evidence_refs: tuple[str, ...] | list[str],
    ) -> RelationRecord:
        if cls is not RelationRecord:
            raise ReconciliationInputError(
                "relation factory requires exact RelationRecord"
            )
        from agent_memory_os.reconcile.rules import make_relation_id

        _validate_required_text(relation_reason, "relation_reason")
        normalized_candidate_ids = _normalize_relation_provenance(
            candidate_ids,
            "candidate_ids",
            allow_list=True,
        )
        normalized_evidence_refs = _normalize_relation_provenance(
            evidence_refs,
            "evidence_refs",
            allow_list=True,
        )
        return cls._from_canonical(
            _RELATION_RECORD_CONSTRUCTION_TOKEN,
            relation_id=make_relation_id(
                relation_type,
                from_fact_id,
                to_fact_id,
            ),
            relation_type=relation_type,
            from_fact_id=from_fact_id,
            to_fact_id=to_fact_id,
            relation_reason=relation_reason,
            candidate_ids=normalized_candidate_ids,
            evidence_refs=normalized_evidence_refs,
        )

    @classmethod
    def _from_canonical(
        cls,
        construction_token: object,
        *,
        relation_id: str,
        relation_type: RelationType,
        from_fact_id: str,
        to_fact_id: str,
        relation_reason: str,
        candidate_ids: tuple[str, ...],
        evidence_refs: tuple[str, ...],
    ) -> RelationRecord:
        if (
            construction_token is not _RELATION_RECORD_CONSTRUCTION_TOKEN
            or cls is not RelationRecord
        ):
            raise TypeError("RelationRecord instances must be created by create")

        from agent_memory_os.reconcile.rules import make_relation_id

        expected_id = make_relation_id(
            relation_type,
            from_fact_id,
            to_fact_id,
        )
        if type(relation_id) is not str or relation_id != expected_id:
            raise ReconciliationInputError(
                "relation_id must match the canonical relation identity"
            )
        _validate_required_text(relation_reason, "relation_reason")
        _validate_sorted_unique_tuple(candidate_ids, "candidate_ids")
        _validate_sorted_unique_tuple(evidence_refs, "evidence_refs")

        instance = object.__new__(cls)
        values = {
            "relation_id": relation_id,
            "relation_type": relation_type,
            "from_fact_id": from_fact_id,
            "to_fact_id": to_fact_id,
            "relation_reason": relation_reason,
            "candidate_ids": candidate_ids,
            "evidence_refs": evidence_refs,
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance

    def to_dict(self) -> dict[str, object]:
        """Serialize the record to deterministic JSON-compatible values."""

        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "from_fact_id": self.from_fact_id,
            "to_fact_id": self.to_fact_id,
            "relation_reason": self.relation_reason,
            "candidate_ids": list(self.candidate_ids),
            "evidence_refs": list(self.evidence_refs),
        }


class ResolutionMethod(str, Enum):
    """The deterministic rule used to classify a reconciled value."""

    DIRECT_CURRENT = "DIRECT_CURRENT"
    SAME_VALUE_MERGE = "SAME_VALUE_MERGE"
    SAME_VALUE_REACTIVATION = "SAME_VALUE_REACTIVATION"
    EXPLICIT_SOURCE_OF_TRUTH = "EXPLICIT_SOURCE_OF_TRUTH"
    CURRENT_EVIDENCE_OVER_HISTORICAL = "CURRENT_EVIDENCE_OVER_HISTORICAL"
    EXPLICIT_SUPERSEDES = "EXPLICIT_SUPERSEDES"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"
    PENDING_SEMANTICS = "PENDING_SEMANTICS"
    TEMPORAL_PENDING = "TEMPORAL_PENDING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EXPLICIT_DEPRECATION = "EXPLICIT_DEPRECATION"


class WarningCode(str, Enum):
    """Stable warning codes emitted by reconciliation rules."""

    EVIDENCE_UNKNOWN = "EVIDENCE_UNKNOWN"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    INVALID_TEMPORAL_ORDER = "INVALID_TEMPORAL_ORDER"
    FUTURE_CLOCK_SKEW = "FUTURE_CLOCK_SKEW"
    FUTURE_VALIDITY = "FUTURE_VALIDITY"
    EXPIRED_WITHOUT_REPLACEMENT = "EXPIRED_WITHOUT_REPLACEMENT"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
    UNRESOLVED_TEMPORAL_COMPARISON = "UNRESOLVED_TEMPORAL_COMPARISON"
    CONTRADICTORY_SUPERSEDES = "CONTRADICTORY_SUPERSEDES"


class LineageOutcome(str, Enum):
    """Candidate-level result for one canonical same-value group."""

    CURRENT = "CURRENT"
    NEUTRAL = "NEUTRAL"
    REACTIVATED = "REACTIVATED"
    DEPRECATED = "DEPRECATED"
    UNRESOLVED = "UNRESOLVED"


class ReconciliationInputError(ValueError):
    """Raised when reconciliation input cannot be audited safely."""


class ReconciliationInvariantError(RuntimeError):
    """Raised when an internally produced result violates an invariant."""


def _default_source_precedence() -> tuple[tuple[SourceType, int], ...]:
    from agent_memory_os.reconcile.precedence import DEFAULT_SOURCE_PRECEDENCE

    return DEFAULT_SOURCE_PRECEDENCE


@dataclass(frozen=True)
class ReconciliationPolicy:
    """Immutable controls for conservative reconciliation decisions."""

    source_precedence: tuple[tuple[SourceType, int], ...] = field(
        default_factory=_default_source_precedence
    )
    active_confidence_threshold: float = 0.5
    allow_explicit_source_of_truth_override: bool = True
    allow_current_evidence_over_historical: bool = True
    require_explicit_hint_for_user_override: bool = True
    conflict_on_equal_precedence_disagreement: bool = True
    max_future_clock_skew_seconds: int = 300

    def __post_init__(self) -> None:
        try:
            entries = tuple(self.source_precedence)
        except TypeError as error:
            raise ReconciliationInputError(
                "source_precedence must be a sequence of (SourceType, rank) pairs"
            ) from error

        normalized: list[tuple[object, object]] = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ReconciliationInputError(
                    "source_precedence must contain (SourceType, rank) pairs"
                )
            normalized.append((entry[0], entry[1]))
        object.__setattr__(self, "source_precedence", tuple(normalized))

        from agent_memory_os.reconcile.precedence import validate_policy

        validate_policy(self)

    def rank_for(self, source_type: SourceType) -> int:
        """Return the configured rank for a validated source category."""

        if not isinstance(source_type, SourceType):
            raise ReconciliationInputError("source_type must be a SourceType")
        for configured_source, rank in self.source_precedence:
            if configured_source is source_type:
                return rank
        raise ReconciliationInputError(
            f"source_precedence is missing SourceType.{source_type.name}"
        )


def _validate_required_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ReconciliationInputError(
            f"{field_name} must be a non-empty, NUL-free string"
        )


def _validate_sorted_unique_tuple(
    value: object,
    field_name: str,
) -> None:
    if type(value) is not tuple or not value:
        raise ReconciliationInputError(
            f"{field_name} must be a non-empty tuple"
        )
    for item in value:
        _validate_required_text(item, field_name)
    if value != tuple(sorted(value)):
        raise ReconciliationInputError(f"{field_name} must be sorted")
    if len(set(value)) != len(value):
        raise ReconciliationInputError(f"{field_name} must not contain duplicates")


def _normalize_relation_provenance(
    value: object,
    field_name: str,
    *,
    allow_list: bool,
) -> tuple[str, ...]:
    """Copy canonical relation provenance into sorted immutable tuples."""

    accepted_types = (tuple, list) if allow_list else (tuple,)
    if type(value) not in accepted_types:
        container_description = "tuple or list" if allow_list else "tuple"
        raise ReconciliationInputError(
            f"{field_name} must be a non-empty exact {container_description}"
        )
    try:
        snapshot = tuple(value)
    except Exception as error:
        raise ReconciliationInputError(
            f"{field_name} could not be copied safely"
        ) from error
    if not snapshot:
        raise ReconciliationInputError(f"{field_name} must be non-empty")
    for item in snapshot:
        _validate_required_text(item, field_name)
    return tuple(sorted(set(snapshot)))


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Return an independent, recursively immutable JSON-compatible value."""

    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ReconciliationInputError("value must contain only finite floats")
        return value
    if type(value) is str:
        return value
    if type(value) is list:
        try:
            snapshot = tuple(value)
        except Exception as error:
            raise ReconciliationInputError(
                "value list could not be copied safely"
            ) from error
        return tuple(freeze_json_value(item) for item in snapshot)
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, freeze_json_value)
    raise ReconciliationInputError("value must be a supported JSON value")


def _freeze_json_mapping(
    value: Mapping[object, object],
    freeze_item: Callable[[object], FrozenJsonValue],
) -> Mapping[str, FrozenJsonValue]:
    """Snapshot every mapping item exactly once before validation/freezing."""

    normalized = _snapshot_json_mapping_items(value)
    return MappingProxyType(
        {
            key: freeze_item(item_value)
            for key, item_value in sorted(normalized, key=lambda item: item[0])
        }
    )


def _snapshot_json_mapping_items(
    value: Mapping[object, object],
) -> tuple[tuple[str, object], ...]:
    """Copy a mapping through one guarded traversal into trusted pairs."""

    try:
        items = tuple(value.items())
    except Exception as error:
        raise ReconciliationInputError(
            "value mapping could not be copied safely"
        ) from error
    normalized: list[tuple[str, object]] = []
    seen_keys: set[str] = set()
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise ReconciliationInputError(
                "value mapping items must be exact key/value pairs"
            )
        key, item_value = item
        if type(key) is not str:
            raise ReconciliationInputError(
                "value mapping keys must be exact strings"
            )
        if key in seen_keys:
            raise ReconciliationInputError(
                "value mapping keys must not contain duplicates"
            )
        seen_keys.add(key)
        normalized.append((key, item_value))
    return tuple(normalized)


def _validate_evidence_value(
    value: object,
    field_name: str,
) -> EvidenceValue[object]:
    if type(value) is not EvidenceValue:
        raise ReconciliationInputError(
            f"{field_name} must be an exact EvidenceValue"
        )
    if type(value.status) is not EvidenceStatus:
        raise ReconciliationInputError(
            f"{field_name} must have an exact EvidenceStatus"
        )
    _validate_required_text(value.source, f"{field_name}.source")
    if value.status is not EvidenceStatus.KNOWN:
        _validate_required_text(value.reason, f"{field_name}.reason")
    try:
        return EvidenceValue(
            status=value.status,
            value=value.value,
            reason=value.reason,
            source=value.source,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ReconciliationInputError(
            f"{field_name} violates EvidenceValue invariants: {error}"
        ) from error


def _freeze_evidence_value(
    value: object,
    field_name: str,
    freeze_known: Callable[[object], object],
) -> EvidenceValue[object]:
    validated = _validate_evidence_value(value, field_name)
    known_value = validated.value
    if validated.status is EvidenceStatus.KNOWN:
        try:
            known_value = freeze_known(validated.value)
        except ReconciliationInputError as error:
            raise ReconciliationInputError(f"{field_name}: {error}") from error
    return EvidenceValue(
        status=validated.status,
        value=known_value,
        reason=validated.reason,
        source=validated.source,
    )


def _freeze_expected_type(
    value: object,
    field_name: str,
    expected_type: type,
    expected_name: str,
) -> EvidenceValue[object]:
    def validate(known_value: object) -> object:
        if type(known_value) is not expected_type:
            raise ReconciliationInputError(f"must be a {expected_name}")
        return known_value

    return _freeze_evidence_value(value, field_name, validate)


@dataclass(frozen=True)
class MemoryCandidate:
    """One immutable, attributable claim submitted for reconciliation."""

    candidate_id: str
    subject: str
    predicate: str
    value: EvidenceValue[JsonValue]
    status_hint: EvidenceValue[CandidateStatusHint]
    source_type: SourceType
    source_ref: str
    observed_at: EvidenceValue[str]
    valid_from: EvidenceValue[str]
    valid_until: EvidenceValue[str]
    confidence: EvidenceValue[float]
    explicit_user_instruction: EvidenceValue[bool]
    supersedes: EvidenceValue[tuple[str, ...]]
    deprecated: EvidenceValue[bool]
    metadata: EvidenceValue[dict[str, JsonValue]]

    def __post_init__(self) -> None:
        for field_name in ("candidate_id", "subject", "predicate", "source_ref"):
            _validate_required_text(getattr(self, field_name), field_name)
        if not isinstance(self.source_type, SourceType):
            raise ReconciliationInputError("source_type must be a SourceType")

        object.__setattr__(
            self,
            "value",
            _freeze_evidence_value(self.value, "value", freeze_json_value),
        )
        object.__setattr__(
            self,
            "status_hint",
            _freeze_expected_type(
                self.status_hint,
                "status_hint",
                CandidateStatusHint,
                "CandidateStatusHint",
            ),
        )
        for field_name in ("observed_at", "valid_from", "valid_until"):
            object.__setattr__(
                self,
                field_name,
                _freeze_expected_type(
                    getattr(self, field_name),
                    field_name,
                    str,
                    "string",
                ),
            )

        def freeze_confidence(known_value: object) -> float:
            if type(known_value) is not float or not math.isfinite(
                known_value
            ) or not 0.0 <= known_value <= 1.0:
                raise ReconciliationInputError(
                    "must be a finite float within [0.0, 1.0]"
                )
            return known_value

        object.__setattr__(
            self,
            "confidence",
            _freeze_evidence_value(
                self.confidence,
                "confidence",
                freeze_confidence,
            ),
        )
        for field_name in ("explicit_user_instruction", "deprecated"):
            object.__setattr__(
                self,
                field_name,
                _freeze_expected_type(
                    getattr(self, field_name),
                    field_name,
                    bool,
                    "boolean",
                ),
            )

        def freeze_supersedes(known_value: object) -> tuple[str, ...]:
            if not isinstance(known_value, (list, tuple)) or not all(
                type(item) is str for item in known_value
            ):
                raise ReconciliationInputError("must be a sequence of strings")
            return tuple(known_value)

        object.__setattr__(
            self,
            "supersedes",
            _freeze_evidence_value(
                self.supersedes,
                "supersedes",
                freeze_supersedes,
            ),
        )

        def freeze_metadata(known_value: object) -> FrozenJsonValue:
            if not isinstance(known_value, Mapping):
                raise ReconciliationInputError("must be a mapping")
            return freeze_json_value(known_value)

        object.__setattr__(
            self,
            "metadata",
            _freeze_evidence_value(
                self.metadata,
                "metadata",
                freeze_metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize immutable containers and enums to JSON-compatible values."""

        return {
            "candidate_id": self.candidate_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value.to_dict(),
            "status_hint": self.status_hint.to_dict(),
            "source_type": self.source_type.value,
            "source_ref": self.source_ref,
            "observed_at": self.observed_at.to_dict(),
            "valid_from": self.valid_from.to_dict(),
            "valid_until": self.valid_until.to_dict(),
            "confidence": self.confidence.to_dict(),
            "explicit_user_instruction": self.explicit_user_instruction.to_dict(),
            "supersedes": self.supersedes.to_dict(),
            "deprecated": self.deprecated.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class ReconciliationWarning:
    """A deterministic, provenance-bearing non-fatal warning."""

    code: WarningCode
    message: str
    candidate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    requires_human_review: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, WarningCode):
            raise ReconciliationInputError("code must be a WarningCode")
        _validate_required_text(self.message, "message")
        _validate_sorted_unique_tuple(self.candidate_ids, "candidate_ids")
        _validate_sorted_unique_tuple(self.evidence_refs, "evidence_refs")
        if not isinstance(self.requires_human_review, bool):
            raise ReconciliationInputError(
                "requires_human_review must be a boolean"
            )


def _warning_sort_key(
    warning: ReconciliationWarning,
) -> tuple[object, ...]:
    return (
        warning.code.value,
        warning.candidate_ids,
        warning.message,
        warning.evidence_refs,
        warning.requires_human_review,
    )


@dataclass(frozen=True)
class TemporalAssessment:
    """Parsed UTC timestamps and semantic temporal eligibility for a candidate."""

    candidate_id: str
    assessed_at: datetime
    max_future_clock_skew_seconds: int
    observed_at: datetime | None
    valid_from: datetime | None
    valid_until: datetime | None
    eligible: bool
    pending_reason: str | None
    warnings: tuple[ReconciliationWarning, ...]

    def __post_init__(self) -> None:
        _validate_required_text(self.candidate_id, "candidate_id")
        for field_name in (
            "assessed_at",
            "observed_at",
            "valid_from",
            "valid_until",
        ):
            value = getattr(self, field_name)
            if value is None and field_name != "assessed_at":
                continue
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
                or value.utcoffset() != timedelta(0)
            ):
                raise ReconciliationInputError(
                    f"{field_name} must be a timezone-aware UTC datetime or None"
                )
        if (
            isinstance(self.max_future_clock_skew_seconds, bool)
            or not isinstance(self.max_future_clock_skew_seconds, int)
            or self.max_future_clock_skew_seconds < 0
        ):
            raise ReconciliationInputError(
                "max_future_clock_skew_seconds must be a non-negative integer"
            )
        if not isinstance(self.eligible, bool):
            raise ReconciliationInputError("eligible must be a boolean")
        if self.pending_reason is not None:
            _validate_required_text(self.pending_reason, "pending_reason")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(warning, ReconciliationWarning)
            for warning in self.warnings
        ):
            raise ReconciliationInputError(
                "warnings must be a tuple of ReconciliationWarning records"
            )
        if self.warnings != tuple(sorted(self.warnings, key=_warning_sort_key)):
            raise ReconciliationInputError("warnings must be sorted")


@dataclass(frozen=True)
class SameValueLineage:
    """Immutable candidate lineage that never emits fact-level self-edges."""

    outcome: LineageOutcome
    surviving_candidate_ids: tuple[str, ...]
    superseded_candidate_ids: tuple[str, ...]
    requires_human_review: bool
    warnings: tuple[ReconciliationWarning, ...] = ()
    fact_relation_requests: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, LineageOutcome):
            raise ReconciliationInputError("outcome must be a LineageOutcome")
        if not isinstance(self.surviving_candidate_ids, tuple):
            raise ReconciliationInputError(
                "surviving_candidate_ids must be a tuple"
            )
        for candidate_id in self.surviving_candidate_ids:
            _validate_required_text(candidate_id, "surviving_candidate_ids")
        if self.surviving_candidate_ids != tuple(
            sorted(set(self.surviving_candidate_ids))
        ):
            raise ReconciliationInputError(
                "surviving_candidate_ids must be sorted and unique"
            )
        if not isinstance(self.superseded_candidate_ids, tuple):
            raise ReconciliationInputError(
                "superseded_candidate_ids must be a tuple"
            )
        for candidate_id in self.superseded_candidate_ids:
            _validate_required_text(candidate_id, "superseded_candidate_ids")
        if self.superseded_candidate_ids != tuple(
            sorted(set(self.superseded_candidate_ids))
        ):
            raise ReconciliationInputError(
                "superseded_candidate_ids must be sorted and unique"
            )
        if set(self.surviving_candidate_ids) & set(
            self.superseded_candidate_ids
        ):
            raise ReconciliationInputError(
                "surviving and superseded candidate IDs must not overlap"
            )
        if not isinstance(self.requires_human_review, bool):
            raise ReconciliationInputError(
                "requires_human_review must be a boolean"
            )
        if (
            self.outcome is not LineageOutcome.UNRESOLVED
            and not self.surviving_candidate_ids
        ):
            raise ReconciliationInputError(
                "resolved lineage requires a surviving candidate"
            )
        if (
            self.outcome is LineageOutcome.UNRESOLVED
            and not self.requires_human_review
        ):
            raise ReconciliationInputError(
                "unresolved lineage requires human review"
            )
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(warning, ReconciliationWarning)
            for warning in self.warnings
        ):
            raise ReconciliationInputError(
                "warnings must be a tuple of ReconciliationWarning records"
            )
        if self.warnings != tuple(sorted(self.warnings, key=_warning_sort_key)):
            raise ReconciliationInputError("warnings must be sorted")
        if self.fact_relation_requests != ():
            raise ReconciliationInputError(
                "same-value lineage cannot request fact-level relations"
            )


_FACT_DECISION_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class FactDecision:
    """Immutable decision constructible only from canonical grouping rules."""

    fact_id: str
    subject: str
    predicate: str
    selected_value: EvidenceValue[FrozenJsonValue]
    candidate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    activation_witness_candidate_ids: tuple[str, ...]
    superseded_candidate_ids: tuple[str, ...]
    status: ReconciliationStatus
    confidence: EvidenceValue[float]
    valid_from: EvidenceValue[str]
    resolution_method: ResolutionMethod
    reason: str
    requires_human_review: bool
    warnings: tuple[ReconciliationWarning, ...] = ()
    relation_requests: tuple[object, ...] = ()

    def __new__(cls, *args: object, **kwargs: object) -> FactDecision:
        raise TypeError("FactDecision instances must be created by classify_group")

    @classmethod
    def _from_canonical_group(
        cls,
        construction_token: object,
        *,
        fact_id: str,
        subject: str,
        predicate: str,
        selected_value: EvidenceValue[FrozenJsonValue],
        candidate_ids: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        activation_witness_candidate_ids: tuple[str, ...],
        superseded_candidate_ids: tuple[str, ...],
        status: ReconciliationStatus,
        confidence: EvidenceValue[float],
        valid_from: EvidenceValue[str],
        resolution_method: ResolutionMethod,
        reason: str,
        requires_human_review: bool,
        warnings: tuple[ReconciliationWarning, ...],
    ) -> FactDecision:
        if construction_token is not _FACT_DECISION_CONSTRUCTION_TOKEN:
            raise TypeError(
                "FactDecision instances must be created by classify_group"
            )
        if set(activation_witness_candidate_ids) & set(
            superseded_candidate_ids
        ):
            raise ReconciliationInputError(
                "activation witness and superseded candidate IDs overlap"
            )
        instance = object.__new__(cls)
        values = {
            "fact_id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "selected_value": selected_value,
            "candidate_ids": candidate_ids,
            "evidence_refs": evidence_refs,
            "activation_witness_candidate_ids": (
                activation_witness_candidate_ids
            ),
            "superseded_candidate_ids": superseded_candidate_ids,
            "status": status,
            "confidence": confidence,
            "valid_from": valid_from,
            "resolution_method": resolution_method,
            "reason": reason,
            "requires_human_review": requires_human_review,
            "warnings": warnings,
            "relation_requests": (),
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance

    @property
    def method(self) -> ResolutionMethod:
        """Compatibility alias for provisional rule code."""

        return self.resolution_method


def _validate_fact_id_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Copy and validate one canonical fact ID/provenance tuple."""

    if type(value) is not tuple:
        raise ReconciliationInputError(f"{field_name} must be an exact tuple")
    snapshot = tuple(value)
    if not allow_empty and not snapshot:
        raise ReconciliationInputError(f"{field_name} must be non-empty")
    for item in snapshot:
        _validate_required_text(item, field_name)
    if snapshot != tuple(sorted(snapshot)):
        raise ReconciliationInputError(f"{field_name} must be sorted")
    if len(snapshot) != len(set(snapshot)):
        raise ReconciliationInputError(
            f"{field_name} must not contain duplicates"
        )
    return snapshot


def _freeze_reconciled_json_value(value: object) -> FrozenJsonValue:
    """Deep-copy both source JSON and already-frozen fact JSON values."""

    if type(value) is tuple:
        return tuple(_freeze_reconciled_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_json_mapping(
            value,
            _freeze_reconciled_json_value,
        )
    return freeze_json_value(value)


_RECONCILED_FACT_CONSTRUCTION_TOKEN = object()


_BASE_FACT_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class BaseFact:
    """One immutable reconciled fact before relation-derived indexes."""

    fact_id: str
    subject: str
    predicate: str
    selected_value: EvidenceValue[FrozenJsonValue]
    status: ReconciliationStatus
    confidence: EvidenceValue[float]
    reason: str
    evidence_refs: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    superseded_candidate_ids: tuple[str, ...]
    valid_from: EvidenceValue[str]
    resolved_at: EvidenceValue[str]
    resolution_method: ResolutionMethod
    requires_human_review: bool
    activation_witness_candidate_ids: tuple[str, ...]

    def __new__(cls, *args: object, **kwargs: object) -> BaseFact:
        raise TypeError("BaseFact instances must be created by reconciliation")

    @classmethod
    def _from_fields(
        cls,
        construction_token: object,
        **values: object,
    ) -> BaseFact:
        if (
            construction_token is not _BASE_FACT_CONSTRUCTION_TOKEN
            or cls is not BaseFact
        ):
            raise TypeError(
                "BaseFact instances must be created by reconciliation"
            )
        expected_fields = tuple(field.name for field in fields(cls))
        if set(values) != set(expected_fields):
            raise ReconciliationInputError(
                "BaseFact construction requires its exact fields"
            )
        instance = object.__new__(cls)
        for field_name in expected_fields:
            object.__setattr__(instance, field_name, values[field_name])
        instance._validate_and_freeze()
        return instance

    def _validate_and_freeze(self) -> None:
        for field_name in ("fact_id", "subject", "predicate", "reason"):
            _validate_required_text(getattr(self, field_name), field_name)
        if type(self.status) is not ReconciliationStatus:
            raise ReconciliationInputError(
                "status must be a ReconciliationStatus"
            )
        if type(self.resolution_method) is not ResolutionMethod:
            raise ReconciliationInputError(
                "resolution_method must be a ResolutionMethod"
            )
        if type(self.requires_human_review) is not bool:
            raise ReconciliationInputError(
                "requires_human_review must be a boolean"
            )
        object.__setattr__(
            self,
            "selected_value",
            _freeze_evidence_value(
                self.selected_value,
                "selected_value",
                _freeze_reconciled_json_value,
            ),
        )

        def freeze_confidence(known_value: object) -> float:
            if (
                type(known_value) is not float
                or not math.isfinite(known_value)
                or not 0.0 <= known_value <= 1.0
            ):
                raise ReconciliationInputError(
                    "must be a finite float within [0.0, 1.0]"
                )
            return known_value

        object.__setattr__(
            self,
            "confidence",
            _freeze_evidence_value(
                self.confidence,
                "confidence",
                freeze_confidence,
            ),
        )
        object.__setattr__(
            self,
            "valid_from",
            _freeze_expected_type(
                self.valid_from,
                "valid_from",
                str,
                "string",
            ),
        )
        resolved_at = _freeze_expected_type(
            self.resolved_at,
            "resolved_at",
            str,
            "string",
        )
        if (
            resolved_at.status is not EvidenceStatus.KNOWN
            or resolved_at.source != "reconciliation:clock"
        ):
            raise ReconciliationInputError(
                "resolved_at must be KNOWN from reconciliation:clock"
            )
        object.__setattr__(self, "resolved_at", resolved_at)
        for field_name, allow_empty in (
            ("evidence_refs", False),
            ("candidate_ids", False),
            ("superseded_candidate_ids", True),
            ("activation_witness_candidate_ids", True),
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_fact_id_tuple(
                    getattr(self, field_name),
                    field_name,
                    allow_empty=allow_empty,
                ),
            )
        candidate_ids = set(self.candidate_ids)
        superseded_ids = set(self.superseded_candidate_ids)
        witness_ids = set(self.activation_witness_candidate_ids)
        if not superseded_ids <= candidate_ids:
            raise ReconciliationInputError(
                "superseded_candidate_ids must be a subset of candidate_ids"
            )
        if not witness_ids <= candidate_ids:
            raise ReconciliationInputError(
                "activation_witness_candidate_ids must be a subset of candidate_ids"
            )
        if witness_ids & superseded_ids:
            raise ReconciliationInputError(
                "activation witnesses and superseded candidates must not overlap"
            )
        if self.status is ReconciliationStatus.ACTIVE:
            if not self.activation_witness_candidate_ids:
                raise ReconciliationInputError(
                    "ACTIVE facts require an activation witness"
                )
        elif self.activation_witness_candidate_ids:
            raise ReconciliationInputError(
                "non-ACTIVE facts cannot retain activation witnesses"
            )


@dataclass(frozen=True, init=False)
class ReconciledFact:
    """One immutable fact whose navigation indexes are graph-derived."""

    fact_id: str
    subject: str
    predicate: str
    selected_value: EvidenceValue[FrozenJsonValue]
    status: ReconciliationStatus
    confidence: EvidenceValue[float]
    reason: str
    evidence_refs: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    superseded_candidate_ids: tuple[str, ...]
    conflict_candidate_ids: tuple[str, ...]
    valid_from: EvidenceValue[str]
    resolved_at: EvidenceValue[str]
    resolution_method: ResolutionMethod
    requires_human_review: bool
    activation_witness_candidate_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    supersedes_fact_ids: tuple[str, ...]
    superseded_by_fact_ids: tuple[str, ...]
    conflict_fact_ids: tuple[str, ...]

    def __new__(cls, *args: object, **kwargs: object) -> ReconciledFact:
        raise TypeError(
            "ReconciledFact instances must be created by reconciliation"
        )

    @classmethod
    def _from_materialized(
        cls,
        construction_token: object,
        base_fact: BaseFact,
        *,
        conflict_candidate_ids: tuple[str, ...],
        relation_ids: tuple[str, ...],
        supersedes_fact_ids: tuple[str, ...],
        superseded_by_fact_ids: tuple[str, ...],
        conflict_fact_ids: tuple[str, ...],
    ) -> ReconciledFact:
        """Build one final fact with all graph-derived indexes at once."""

        if type(base_fact) is not BaseFact:
            raise ReconciliationInputError(
                "materialization requires an exact authenticated BaseFact"
            )
        return cls._from_fields(
            construction_token,
            fact_id=base_fact.fact_id,
            subject=base_fact.subject,
            predicate=base_fact.predicate,
            selected_value=base_fact.selected_value,
            status=base_fact.status,
            confidence=base_fact.confidence,
            reason=base_fact.reason,
            evidence_refs=base_fact.evidence_refs,
            candidate_ids=base_fact.candidate_ids,
            superseded_candidate_ids=base_fact.superseded_candidate_ids,
            conflict_candidate_ids=conflict_candidate_ids,
            valid_from=base_fact.valid_from,
            resolved_at=base_fact.resolved_at,
            resolution_method=base_fact.resolution_method,
            requires_human_review=base_fact.requires_human_review,
            activation_witness_candidate_ids=(
                base_fact.activation_witness_candidate_ids
            ),
            relation_ids=relation_ids,
            supersedes_fact_ids=supersedes_fact_ids,
            superseded_by_fact_ids=superseded_by_fact_ids,
            conflict_fact_ids=conflict_fact_ids,
        )

    @classmethod
    def _from_fields(
        cls,
        construction_token: object,
        **values: object,
    ) -> ReconciledFact:
        if (
            construction_token is not _RECONCILED_FACT_CONSTRUCTION_TOKEN
            or cls is not ReconciledFact
        ):
            raise TypeError(
                "ReconciledFact instances must be created by reconciliation"
            )
        expected_fields = tuple(field.name for field in fields(cls))
        if set(values) != set(expected_fields):
            raise ReconciliationInputError(
                "ReconciledFact construction requires its exact fields"
            )
        instance = object.__new__(cls)
        for field_name in expected_fields:
            object.__setattr__(instance, field_name, values[field_name])
        instance._validate_and_freeze()
        return instance

    def _validate_and_freeze(self) -> None:
        for field_name in ("fact_id", "subject", "predicate", "reason"):
            _validate_required_text(getattr(self, field_name), field_name)
        if type(self.status) is not ReconciliationStatus:
            raise ReconciliationInputError(
                "status must be a ReconciliationStatus"
            )
        if type(self.resolution_method) is not ResolutionMethod:
            raise ReconciliationInputError(
                "resolution_method must be a ResolutionMethod"
            )
        if type(self.requires_human_review) is not bool:
            raise ReconciliationInputError(
                "requires_human_review must be a boolean"
            )

        object.__setattr__(
            self,
            "selected_value",
            _freeze_evidence_value(
                self.selected_value,
                "selected_value",
                _freeze_reconciled_json_value,
            ),
        )

        def freeze_confidence(known_value: object) -> float:
            if (
                type(known_value) is not float
                or not math.isfinite(known_value)
                or not 0.0 <= known_value <= 1.0
            ):
                raise ReconciliationInputError(
                    "must be a finite float within [0.0, 1.0]"
                )
            return known_value

        object.__setattr__(
            self,
            "confidence",
            _freeze_evidence_value(
                self.confidence,
                "confidence",
                freeze_confidence,
            ),
        )
        object.__setattr__(
            self,
            "valid_from",
            _freeze_expected_type(
                self.valid_from,
                "valid_from",
                str,
                "string",
            ),
        )
        resolved_at = _freeze_expected_type(
            self.resolved_at,
            "resolved_at",
            str,
            "string",
        )
        if (
            resolved_at.status is not EvidenceStatus.KNOWN
            or resolved_at.source != "reconciliation:clock"
        ):
            raise ReconciliationInputError(
                "resolved_at must be KNOWN from reconciliation:clock"
            )
        object.__setattr__(self, "resolved_at", resolved_at)

        required_collections = ("evidence_refs", "candidate_ids")
        optional_collections = (
            "superseded_candidate_ids",
            "conflict_candidate_ids",
            "activation_witness_candidate_ids",
            "relation_ids",
            "supersedes_fact_ids",
            "superseded_by_fact_ids",
            "conflict_fact_ids",
        )
        for field_name in required_collections:
            object.__setattr__(
                self,
                field_name,
                _validate_fact_id_tuple(
                    getattr(self, field_name),
                    field_name,
                    allow_empty=False,
                ),
            )
        for field_name in optional_collections:
            object.__setattr__(
                self,
                field_name,
                _validate_fact_id_tuple(
                    getattr(self, field_name),
                    field_name,
                    allow_empty=True,
                ),
            )

        candidate_ids = set(self.candidate_ids)
        superseded_ids = set(self.superseded_candidate_ids)
        witness_ids = set(self.activation_witness_candidate_ids)
        if not superseded_ids <= candidate_ids:
            raise ReconciliationInputError(
                "superseded_candidate_ids must be a subset of candidate_ids"
            )
        if not witness_ids <= candidate_ids:
            raise ReconciliationInputError(
                "activation_witness_candidate_ids must be a subset of candidate_ids"
            )
        if witness_ids & superseded_ids:
            raise ReconciliationInputError(
                "activation witnesses and superseded candidates must not overlap"
            )
        if self.status is ReconciliationStatus.ACTIVE:
            if not self.activation_witness_candidate_ids:
                raise ReconciliationInputError(
                    "ACTIVE facts require an activation witness"
                )
        elif self.activation_witness_candidate_ids:
            raise ReconciliationInputError(
                "non-ACTIVE facts cannot retain activation witnesses"
            )


_RELATION_GRAPH_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class RelationGraph:
    """A complete immutable relation set and its materialized facts."""

    facts: tuple[ReconciledFact, ...]
    relations: tuple[RelationRecord, ...]
    _facts_by_id: Mapping[str, ReconciledFact] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __new__(cls, *args: object, **kwargs: object) -> RelationGraph:
        raise TypeError(
            "RelationGraph instances must be created by materialization"
        )

    @classmethod
    def _from_materialized(
        cls,
        construction_token: object,
        *,
        facts: tuple[ReconciledFact, ...],
        relations: tuple[RelationRecord, ...],
    ) -> RelationGraph:
        if (
            construction_token is not _RELATION_GRAPH_CONSTRUCTION_TOKEN
            or cls is not RelationGraph
        ):
            raise TypeError(
                "RelationGraph instances must be created by materialization"
            )
        if type(facts) is not tuple or any(
            type(fact) is not ReconciledFact for fact in facts
        ):
            raise ReconciliationInputError(
                "facts must be an exact tuple of ReconciledFact records"
            )
        if tuple(fact.fact_id for fact in facts) != tuple(
            sorted(fact.fact_id for fact in facts)
        ) or len({fact.fact_id for fact in facts}) != len(facts):
            raise ReconciliationInputError(
                "materialized facts must have sorted unique fact IDs"
            )
        if type(relations) is not tuple or any(
            type(relation) is not RelationRecord for relation in relations
        ):
            raise ReconciliationInputError(
                "relations must be an exact tuple of RelationRecord records"
            )
        relation_keys = tuple(
            (
                relation.relation_type.value,
                relation.from_fact_id,
                relation.to_fact_id,
            )
            for relation in relations
        )
        if relation_keys != tuple(sorted(relation_keys)) or len(
            {relation.relation_id for relation in relations}
        ) != len(relations):
            raise ReconciliationInputError(
                "materialized relations must be sorted and unique"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "facts", facts)
        object.__setattr__(instance, "relations", relations)
        object.__setattr__(
            instance,
            "_facts_by_id",
            MappingProxyType({fact.fact_id: fact for fact in facts}),
        )
        return instance

    def fact(self, fact_id: str) -> ReconciledFact:
        """Return one materialized fact by its stable identifier."""

        _validate_required_text(fact_id, "fact_id")
        return self._facts_by_id[fact_id]


@dataclass(frozen=True)
class UnresolvedCandidate:
    """Provenance retained for a candidate whose value cannot be resolved."""

    candidate_id: str
    subject: str
    predicate: str
    evidence_status: EvidenceStatus
    reason: str
    source_type: SourceType
    source_ref: str
    field_source: str
    related_fact_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "subject",
            "predicate",
            "reason",
            "source_ref",
            "field_source",
            "related_fact_id",
        ):
            _validate_required_text(getattr(self, field_name), field_name)
        if not isinstance(self.evidence_status, EvidenceStatus):
            raise ReconciliationInputError(
                "evidence_status must be an EvidenceStatus"
            )
        if self.evidence_status is EvidenceStatus.KNOWN:
            raise ReconciliationInputError(
                "evidence_status must be UNKNOWN or UNAVAILABLE"
            )
        if not isinstance(self.source_type, SourceType):
            raise ReconciliationInputError("source_type must be a SourceType")


_SUMMARY_COUNTS_CONSTRUCTION_TOKEN = object()


def _make_summary_count_token_authority():
    trusted_token = _SUMMARY_COUNTS_CONSTRUCTION_TOKEN

    def issue() -> object:
        return trusted_token

    def validate(candidate: object) -> bool:
        return candidate is trusted_token

    return issue, validate


_issue_summary_count_token, _is_summary_count_token = (
    _make_summary_count_token_authority()
)
del _make_summary_count_token_authority


@dataclass(frozen=True, init=False)
class SummaryCounts:
    """Exact result counts derived only from validated result collections."""

    active: int
    superseded: int
    conflicted: int
    pending: int
    deprecated: int
    relations: int
    warnings: int
    unresolved: int

    def __new__(cls, *args: object, **kwargs: object) -> SummaryCounts:
        raise TypeError("SummaryCounts instances are derived by reconciliation")

    @classmethod
    def _derive(
        cls,
        construction_token: object,
        *,
        active: tuple[ReconciledFact, ...],
        superseded: tuple[ReconciledFact, ...],
        conflicted: tuple[ReconciledFact, ...],
        pending: tuple[ReconciledFact, ...],
        deprecated: tuple[ReconciledFact, ...],
        relations: tuple[RelationRecord, ...],
        warnings: tuple[ReconciliationWarning, ...],
        unresolved: tuple[UnresolvedCandidate, ...],
    ) -> SummaryCounts:
        if (
            cls is not SummaryCounts
            or not _is_summary_count_token(construction_token)
        ):
            raise TypeError(
                "SummaryCounts instances are derived by reconciliation"
            )
        instance = object.__new__(cls)
        values = {
            "active": len(active),
            "superseded": len(superseded),
            "conflicted": len(conflicted),
            "pending": len(pending),
            "deprecated": len(deprecated),
            "relations": len(relations),
            "warnings": len(warnings),
            "unresolved": len(unresolved),
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance

    def to_dict(self) -> dict[str, int]:
        """Return exact primitive count fields in model order."""

        return {
            field_name: object.__getattribute__(self, field_name)
            for field_name in (
                "active",
                "superseded",
                "conflicted",
                "pending",
                "deprecated",
                "relations",
                "warnings",
                "unresolved",
            )
        }


_RECONCILIATION_RESULT_CONSTRUCTION_TOKEN = object()


def _make_result_token_authority():
    trusted_token = _RECONCILIATION_RESULT_CONSTRUCTION_TOKEN

    def issue() -> object:
        return trusted_token

    def validate(candidate: object) -> bool:
        return candidate is trusted_token

    return issue, validate


_issue_result_token, _is_result_token = _make_result_token_authority()
del _make_result_token_authority


def _snapshot_result_sequence(
    value: object,
    field_name: str,
) -> tuple[object, ...]:
    """Take one guarded snapshot of an externally supplied collection."""

    if type(value) not in (list, tuple):
        raise ReconciliationInputError(
            f"{field_name} must be an exact list or tuple"
        )
    try:
        return tuple(value)
    except Exception as error:
        raise ReconciliationInputError(
            f"{field_name} must be an exact list or tuple"
        ) from error


def _result_fact_key(fact: ReconciledFact) -> tuple[str, str, str]:
    return (fact.subject, fact.predicate, fact.fact_id)


def _result_relation_key(
    relation: RelationRecord,
) -> tuple[str, str, str, str]:
    return (
        relation.relation_type.value,
        relation.from_fact_id,
        relation.to_fact_id,
        relation.relation_id,
    )


def _result_unresolved_key(
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


def _require_exact_result_records(
    values: tuple[object, ...],
    expected_type: type,
    field_name: str,
) -> None:
    if any(type(value) is not expected_type for value in values):
        raise ReconciliationInvariantError(
            f"{field_name} must contain only exact {expected_type.__name__} records"
        )


def _sorted_result_records(
    values: tuple[object, ...],
    key: Callable[[object], object],
    field_name: str,
) -> tuple:
    try:
        return tuple(sorted(values, key=key))
    except Exception as error:
        raise ReconciliationInvariantError(
            f"{field_name} must contain complete canonical records"
        ) from error


@dataclass(frozen=True, init=False)
class ReconciliationResult:
    """Immutable, authenticated, five-partition reconciliation output."""

    schema_version: str
    project_id: str
    snapshot_id: str
    reconciled_at: str
    active: tuple[ReconciledFact, ...]
    superseded: tuple[ReconciledFact, ...]
    conflicted: tuple[ReconciledFact, ...]
    pending: tuple[ReconciledFact, ...]
    deprecated: tuple[ReconciledFact, ...]
    relations: tuple[RelationRecord, ...]
    unresolved: tuple[UnresolvedCandidate, ...]
    warnings: tuple[ReconciliationWarning, ...]
    unresolved_count: int
    human_review_required: bool
    summary_counts: SummaryCounts

    def __new__(cls, *args: object, **kwargs: object) -> ReconciliationResult:
        raise TypeError(
            "ReconciliationResult instances must be created by create"
        )

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        snapshot_id: str,
        reconciled_at: str,
        graph: RelationGraph,
        source_decisions: list[FactDecision] | tuple[FactDecision, ...],
        unresolved: list[UnresolvedCandidate]
        | tuple[UnresolvedCandidate, ...] = (),
        warnings: list[ReconciliationWarning]
        | tuple[ReconciliationWarning, ...] = (),
    ) -> ReconciliationResult:
        """Validate authenticated graph context and derive all result truth."""

        if cls is not ReconciliationResult:
            raise ReconciliationInputError(
                "result factory requires exact ReconciliationResult"
            )
        for field_name, value in (
            ("project_id", project_id),
            ("snapshot_id", snapshot_id),
            ("reconciled_at", reconciled_at),
        ):
            _validate_required_text(value, field_name)
        if type(graph) is not RelationGraph:
            raise ReconciliationInputError(
                "graph must be an exact RelationGraph"
            )

        decision_snapshot = _snapshot_result_sequence(
            source_decisions,
            "source_decisions",
        )
        unresolved_snapshot = _snapshot_result_sequence(
            unresolved,
            "unresolved",
        )
        warning_snapshot = _snapshot_result_sequence(warnings, "warnings")
        _require_exact_result_records(
            decision_snapshot,
            FactDecision,
            "source_decisions",
        )
        _require_exact_result_records(
            unresolved_snapshot,
            UnresolvedCandidate,
            "unresolved",
        )
        _require_exact_result_records(
            warning_snapshot,
            ReconciliationWarning,
            "warnings",
        )

        from agent_memory_os.reconcile.reconciler import (
            _complete_warning_union,
            validate_relation_graph,
            validate_result_fields,
        )

        warning_snapshot = _complete_warning_union(
            decision_snapshot,
            warning_snapshot,
        )

        validate_relation_graph(graph)
        try:
            cohort = _validate_reconciliation_cohort(
                object.__getattribute__(graph, "_cohort"),
                require_bound=True,
            )
        except (AttributeError, TypeError, ReconciliationInputError) as error:
            raise ReconciliationInvariantError(
                "result graph must retain a valid reconciliation cohort"
            ) from error
        for field_name, supplied, canonical in (
            ("project_id", project_id, cohort.project_id),
            ("snapshot_id", snapshot_id, cohort.snapshot_id),
            ("reconciled_at", reconciled_at, cohort.reconciled_at),
        ):
            if supplied != canonical:
                raise ReconciliationInputError(
                    f"{field_name} must match the authenticated graph cohort"
                )

        facts = tuple(object.__getattribute__(graph, "facts"))
        relations = tuple(object.__getattribute__(graph, "relations"))
        _require_exact_result_records(facts, ReconciledFact, "facts")
        _require_exact_result_records(relations, RelationRecord, "relations")
        facts = _sorted_result_records(facts, _result_fact_key, "facts")
        relations = _sorted_result_records(
            relations,
            _result_relation_key,
            "relations",
        )
        status_partitions = {
            status: tuple(fact for fact in facts if fact.status is status)
            for status in ReconciliationStatus
        }
        checked_unresolved = _sorted_result_records(
            unresolved_snapshot,
            _result_unresolved_key,
            "unresolved",
        )
        checked_warnings = _sorted_result_records(
            warning_snapshot,
            _warning_sort_key,
            "warnings",
        )
        try:
            review_required = (
                any(fact.requires_human_review for fact in facts)
                or bool(checked_unresolved)
                or any(
                    warning.requires_human_review
                    for warning in checked_warnings
                )
            )
        except (AttributeError, TypeError) as error:
            raise ReconciliationInvariantError(
                "facts and warnings must retain complete review fields"
            ) from error
        validate_result_fields(
            facts=facts,
            relations=relations,
            partitions=status_partitions,
            unresolved=checked_unresolved,
            warnings=checked_warnings,
            human_review_required=review_required,
            source_decisions=decision_snapshot,
        )

        active = status_partitions[ReconciliationStatus.ACTIVE]
        superseded = status_partitions[ReconciliationStatus.SUPERSEDED]
        conflicted = status_partitions[ReconciliationStatus.CONFLICTED]
        pending = status_partitions[ReconciliationStatus.PENDING]
        deprecated = status_partitions[ReconciliationStatus.DEPRECATED]
        summary_counts = SummaryCounts._derive(
            _issue_summary_count_token(),
            active=active,
            superseded=superseded,
            conflicted=conflicted,
            pending=pending,
            deprecated=deprecated,
            relations=relations,
            warnings=checked_warnings,
            unresolved=checked_unresolved,
        )
        return cls._from_validated_fields(
            _issue_result_token(),
            schema_version="reconciliation:v1",
            project_id=project_id,
            snapshot_id=snapshot_id,
            reconciled_at=reconciled_at,
            active=active,
            superseded=superseded,
            conflicted=conflicted,
            pending=pending,
            deprecated=deprecated,
            relations=relations,
            unresolved=checked_unresolved,
            warnings=checked_warnings,
            unresolved_count=len(checked_unresolved),
            human_review_required=review_required,
            summary_counts=summary_counts,
            graph=graph,
            source_decisions=_sorted_result_records(
                decision_snapshot,
                lambda item: item.fact_id,
                "source_decisions",
            ),
        )

    @classmethod
    def _from_validated_fields(
        cls,
        construction_token: object,
        *,
        graph: RelationGraph,
        source_decisions: tuple[FactDecision, ...],
        **values: object,
    ) -> ReconciliationResult:
        if (
            cls is not ReconciliationResult
            or not _is_result_token(construction_token)
        ):
            raise TypeError(
                "ReconciliationResult instances must be created by create"
            )
        expected_fields = tuple(field.name for field in fields(cls))
        if set(values) != set(expected_fields):
            raise ReconciliationInputError(
                "ReconciliationResult construction requires its exact fields"
            )
        instance = object.__new__(cls)
        for field_name in expected_fields:
            object.__setattr__(instance, field_name, values[field_name])
        object.__setattr__(instance, "_graph", graph)
        object.__setattr__(instance, "_source_decisions", source_decisions)
        instance._validate_for_serialization()
        return instance

    def all_facts(self) -> tuple[ReconciledFact, ...]:
        """Return every status partition in fixed status order."""

        return (
            self.active
            + self.superseded
            + self.conflicted
            + self.pending
            + self.deprecated
        )

    def status_partitions(
        self,
    ) -> dict[ReconciliationStatus, tuple[ReconciledFact, ...]]:
        """Return fresh status-keyed views for invariant validation."""

        return {
            ReconciliationStatus.ACTIVE: self.active,
            ReconciliationStatus.SUPERSEDED: self.superseded,
            ReconciliationStatus.CONFLICTED: self.conflicted,
            ReconciliationStatus.PENDING: self.pending,
            ReconciliationStatus.DEPRECATED: self.deprecated,
        }

    def _validate_for_serialization(self) -> None:
        """Revalidate public values against retained Task 19 authority."""

        from agent_memory_os.reconcile.reconciler import (
            validate_relation_graph,
            validate_result_fields,
        )

        try:
            values = {
                field.name: object.__getattribute__(self, field.name)
                for field in fields(ReconciliationResult)
            }
            graph = object.__getattribute__(self, "_graph")
            source_decisions = object.__getattribute__(
                self,
                "_source_decisions",
            )
        except (AttributeError, TypeError) as error:
            raise ReconciliationInvariantError(
                "ReconciliationResult must retain complete validated fields"
            ) from error
        if type(values["schema_version"]) is not str or (
            values["schema_version"] != "reconciliation:v1"
        ):
            raise ReconciliationInvariantError(
                "schema_version must be exactly reconciliation:v1"
            )
        for field_name in ("project_id", "snapshot_id", "reconciled_at"):
            if type(values[field_name]) is not str:
                raise ReconciliationInvariantError(
                    f"{field_name} must be an exact string"
                )
            try:
                _validate_required_text(values[field_name], field_name)
            except ReconciliationInputError as error:
                raise ReconciliationInvariantError(str(error)) from error

        partition_names = (
            "active",
            "superseded",
            "conflicted",
            "pending",
            "deprecated",
        )
        for field_name in (*partition_names, "relations", "unresolved", "warnings"):
            if type(values[field_name]) is not tuple:
                raise ReconciliationInvariantError(
                    f"{field_name} must be an exact immutable tuple"
                )
        facts = self.all_facts()
        _require_exact_result_records(facts, ReconciledFact, "facts")
        _require_exact_result_records(
            values["relations"],
            RelationRecord,
            "relations",
        )
        _require_exact_result_records(
            values["unresolved"],
            UnresolvedCandidate,
            "unresolved",
        )
        _require_exact_result_records(
            values["warnings"],
            ReconciliationWarning,
            "warnings",
        )
        for partition_name in partition_names:
            partition = values[partition_name]
            if partition != _sorted_result_records(
                partition,
                _result_fact_key,
                partition_name,
            ):
                raise ReconciliationInvariantError(
                    f"{partition_name} facts must be canonically sorted"
                )
        if values["relations"] != _sorted_result_records(
            values["relations"],
            _result_relation_key,
            "relations",
        ):
            raise ReconciliationInvariantError(
                "relations must be canonically sorted"
            )
        if values["warnings"] != _sorted_result_records(
            values["warnings"],
            _warning_sort_key,
            "warnings",
        ):
            raise ReconciliationInvariantError(
                "warnings must be canonically sorted"
            )
        if values["unresolved"] != _sorted_result_records(
            values["unresolved"],
            _result_unresolved_key,
            "unresolved",
        ):
            raise ReconciliationInvariantError(
                "unresolved must be canonically sorted"
            )

        if type(values["unresolved_count"]) is not int or (
            values["unresolved_count"] != len(values["unresolved"])
        ):
            raise ReconciliationInvariantError(
                "unresolved_count must equal the exact unresolved count"
            )
        try:
            expected_review = (
                any(fact.requires_human_review for fact in facts)
                or bool(values["unresolved"])
                or any(
                    warning.requires_human_review
                    for warning in values["warnings"]
                )
            )
        except (AttributeError, TypeError) as error:
            raise ReconciliationInvariantError(
                "facts and warnings must retain complete review fields"
            ) from error
        if type(values["human_review_required"]) is not bool or (
            values["human_review_required"] != expected_review
        ):
            raise ReconciliationInvariantError(
                "human_review_required must equal the exact review formula"
            )
        expected_counts = {
            "active": len(values["active"]),
            "superseded": len(values["superseded"]),
            "conflicted": len(values["conflicted"]),
            "pending": len(values["pending"]),
            "deprecated": len(values["deprecated"]),
            "relations": len(values["relations"]),
            "warnings": len(values["warnings"]),
            "unresolved": len(values["unresolved"]),
        }
        summary = values["summary_counts"]
        if type(summary) is not SummaryCounts:
            raise ReconciliationInvariantError(
                "summary_counts must be an exact SummaryCounts"
            )
        try:
            actual_counts = summary.to_dict()
        except (AttributeError, TypeError) as error:
            raise ReconciliationInvariantError(
                "summary_counts must retain complete integer fields"
            ) from error
        if any(type(value) is not int or value < 0 for value in actual_counts.values()):
            raise ReconciliationInvariantError(
                "summary_counts fields must be non-negative exact integers"
            )
        if actual_counts != expected_counts:
            raise ReconciliationInvariantError(
                "summary_counts must equal exact derived collection counts"
            )

        if type(graph) is not RelationGraph:
            raise ReconciliationInvariantError(
                "result must retain an exact RelationGraph"
            )
        if type(source_decisions) is not tuple:
            raise ReconciliationInvariantError(
                "result must retain exact source_decisions"
            )
        _require_exact_result_records(
            source_decisions,
            FactDecision,
            "source_decisions",
        )
        validate_relation_graph(graph)
        try:
            cohort = _validate_reconciliation_cohort(
                object.__getattribute__(graph, "_cohort"),
                require_bound=True,
            )
            graph_facts = object.__getattribute__(graph, "facts")
            graph_relations = object.__getattribute__(graph, "relations")
        except (AttributeError, TypeError, ReconciliationInputError) as error:
            raise ReconciliationInvariantError(
                "result graph context is invalid"
            ) from error
        if (
            values["project_id"] != cohort.project_id
            or values["snapshot_id"] != cohort.snapshot_id
            or values["reconciled_at"] != cohort.reconciled_at
        ):
            raise ReconciliationInvariantError(
                "result identity must match its authenticated graph cohort"
            )
        try:
            facts_by_id = {fact.fact_id: fact for fact in facts}
        except (AttributeError, TypeError) as error:
            raise ReconciliationInvariantError(
                "result facts must retain complete identity fields"
            ) from error
        if len(graph_facts) != len(facts_by_id) or any(
            facts_by_id.get(fact.fact_id) is not fact for fact in graph_facts
        ):
            raise ReconciliationInvariantError(
                "result facts must be the retained graph members"
            )
        if len(graph_relations) != len(values["relations"]) or any(
            left is not right
            for left, right in zip(graph_relations, values["relations"])
        ):
            raise ReconciliationInvariantError(
                "result relations must be the retained graph members"
            )
        validate_result_fields(
            facts=facts,
            relations=values["relations"],
            partitions=self.status_partitions(),
            unresolved=values["unresolved"],
            warnings=values["warnings"],
            human_review_required=values["human_review_required"],
            source_decisions=source_decisions,
        )

    def to_dict(self) -> dict[str, object]:
        """Return fully validated JSON-compatible primitive fields."""

        self._validate_for_serialization()
        from agent_memory_os.reconcile.serialization import (
            _result_to_primitive_dict,
        )

        return _result_to_primitive_dict(self)
