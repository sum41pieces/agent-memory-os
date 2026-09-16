"""Typed evidence values, snapshots, and deterministic serialization."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Generic, Mapping, TypeVar, cast, get_args, get_origin, get_type_hints


T = TypeVar("T")


class EvidenceStatus(str, Enum):
    """Whether a value is known, insufficiently evidenced, or unavailable."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class EvidenceValue(Generic[T]):
    """A value plus explicit epistemic state and field-level provenance."""

    status: EvidenceStatus
    value: T | None = None
    reason: str | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.status is EvidenceStatus.KNOWN:
            if self.value is None:
                raise ValueError("KNOWN evidence requires a value")
            if self.reason is not None:
                raise ValueError("KNOWN evidence cannot have a reason")
            return
        if self.value is not None:
            raise ValueError("non-known evidence forbids value")
        if not self.reason:
            raise ValueError("non-known evidence requires a reason")

    @classmethod
    def known(cls, value: T, *, source: str) -> EvidenceValue[T]:
        return cls(status=EvidenceStatus.KNOWN, value=value, source=source)

    @classmethod
    def unknown(cls, *, reason: str, source: str) -> EvidenceValue[T]:
        return cls(status=EvidenceStatus.UNKNOWN, reason=reason, source=source)

    @classmethod
    def unavailable(cls, *, reason: str, source: str) -> EvidenceValue[T]:
        return cls(status=EvidenceStatus.UNAVAILABLE, reason=reason, source=source)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status.value,
            "source": self.source,
        }
        if self.status is EvidenceStatus.KNOWN:
            result["value"] = to_primitive(self.value)
        else:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class DocumentRecord:
    path: EvidenceValue[str]
    title: EvidenceValue[str]
    summary: EvidenceValue[str]
    last_updated: EvidenceValue[str]


@dataclass(frozen=True)
class CommitRecord:
    sha: EvidenceValue[str]
    short_sha: EvidenceValue[str]
    authored_at: EvidenceValue[str]
    subject: EvidenceValue[str]


@dataclass(frozen=True)
class ChangedFileRecord:
    path: EvidenceValue[str]
    status: EvidenceValue[str]


@dataclass(frozen=True)
class DiscoveredCommand:
    name: EvidenceValue[str]
    command: EvidenceValue[str]
    kind: EvidenceValue[str]


@dataclass(frozen=True)
class RecordedTestResult:
    summary: EvidenceValue[str]
    execution_status: EvidenceValue[str]
    source_document: EvidenceValue[str]


@dataclass(frozen=True)
class RepositoryEvidence:
    path: EvidenceValue[str]
    exists: EvidenceValue[bool]
    project_name: EvidenceValue[str]


@dataclass(frozen=True)
class GitEvidence:
    is_repository: EvidenceValue[bool]
    branch: EvidenceValue[str]
    head_sha: EvidenceValue[str]
    head_short: EvidenceValue[str]
    tags_at_head: EvidenceValue[list[str]]
    remote_count: EvidenceValue[int]
    staged_count: EvidenceValue[int]
    modified_count: EvidenceValue[int]
    untracked_count: EvidenceValue[int]
    recent_commits: EvidenceValue[list[CommitRecord]]


@dataclass(frozen=True)
class DocsEvidence:
    discovered: EvidenceValue[list[str]]
    agents: EvidenceValue[DocumentRecord]
    project_context: EvidenceValue[DocumentRecord]
    state: EvidenceValue[DocumentRecord]
    next: EvidenceValue[DocumentRecord]
    decisions: EvidenceValue[DocumentRecord]
    session_log: EvidenceValue[DocumentRecord]
    readme: EvidenceValue[DocumentRecord]


@dataclass(frozen=True)
class TestsEvidence:
    discovered_test_roots: EvidenceValue[list[str]]
    python_test_files: EvidenceValue[list[str]]
    frontend_test_files: EvidenceValue[list[str]]
    discovered_commands: EvidenceValue[list[DiscoveredCommand]]
    last_recorded_results: EvidenceValue[list[RecordedTestResult]]


@dataclass(frozen=True)
class RecentChangesEvidence:
    changed_files: EvidenceValue[list[ChangedFileRecord]]
    diff_stat: EvidenceValue[str]
    recent_commit_summary: EvidenceValue[list[str]]


@dataclass(frozen=True)
class ShadowGuardReport:
    head_before: EvidenceValue[str]
    head_after: EvidenceValue[str]
    index_hash_before: EvidenceValue[str]
    index_hash_after: EvidenceValue[str]
    status_hash_before: EvidenceValue[str]
    status_hash_after: EvidenceValue[str]
    tracked_manifest_before: EvidenceValue[str]
    tracked_manifest_after: EvidenceValue[str]
    head_unchanged: EvidenceValue[bool]
    index_unchanged: EvidenceValue[bool]
    status_unchanged: EvidenceValue[bool]
    tracked_content_unchanged: EvidenceValue[bool]
    changed_components: EvidenceValue[list[str]]
    verdict: EvidenceValue[str]


@dataclass(frozen=True)
class CollectorEvidence:
    warnings: EvidenceValue[list[str]]
    errors: EvidenceValue[list[str]]
    evidence_sources: EvidenceValue[list[str]]
    shadow_guard: EvidenceValue[ShadowGuardReport]


@dataclass(frozen=True)
class EvidenceSnapshot:
    schema_version: EvidenceValue[str]
    project_id: EvidenceValue[str]
    captured_at: EvidenceValue[str]
    source_mode: EvidenceValue[str]
    repository: RepositoryEvidence
    git: GitEvidence
    docs: DocsEvidence
    tests: TestsEvidence
    recent_changes: RecentChangesEvidence
    collector: CollectorEvidence

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        if not isinstance(value, dict):
            raise TypeError("snapshot must serialize to a dictionary")
        return value


@dataclass(frozen=True)
class EvidenceDifference:
    path: str
    before: object
    after: object


def to_primitive(value: object) -> object:
    """Convert evidence dataclasses to JSON-compatible primitives."""

    if isinstance(value, EvidenceValue):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_primitive(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit timezone offset."""

    return datetime.now(timezone.utc).isoformat()


def snapshot_to_json(snapshot: EvidenceSnapshot) -> str:
    """Serialize a snapshot deterministically without escaping Unicode."""

    return json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def snapshot_from_json(text: str) -> EvidenceSnapshot:
    """Deserialize a snapshot while rejecting incompatible wire shapes."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evidence JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("evidence root must be an object")
    return _decode_dataclass(EvidenceSnapshot, payload, path="$")


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing keys at {path}: {missing}")
    if unexpected:
        raise ValueError(f"unexpected keys at {path}: {unexpected}")


def _decode_dataclass(
    model_type: type[T],
    payload: object,
    *,
    path: str,
) -> T:
    if type(payload) is not dict:
        raise ValueError(f"expected object at {path}")

    model_fields = fields(model_type)
    _require_exact_keys(payload, {field.name for field in model_fields}, path)
    type_hints = get_type_hints(model_type)
    values = {
        field.name: _decode_value(
            type_hints[field.name],
            payload[field.name],
            path=f"{path}.{field.name}",
        )
        for field in model_fields
    }
    return model_type(**values)


def _decode_value(expected_type: object, payload: object, *, path: str) -> object:
    origin = get_origin(expected_type)
    if origin is EvidenceValue:
        value_types = get_args(expected_type)
        if len(value_types) != 1:
            raise ValueError(f"unsupported EvidenceValue type at {path}")
        return _decode_evidence_value(value_types[0], payload, path=path)

    if origin is list:
        item_types = get_args(expected_type)
        if len(item_types) != 1:
            raise ValueError(f"unsupported list type at {path}")
        if type(payload) is not list:
            raise ValueError(f"expected list at {path}")
        return [
            _decode_value(item_types[0], item, path=f"{path}[{index}]")
            for index, item in enumerate(payload)
        ]

    if expected_type is EvidenceStatus:
        if type(payload) is not str:
            raise ValueError(f"invalid EvidenceStatus at {path}: {payload!r}")
        try:
            return EvidenceStatus(payload)
        except ValueError as error:
            raise ValueError(
                f"invalid EvidenceStatus at {path}: {payload!r}"
            ) from error

    if expected_type in {str, int, bool}:
        if type(payload) is not expected_type:
            raise ValueError(f"expected {expected_type.__name__} at {path}")
        return payload

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _decode_dataclass(expected_type, payload, path=path)

    raise ValueError(f"unsupported evidence type at {path}: {expected_type!r}")


def _decode_evidence_value(
    value_type: object,
    payload: object,
    *,
    path: str,
) -> EvidenceValue[object]:
    if type(payload) is not dict:
        raise ValueError(f"expected object at {path}")

    missing_base_keys = sorted({"status", "source"} - set(payload))
    if missing_base_keys:
        raise ValueError(f"missing keys at {path}: {missing_base_keys}")

    status = cast(
        EvidenceStatus,
        _decode_value(
            EvidenceStatus,
            payload["status"],
            path=f"{path}.status",
        ),
    )

    if status is EvidenceStatus.KNOWN:
        _require_exact_keys(payload, {"status", "source", "value"}, path)
        source = cast(
            str,
            _decode_value(str, payload["source"], path=f"{path}.source"),
        )
        value = _decode_value(
            value_type,
            payload["value"],
            path=f"{path}.value",
        )
        try:
            return EvidenceValue(status=status, value=value, source=source)
        except ValueError as error:
            raise ValueError(f"invalid EvidenceValue at {path}: {error}") from error

    _require_exact_keys(payload, {"status", "source", "reason"}, path)
    source = cast(
        str,
        _decode_value(str, payload["source"], path=f"{path}.source"),
    )
    reason = cast(
        str,
        _decode_value(str, payload["reason"], path=f"{path}.reason"),
    )
    try:
        return EvidenceValue(status=status, reason=reason, source=source)
    except ValueError as error:
        raise ValueError(f"invalid EvidenceValue at {path}: {error}") from error


def diff_snapshots(
    before: EvidenceSnapshot,
    after: EvidenceSnapshot,
    *,
    ignore_paths: set[str] | None = None,
) -> list[EvidenceDifference]:
    """Return deterministic, path-addressed structural evidence differences."""

    differences: list[EvidenceDifference] = []
    _diff_values(
        before.to_dict(),
        after.to_dict(),
        path="",
        ignored=ignore_paths or set(),
        differences=differences,
    )
    return differences


def _diff_values(
    before: object,
    after: object,
    *,
    path: str,
    ignored: set[str],
    differences: list[EvidenceDifference],
) -> None:
    if path in ignored or any(path.startswith(f"{item}/") for item in ignored):
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}/{key}"
            if key not in before:
                differences.append(EvidenceDifference(child_path, "<missing>", after[key]))
            elif key not in after:
                differences.append(EvidenceDifference(child_path, before[key], "<missing>"))
            else:
                _diff_values(
                    before[key],
                    after[key],
                    path=child_path,
                    ignored=ignored,
                    differences=differences,
                )
        return
    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child_path = f"{path}/{index}"
            if index >= len(before):
                differences.append(EvidenceDifference(child_path, "<missing>", after[index]))
            elif index >= len(after):
                differences.append(EvidenceDifference(child_path, before[index], "<missing>"))
            else:
                _diff_values(
                    before[index],
                    after[index],
                    path=child_path,
                    ignored=ignored,
                    differences=differences,
                )
        return
    if before != after:
        differences.append(EvidenceDifference(path, before, after))
