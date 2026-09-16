"""Deterministic Task 20 serialization and product-local result writing."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import ntpath
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

from agent_memory_os.evidence.models import (
    EvidenceSnapshot,
    EvidenceValue,
    snapshot_to_json,
)
from agent_memory_os.reconcile.models import (
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationResult,
)


_RESERVED_DESTINATION_COMPONENTS = frozenset({"shadow", "vault", "runtime"})
_FORBIDDEN_PROJECT_ROOTS = (
    r"C:\Users\demo\projects\interview-agent-finals",
    r"C:\Users\demo\projects\interview-agent-v1",
)


def _path_text_without_filesystem(value: object, field_name: str) -> str:
    if type(value) is str:
        text = value
    elif isinstance(value, Path):
        text = str(value)
    else:
        raise ReconciliationInputError(
            f"{field_name} must be an exact string or Path"
        )
    if not text or "\x00" in text:
        raise ReconciliationInputError(
            f"{field_name} must be a non-empty NUL-free path"
        )
    return text


def _windows_lexical_key(value: object) -> tuple[str, tuple[str, ...]]:
    text = _path_text_without_filesystem(value, "path").replace("/", "\\")
    if text.casefold().startswith("\\\\?\\"):
        text = text[4:]
    normalized = ntpath.normpath(text)
    drive, tail = ntpath.splitdrive(normalized)
    components = tuple(
        component.rstrip(" .").casefold()
        for component in tail.split("\\")
        if component not in ("", ".")
    )
    return drive.rstrip(" .").casefold(), components


_FORBIDDEN_PROJECT_KEYS = tuple(
    _windows_lexical_key(root) for root in _FORBIDDEN_PROJECT_ROOTS
)


def _is_forbidden_project_path(value: str | Path) -> bool:
    """Purely lexically detect either prohibited root or any descendant."""

    candidate_drive, candidate_parts = _windows_lexical_key(value)
    return any(
        candidate_drive == root_drive
        and len(candidate_parts) >= len(root_parts)
        and candidate_parts[: len(root_parts)] == root_parts
        for root_drive, root_parts in _FORBIDDEN_PROJECT_KEYS
    )


def _reject_forbidden_project_paths(
    output: object,
    product_root: object,
) -> None:
    for field_name, value in (
        ("output", output),
        ("product_root", product_root),
    ):
        _path_text_without_filesystem(value, field_name)
        if _is_forbidden_project_path(value):
            raise ReconciliationInputError(
                f"{field_name} targets a forbidden project root"
            )


def make_snapshot_id(snapshot: EvidenceSnapshot) -> str:
    """Hash the exact canonical Phase 1 snapshot serialization."""

    if type(snapshot) is not EvidenceSnapshot:
        raise ReconciliationInputError(
            "snapshot must be an exact EvidenceSnapshot"
        )
    digest = hashlib.sha256(
        snapshot_to_json(snapshot).encode("utf-8")
    ).hexdigest()
    return f"snapshot:v1:{digest}"


def _to_primitive(value: object) -> object:
    """Recursively copy supported immutable model values to JSON primitives."""

    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Enum):
        return value.value
    if type(value) is EvidenceValue:
        return _to_primitive(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            model_field.name: _to_primitive(
                object.__getattribute__(value, model_field.name)
            )
            for model_field in fields(value)
        }
    if isinstance(value, Mapping):
        try:
            items = tuple(value.items())
        except Exception as error:
            raise ReconciliationInvariantError(
                "result mapping could not be inspected safely"
            ) from error
        converted: dict[str, object] = {}
        for item in items:
            if type(item) is not tuple or len(item) != 2:
                raise ReconciliationInvariantError(
                    "result mappings must contain exact key/value pairs"
                )
            key, child = item
            if type(key) is not str:
                raise ReconciliationInvariantError(
                    "result mapping keys must be exact strings"
                )
            converted[key] = _to_primitive(child)
        return converted
    if type(value) in (tuple, list):
        return [_to_primitive(child) for child in value]
    raise ReconciliationInvariantError(
        f"unsupported result serialization type: {type(value).__name__}"
    )


def _result_to_primitive_dict(
    result: ReconciliationResult,
) -> dict[str, object]:
    """Convert only the exact public result fields, never private authority."""

    converted = {
        model_field.name: _to_primitive(
            object.__getattribute__(result, model_field.name)
        )
        for model_field in fields(ReconciliationResult)
    }
    if type(converted) is not dict:
        raise ReconciliationInvariantError(
            "result must serialize to a primitive dictionary"
        )
    return converted


def result_to_json(result: ReconciliationResult) -> str:
    """Fully revalidate and serialize one exact result deterministically."""

    if type(result) is not ReconciliationResult:
        raise ReconciliationInputError(
            "result must be an exact ReconciliationResult"
        )
    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _resolved_path(value: object, field_name: str) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise ReconciliationInputError(
            f"{field_name} must be an exact string or Path"
        )
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReconciliationInputError(
            f"{field_name} could not be resolved safely"
        ) from error


def _contains_reserved_component(path: Path) -> bool:
    return any(
        part.casefold() in _RESERVED_DESTINATION_COMPONENTS
        for part in path.parts
    )


def _validate_output_path(output: object, product_root: object) -> Path:
    resolved_root = _resolved_path(product_root, "product_root")
    resolved_output = _resolved_path(output, "output")
    if _is_forbidden_project_path(resolved_root) or (
        _is_forbidden_project_path(resolved_output)
    ):
        raise ReconciliationInputError(
            "resolved output targets a forbidden project root"
        )
    resolved_artifacts = (resolved_root / "artifacts").resolve(strict=False)
    if _contains_reserved_component(resolved_root) or _contains_reserved_component(
        resolved_output
    ):
        raise ReconciliationInputError(
            "output path contains a reserved Shadow/Vault/runtime component"
        )
    if resolved_output == resolved_artifacts:
        raise ReconciliationInputError("output must name a file below artifacts")
    try:
        contained = resolved_output.is_relative_to(resolved_artifacts)
    except (OSError, ValueError) as error:
        raise ReconciliationInputError(
            "output containment could not be checked safely"
        ) from error
    if not contained:
        raise ReconciliationInputError(
            "output must remain below product_root/artifacts"
        )
    try:
        if resolved_output.exists() and resolved_output.is_dir():
            raise ReconciliationInputError("output must name a file")
    except OSError as error:
        raise ReconciliationInputError(
            "output file type could not be checked safely"
        ) from error
    return resolved_output


def write_reconciliation_result(
    result: ReconciliationResult,
    output: str | Path,
    product_root: str | Path,
) -> None:
    """Atomically write validated UTF-8 bytes inside product artifacts only."""

    _reject_forbidden_project_paths(output, product_root)
    encoded = result_to_json(result).encode("utf-8")
    resolved_output = _validate_output_path(output, product_root)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve again after directory creation so a pre-existing symlink or
    # junction cannot become an unchecked destination through missing parents.
    checked_output = _validate_output_path(resolved_output, product_root)
    if checked_output != resolved_output:
        raise ReconciliationInputError(
            "output changed while validating its destination"
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
            dir=resolved_output.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        final_output = _validate_output_path(resolved_output, product_root)
        if final_output != resolved_output:
            raise ReconciliationInputError(
                "output changed before atomic replacement"
            )
        os.replace(temporary_path, resolved_output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "make_snapshot_id",
    "result_to_json",
    "write_reconciliation_result",
]
