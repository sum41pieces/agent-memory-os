"""Task 20 result construction, deterministic serialization, and safe writes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
import hashlib
import json
import os
from pathlib import Path

import pytest

from agent_memory_os.evidence.models import (
    EvidenceStatus,
    snapshot_to_json,
)
import agent_memory_os.reconcile.models as reconciliation_models
import agent_memory_os.reconcile.serialization as reconciliation_serialization
from agent_memory_os.reconcile.models import (
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationWarning,
    SourceType,
    SummaryCounts,
    UnresolvedCandidate,
    WarningCode,
)
from agent_memory_os.reconcile.reconciler import materialize_relation_graph
from agent_memory_os.reconcile.rules import (
    classify_group,
    group_candidates,
    resolve_non_known_groups,
    resolve_predicate,
)
from agent_memory_os.reconcile.serialization import (
    _is_forbidden_project_path,
    make_snapshot_id,
    result_to_json,
    write_reconciliation_result,
)

from reconciliation_helpers import (
    NOW,
    complete_result,
    default_policy,
    make_candidate,
    make_current_candidate,
    make_snapshot,
    provisional_fact,
    unavailable,
    unknown,
    _test_only_graph,
)


PROJECT_ID = "synthetic-project"
SNAPSHOT_ID = "snapshot:v1:synthetic-task20"
RECONCILED_AT = NOW.isoformat()
FORBIDDEN_PROJECT_ROOTS = (
    r"C:\Users\demo\projects\interview-agent-finals",
    r"C:\Users\demo\projects\interview-agent-v1",
)


class _SequenceDouble(Sequence):
    def __init__(self, values: list[object] | tuple[object, ...]) -> None:
        self._values = tuple(values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> object:
        return self._values[index]


class _EqualityProbe:
    def __init__(self) -> None:
        self.calls = 0

    def __eq__(self, other: object) -> bool:
        self.calls += 1
        raise AssertionError("schema comparison must not execute")


def _active_graph(
    *,
    reverse_inputs: bool = False,
    mutable_value: dict[str, object] | None = None,
):
    candidates = [
        make_current_candidate(
            candidate_id="candidate-z",
            predicate="zeta-path",
            value=(
                {"path": r"D:\模拟\项目", "nested": ["原始"]}
                if mutable_value is None
                else mutable_value
            ),
            source_ref="synthetic:中文路径",
        ),
        make_current_candidate(
            candidate_id="candidate-a",
            predicate="alpha-port",
            value=8000,
            source_ref="synthetic:port",
        ),
    ]
    if reverse_inputs:
        candidates.reverse()
    groups = group_candidates(
        PROJECT_ID,
        candidates,
        snapshot_id=SNAPSHOT_ID,
    )
    decisions = tuple(
        classify_group(group, NOW, default_policy()) for group in groups
    )
    return materialize_relation_graph(decisions, ()), decisions


def _conflict_graph(*, reverse_inputs: bool = False):
    candidates = [
        make_current_candidate(
            candidate_id="candidate-c",
            predicate="runtime-port",
            value=8000,
        ),
        make_current_candidate(
            candidate_id="candidate-b",
            predicate="runtime-port",
            value=8010,
        ),
        make_current_candidate(
            candidate_id="candidate-a",
            predicate="runtime-port",
            value=8020,
        ),
    ]
    if reverse_inputs:
        candidates.reverse()
    decisions = tuple(
        classify_group(group, NOW, default_policy())
        for group in group_candidates(
            PROJECT_ID,
            candidates,
            snapshot_id=SNAPSHOT_ID,
        )
    )
    outcome = resolve_predicate(decisions, default_policy())
    return (
        materialize_relation_graph(
            outcome.decisions,
            outcome.relation_requests,
        ),
        outcome.decisions,
    )


def _graph_with_unresolved(
    *,
    conflict: bool,
    reverse_inputs: bool = False,
    mutable_value: dict[str, object] | None = None,
):
    candidates = (
        [
            make_current_candidate(
                candidate_id="candidate-c",
                predicate="runtime-port",
                value=8000,
            ),
            make_current_candidate(
                candidate_id="candidate-b",
                predicate="runtime-port",
                value=8010,
            ),
        ]
        if conflict
        else [
            make_current_candidate(
                candidate_id="candidate-z",
                predicate="zeta-path",
                value=(
                    {"path": r"D:\模拟\项目", "nested": ["原始"]}
                    if mutable_value is None
                    else mutable_value
                ),
            ),
            make_current_candidate(
                candidate_id="candidate-a",
                predicate="alpha-port",
                value=8000,
            ),
        ]
    )
    candidates.extend(
        (
            make_candidate(
                candidate_id="unresolved-z",
                predicate="missing-z",
                value=unknown(
                    "value was not recorded",
                    source="unresolved-z:value",
                ),
            ),
            make_candidate(
                candidate_id="unresolved-a",
                predicate="missing-a",
                value=unavailable(
                    "value source failed",
                    source="unresolved-a:value",
                ),
            ),
        )
    )
    if reverse_inputs:
        candidates.reverse()
    groups = group_candidates(
        PROJECT_ID,
        candidates,
        snapshot_id=SNAPSHOT_ID,
    )
    policy = default_policy()
    classified = tuple(classify_group(group, NOW, policy) for group in groups)
    by_predicate: dict[str, list] = {}
    for decision in classified:
        by_predicate.setdefault(decision.predicate, []).append(decision)
    resolved = []
    requests = []
    for predicate in sorted(by_predicate):
        outcome = resolve_predicate(by_predicate[predicate], policy)
        resolved.extend(outcome.decisions)
        requests.extend(outcome.relation_requests)
    _, unresolved, _ = resolve_non_known_groups(
        tuple(
            group
            for group in groups
            if group.candidates[0].value.status is not EvidenceStatus.KNOWN
        ),
        NOW,
        policy,
    )
    decisions = tuple(sorted(resolved, key=lambda item: item.fact_id))
    return materialize_relation_graph(decisions, requests), decisions, unresolved


def _warning(code: WarningCode, candidate_id: str) -> ReconciliationWarning:
    return ReconciliationWarning(
        code=code,
        message=f"synthetic warning for {candidate_id}",
        candidate_ids=(candidate_id,),
        evidence_refs=(f"synthetic:{candidate_id}",),
        requires_human_review=True,
    )


def _unresolved(candidate_id: str, related_fact_id: str) -> UnresolvedCandidate:
    return UnresolvedCandidate(
        candidate_id=candidate_id,
        subject="project",
        predicate="missing-setting",
        evidence_status=EvidenceStatus.UNKNOWN,
        reason="synthetic missing evidence",
        source_type=SourceType.CURRENT_EVIDENCE,
        source_ref=f"synthetic:{candidate_id}",
        field_source=f"synthetic:{candidate_id}:value",
        related_fact_id=related_fact_id,
    )


def _result(
    *,
    reverse_inputs: bool = False,
    warnings: list[ReconciliationWarning] | None = None,
    unresolved: list[UnresolvedCandidate] | None = None,
) -> ReconciliationResult:
    graph, decisions = _active_graph(reverse_inputs=reverse_inputs)
    return ReconciliationResult.create(
        project_id=PROJECT_ID,
        snapshot_id=SNAPSHOT_ID,
        reconciled_at=RECONCILED_AT,
        graph=graph,
        source_decisions=list(decisions),
        warnings=[] if warnings is None else warnings,
        unresolved=[] if unresolved is None else unresolved,
    )


def _corrupt_result(
    result: ReconciliationResult,
    **changes: object,
) -> ReconciliationResult:
    corrupt = object.__new__(ReconciliationResult)
    for model_field in fields(ReconciliationResult):
        value = changes.get(
            model_field.name,
            object.__getattribute__(result, model_field.name),
        )
        object.__setattr__(corrupt, model_field.name, value)
    for private_name in ("_graph", "_source_decisions"):
        object.__setattr__(
            corrupt,
            private_name,
            object.__getattribute__(result, private_name),
        )
    return corrupt


def test_result_models_have_exact_frozen_fields_and_no_public_truth_inputs() -> None:
    assert tuple(field.name for field in fields(SummaryCounts)) == (
        "active",
        "superseded",
        "conflicted",
        "pending",
        "deprecated",
        "relations",
        "warnings",
        "unresolved",
    )
    assert tuple(field.name for field in fields(ReconciliationResult)) == (
        "schema_version",
        "project_id",
        "snapshot_id",
        "reconciled_at",
        "active",
        "superseded",
        "conflicted",
        "pending",
        "deprecated",
        "relations",
        "unresolved",
        "warnings",
        "unresolved_count",
        "human_review_required",
        "summary_counts",
    )

    result = _result()
    assert result.schema_version == "reconciliation:v1"
    assert result.unresolved_count == 0
    assert result.summary_counts.active == 2
    assert result.summary_counts.relations == 0
    with pytest.raises(FrozenInstanceError):
        result.project_id = "forged"
    with pytest.raises(TypeError):
        SummaryCounts(
            active=99,
            superseded=0,
            conflicted=0,
            pending=0,
            deprecated=0,
            relations=0,
            warnings=0,
            unresolved=0,
        )
    with pytest.raises(TypeError):
        ReconciliationResult(
            schema_version="reconciliation:v1",
            project_id=PROJECT_ID,
            snapshot_id=SNAPSHOT_ID,
            reconciled_at=RECONCILED_AT,
            active=(),
            superseded=(),
            conflicted=(),
            pending=(),
            deprecated=(),
            relations=(),
            unresolved=(),
            warnings=(),
            unresolved_count=0,
            human_review_required=False,
            summary_counts=object(),
        )


def test_replacing_module_construction_tokens_cannot_forge_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    replacement_result_token = object()
    replacement_counts_token = object()
    monkeypatch.setattr(
        reconciliation_models,
        "_RECONCILIATION_RESULT_CONSTRUCTION_TOKEN",
        replacement_result_token,
    )
    monkeypatch.setattr(
        reconciliation_models,
        "_SUMMARY_COUNTS_CONSTRUCTION_TOKEN",
        replacement_counts_token,
    )

    with pytest.raises(TypeError):
        ReconciliationResult._from_validated_fields(
            replacement_result_token,
            graph=object.__getattribute__(result, "_graph"),
            source_decisions=object.__getattribute__(
                result,
                "_source_decisions",
            ),
            **{
                model_field.name: object.__getattribute__(
                    result,
                    model_field.name,
                )
                for model_field in fields(ReconciliationResult)
            },
        )
    with pytest.raises(TypeError):
        SummaryCounts._derive(
            replacement_counts_token,
            active=result.active,
            superseded=result.superseded,
            conflicted=result.conflicted,
            pending=result.pending,
            deprecated=result.deprecated,
            relations=result.relations,
            warnings=result.warnings,
            unresolved=result.unresolved,
        )


def test_fixed_clock_result_json_is_byte_identical_and_keeps_unicode() -> None:
    first = complete_result(clock_value=RECONCILED_AT)
    second = complete_result(
        clock_value=RECONCILED_AT,
        reverse_inputs=True,
    )

    first_json = result_to_json(first)
    assert first_json == result_to_json(second)
    assert "模拟" in first_json
    selected_paths = [
        fact["selected_value"]["value"]
        for fact in json.loads(first_json)["active"]
        if fact["predicate"] == "project-path"
    ]
    assert selected_paths == [r"D:\模拟\项目"]
    assert "\\u6a21" not in first_json
    assert first_json.endswith("\n")
    assert result_to_json(first) == first_json


def test_relation_warning_and_unresolved_input_order_is_canonical() -> None:
    graph, decisions, unresolved = _graph_with_unresolved(conflict=True)
    warnings = [
        _warning(WarningCode.INVALID_TEMPORAL_ORDER, "candidate-z"),
        _warning(WarningCode.EVIDENCE_UNKNOWN, "candidate-a"),
    ]
    first = ReconciliationResult.create(
        project_id=PROJECT_ID,
        snapshot_id=SNAPSHOT_ID,
        reconciled_at=RECONCILED_AT,
        graph=graph,
        source_decisions=list(decisions),
        warnings=warnings,
        unresolved=unresolved,
    )
    graph_reversed, decisions_reversed, unresolved_reversed = (
        _graph_with_unresolved(conflict=True, reverse_inputs=True)
    )
    second = ReconciliationResult.create(
        project_id=PROJECT_ID,
        snapshot_id=SNAPSHOT_ID,
        reconciled_at=RECONCILED_AT,
        graph=graph_reversed,
        source_decisions=list(reversed(decisions_reversed)),
        warnings=list(reversed(warnings)),
        unresolved=list(reversed(unresolved_reversed)),
    )

    assert result_to_json(first) == result_to_json(second)
    assert tuple(warning.code.value for warning in first.warnings) == (
        "EVIDENCE_UNAVAILABLE",
        "EVIDENCE_UNKNOWN",
        "EVIDENCE_UNKNOWN",
        "INVALID_TEMPORAL_ORDER",
    )
    assert tuple(item.candidate_id for item in first.unresolved) == (
        "unresolved-a",
        "unresolved-z",
    )
    relation_keys = tuple(
        (
            relation.relation_type.value,
            relation.from_fact_id,
            relation.to_fact_id,
            relation.relation_id,
        )
        for relation in first.relations
    )
    assert relation_keys == tuple(sorted(relation_keys))


def test_result_isolated_from_caller_lists_and_nested_mutable_values() -> None:
    mutable_value: dict[str, object] = {
        "path": r"D:\模拟\项目",
        "nested": ["before"],
    }
    graph, decisions, canonical_unresolved = _graph_with_unresolved(
        conflict=False,
        mutable_value=mutable_value,
    )
    warning_list = [_warning(WarningCode.EVIDENCE_UNKNOWN, "candidate-z")]
    unresolved_list = list(canonical_unresolved)
    decision_list = list(decisions)
    result = ReconciliationResult.create(
        project_id=PROJECT_ID,
        snapshot_id=SNAPSHOT_ID,
        reconciled_at=RECONCILED_AT,
        graph=graph,
        source_decisions=decision_list,
        warnings=warning_list,
        unresolved=unresolved_list,
    )
    before = result_to_json(result)

    mutable_value["path"] = "changed"
    mutable_value["nested"].append("after")
    warning_list.clear()
    unresolved_list.clear()
    decision_list.clear()

    assert result_to_json(result) == before


@pytest.mark.parametrize(
    "field_name",
    ("source_decisions", "warnings", "unresolved"),
)
def test_result_factory_rejects_non_builtin_sequence_collections(
    field_name: str,
) -> None:
    graph, decisions = _active_graph()
    warning = _warning(WarningCode.EVIDENCE_UNKNOWN, "candidate-z")
    unresolved = _unresolved("unresolved-z", graph.facts[0].fact_id)
    inputs: dict[str, object] = {
        "source_decisions": list(decisions),
        "warnings": [warning],
        "unresolved": [unresolved],
    }
    inputs[field_name] = _SequenceDouble(inputs[field_name])

    with pytest.raises(
        ReconciliationInputError,
        match=rf"{field_name}.*exact list or tuple",
    ):
        ReconciliationResult.create(
            project_id=PROJECT_ID,
            snapshot_id=SNAPSHOT_ID,
            reconciled_at=RECONCILED_AT,
            graph=graph,
            source_decisions=inputs["source_decisions"],
            warnings=inputs["warnings"],
            unresolved=inputs["unresolved"],
        )


def test_to_dict_contains_only_json_primitives() -> None:
    result = _result()
    payload = result.to_dict()

    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert type(payload) is dict
    assert all(type(item) is dict for item in payload["active"])
    assert type(payload["active"][0]["candidate_ids"]) is list
    assert "reason" not in payload["active"][0]["selected_value"]
    assert type(payload["active"][1]["selected_value"]["value"]) is dict
    assert type(
        payload["active"][1]["selected_value"]["value"]["nested"]
    ) is list


def test_snapshot_id_is_hash_of_exact_canonical_snapshot_json() -> None:
    snapshot = make_snapshot()
    expected_digest = hashlib.sha256(
        snapshot_to_json(snapshot).encode("utf-8")
    ).hexdigest()
    first = make_snapshot_id(snapshot)
    second = make_snapshot_id(snapshot)

    assert first == second == f"snapshot:v1:{expected_digest}"


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("project_id", "other-project"),
        ("snapshot_id", "snapshot:v1:other"),
        ("reconciled_at", "2026-09-14T00:00:01+00:00"),
    ),
)
def test_result_factory_binds_identity_to_authenticated_cohort(
    field_name: str,
    invalid: str,
) -> None:
    graph, decisions = _active_graph()
    identity = {
        "project_id": PROJECT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "reconciled_at": RECONCILED_AT,
    }
    identity[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        ReconciliationResult.create(
            **identity,
            graph=graph,
            source_decisions=decisions,
        )


def test_factory_rejects_wrong_stage_graph_and_source_decisions() -> None:
    graph, decisions = _active_graph()
    wrong_stage_graph = _test_only_graph(
        (provisional_fact("wrong-stage", ReconciliationStatus.ACTIVE),),
        (),
    )
    with pytest.raises(ReconciliationInvariantError):
        ReconciliationResult.create(
            project_id=PROJECT_ID,
            snapshot_id=SNAPSHOT_ID,
            reconciled_at=RECONCILED_AT,
            graph=wrong_stage_graph,
            source_decisions=decisions,
        )

    with pytest.raises(ReconciliationInvariantError, match="source_decisions"):
        ReconciliationResult.create(
            project_id=PROJECT_ID,
            snapshot_id=SNAPSHOT_ID,
            reconciled_at=RECONCILED_AT,
            graph=graph,
            source_decisions=(),
        )


@pytest.mark.parametrize("corruption", ("counts", "review", "partitions"))
def test_serializer_revalidates_and_rejects_corrupt_derived_fields(
    corruption: str,
) -> None:
    warning = _warning(WarningCode.EVIDENCE_UNKNOWN, "candidate-z")
    result = _result(warnings=[warning])
    if corruption == "counts":
        fake_counts = object.__new__(SummaryCounts)
        for name, value in result.summary_counts.to_dict().items():
            object.__setattr__(
                fake_counts,
                name,
                value + (1 if name == "active" else 0),
            )
        corrupt = _corrupt_result(result, summary_counts=fake_counts)
    elif corruption == "review":
        corrupt = _corrupt_result(result, human_review_required=False)
    else:
        corrupt = _corrupt_result(result, active=())

    with pytest.raises(ReconciliationInvariantError):
        result_to_json(corrupt)


def test_serializer_rejects_wrong_result_type() -> None:
    with pytest.raises(ReconciliationInputError, match="exact ReconciliationResult"):
        result_to_json(object())


def test_schema_version_rejects_non_string_before_equality_dispatch() -> None:
    result = _result()
    probe = _EqualityProbe()
    corrupt = _corrupt_result(result, schema_version=probe)

    with pytest.raises(ReconciliationInvariantError, match="schema_version"):
        result_to_json(corrupt)
    assert probe.calls == 0


@pytest.mark.parametrize(
    ("field_name", "record_type"),
    (
        ("warnings", ReconciliationWarning),
        ("unresolved", UnresolvedCandidate),
    ),
)
def test_serializer_reports_stable_error_for_incomplete_exact_records(
    field_name: str,
    record_type: type,
) -> None:
    result = _result()
    incomplete = object.__new__(record_type)
    corrupt = _corrupt_result(result, **{field_name: (incomplete,)})

    with pytest.raises(ReconciliationInvariantError, match=field_name):
        result_to_json(corrupt)


def test_writer_requires_product_artifacts_containment(tmp_path: Path) -> None:
    result = _result()
    product_root = tmp_path / "product"
    outside = tmp_path / "outside.json"

    with pytest.raises(ReconciliationInputError, match="artifacts"):
        write_reconciliation_result(result, outside, product_root)
    assert not outside.exists()

    artifacts = product_root / "artifacts"
    artifacts.mkdir(parents=True)
    with pytest.raises(ReconciliationInputError, match="file"):
        write_reconciliation_result(result, artifacts, product_root)


@pytest.mark.parametrize(
    ("output", "product_root"),
    (
        (
            FORBIDDEN_PROJECT_ROOTS[0] + r"\artifacts\result.json",
            FORBIDDEN_PROJECT_ROOTS[0],
        ),
        (
            FORBIDDEN_PROJECT_ROOTS[0].upper()
            + r"\CHILD\artifacts\result.json",
            r"D:\safe-product",
        ),
        (
            Path(FORBIDDEN_PROJECT_ROOTS[1]),
            Path(FORBIDDEN_PROJECT_ROOTS[1]),
        ),
        (
            Path(FORBIDDEN_PROJECT_ROOTS[1].swapcase())
            / "child"
            / "result.json",
            Path(r"D:\safe-product"),
        ),
    ),
)
def test_writer_lexically_rejects_forbidden_roots_before_filesystem_access(
    output: str | Path,
    product_root: str | Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    touched: list[str] = []

    def unexpected_access(*args: object, **kwargs: object) -> object:
        touched.append("filesystem")
        raise AssertionError("forbidden path reached filesystem access")

    for owner, attribute in (
        (Path, "resolve"),
        (Path, "stat"),
        (Path, "exists"),
        (Path, "is_dir"),
        (Path, "mkdir"),
        (Path, "open"),
        (Path, "read_text"),
        (Path, "read_bytes"),
        (Path, "write_text"),
        (Path, "write_bytes"),
        (reconciliation_serialization.tempfile, "mkstemp"),
        (os, "fdopen"),
        (os, "fsync"),
        (os, "replace"),
    ):
        monkeypatch.setattr(owner, attribute, unexpected_access)

    with pytest.raises(ReconciliationInputError, match="forbidden project root"):
        write_reconciliation_result(result, output, product_root)
    monkeypatch.undo()
    assert touched == []


def test_writer_rechecks_resolved_paths_before_any_downstream_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    forbidden_root = FORBIDDEN_PROJECT_ROOTS[0]
    resolve_calls: list[str] = []
    downstream_access: list[str] = []

    def redirected_resolve(path: Path, *, strict: bool = False) -> Path:
        resolve_calls.append(str(path))
        if len(resolve_calls) == 1:
            return Path(forbidden_root)
        if len(resolve_calls) == 2:
            return Path(forbidden_root) / "artifacts" / "result.json"
        downstream_access.append("additional-resolve")
        raise AssertionError("resolved forbidden path was used again")

    def unexpected_access(*args: object, **kwargs: object) -> object:
        downstream_access.append("filesystem")
        raise AssertionError("resolved forbidden path reached filesystem access")

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    for owner, attribute in (
        (Path, "stat"),
        (Path, "exists"),
        (Path, "is_dir"),
        (Path, "mkdir"),
        (Path, "open"),
        (Path, "read_text"),
        (Path, "read_bytes"),
        (Path, "write_text"),
        (Path, "write_bytes"),
        (reconciliation_serialization.tempfile, "mkstemp"),
        (os, "fdopen"),
        (os, "fsync"),
        (os, "replace"),
    ):
        monkeypatch.setattr(owner, attribute, unexpected_access)

    caught: BaseException | None = None
    try:
        write_reconciliation_result(
            result,
            r"D:\safe-product\artifacts\result.json",
            r"D:\safe-product",
        )
    except BaseException as error:
        caught = error
    finally:
        monkeypatch.undo()
    assert isinstance(caught, ReconciliationInputError)
    assert "forbidden project root" in str(caught)
    assert len(resolve_calls) == 2
    assert downstream_access == []


@pytest.mark.parametrize(
    "nearby_path",
    (
        FORBIDDEN_PROJECT_ROOTS[0] + "-backup",
        FORBIDDEN_PROJECT_ROOTS[1] + "2",
    ),
)
def test_forbidden_root_lexical_check_respects_component_boundaries(
    nearby_path: str,
) -> None:
    assert _is_forbidden_project_path(nearby_path) is False


@pytest.mark.parametrize("reserved", ("Shadow", "Vault", "runtime"))
def test_writer_rejects_reserved_destination_components(
    tmp_path: Path,
    reserved: str,
) -> None:
    result = _result()
    product_root = tmp_path / reserved / "product"
    output = product_root / "artifacts" / "result.json"

    with pytest.raises(ReconciliationInputError, match="reserved"):
        write_reconciliation_result(result, output, product_root)
    assert not output.exists()


def test_writer_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    result = _result()
    product_root = tmp_path / "product"
    artifacts = product_root / "artifacts"
    outside = tmp_path / "outside"
    artifacts.mkdir(parents=True)
    outside.mkdir()
    link = artifacts / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    escaped = link / "result.json"
    with pytest.raises(ReconciliationInputError, match="artifacts"):
        write_reconciliation_result(result, escaped, product_root)
    assert not (outside / "result.json").exists()


def test_writer_uses_sibling_atomic_replace_and_exact_utf8_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    product_root = tmp_path / "product"
    output = product_root / "artifacts" / "nested" / "结果.json"
    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def checked_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.exists()
        assert not destination_path.exists()
        calls.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(os, "replace", checked_replace)
    write_reconciliation_result(result, output, product_root)

    assert len(calls) == 1
    assert calls[0][1] == output.resolve()
    assert output.read_bytes() == result_to_json(result).encode("utf-8")
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_writer_replace_failure_keeps_existing_file_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    product_root = tmp_path / "product"
    output = product_root / "artifacts" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"previous")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        write_reconciliation_result(result, output, product_root)

    assert output.read_bytes() == b"previous"
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_writer_rejects_corrupt_result_before_creating_output_or_temp(
    tmp_path: Path,
) -> None:
    result = _result()
    corrupt = _corrupt_result(result, unresolved_count=1)
    product_root = tmp_path / "product"
    output = product_root / "artifacts" / "result.json"

    with pytest.raises(ReconciliationInvariantError):
        write_reconciliation_result(corrupt, output, product_root)

    assert not output.exists()
    assert not product_root.exists()
