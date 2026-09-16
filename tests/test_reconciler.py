"""Task 22 public deterministic reconciliation pipeline."""

from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory_os.evidence.models import EvidenceSnapshot
from agent_memory_os.reconcile import reconcile
from agent_memory_os.reconcile.models import ReconciliationInputError
from agent_memory_os.reconcile.serialization import result_to_json
from reconciliation_helpers import (
    CAPTURED_AT,
    default_policy,
    fixed_clock,
    known,
    make_candidate,
    make_snapshot,
    synthetic_five_status_candidate_set,
)


def reconcile_source_files() -> tuple[Path, ...]:
    root = Path("src/agent_memory_os/reconcile")
    return tuple(sorted(root.glob("*.py")))


def test_reconcile_emits_all_five_status_partitions_deterministically() -> None:
    candidates = synthetic_five_status_candidate_set()

    first = reconcile(
        make_snapshot(),
        candidates,
        default_policy(),
        clock=fixed_clock,
    )
    second = reconcile(
        make_snapshot(),
        list(reversed(candidates)),
        default_policy(),
        clock=fixed_clock,
    )

    assert first == second
    assert result_to_json(first) == result_to_json(second)
    assert len(first.active) == 2
    assert len(first.superseded) == 2
    assert len(first.conflicted) == 2
    assert len(first.pending) == 1
    assert len(first.deprecated) == 1
    assert first.schema_version == "reconciliation:v1"
    assert sum(
        (
            first.summary_counts.active,
            first.summary_counts.superseded,
            first.summary_counts.conflicted,
            first.summary_counts.pending,
            first.summary_counts.deprecated,
        )
    ) == 8


def test_reconcile_zero_candidates_returns_valid_empty_result() -> None:
    result = reconcile(
        make_snapshot(),
        [],
        default_policy(),
        clock=fixed_clock,
    )

    assert result.active == ()
    assert result.superseded == ()
    assert result.conflicted == ()
    assert result.pending == ()
    assert result.deprecated == ()
    assert result.relations == ()
    assert result.unresolved == ()
    assert result.warnings == ()
    assert result.unresolved_count == 0
    assert result.human_review_required is False
    assert set(result.summary_counts.to_dict().values()) == {0}


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (object(), "exact EvidenceSnapshot"),
        (
            replace(make_snapshot(), schema_version=known("2.0.0")),
            "schema_version",
        ),
        (
            replace(make_snapshot(), project_id=known("")),
            "project_id",
        ),
        (
            replace(make_snapshot(), captured_at=known("2026-09-14")),
            "captured_at",
        ),
    ],
)
def test_reconcile_rejects_invalid_snapshot_before_pipeline(
    snapshot: object,
    message: str,
) -> None:
    with pytest.raises(ReconciliationInputError, match=message):
        reconcile(snapshot, [], default_policy(), clock=fixed_clock)


def test_reconcile_rejects_duplicate_ids_before_candidate_timestamp_parsing() -> None:
    candidates = [
        make_candidate(candidate_id="duplicate", observed_at="not-a-time"),
        make_candidate(candidate_id="duplicate", predicate="other"),
    ]

    with pytest.raises(
        ReconciliationInputError,
        match="duplicate candidate_id: duplicate",
    ):
        reconcile(make_snapshot(), candidates, default_policy(), clock=fixed_clock)


def test_reconcile_rejects_static_input_before_calling_clock() -> None:
    calls = 0

    def clock() -> str:
        nonlocal calls
        calls += 1
        return CAPTURED_AT

    invalid_snapshot = replace(make_snapshot(), schema_version=known("2.0.0"))
    with pytest.raises(ReconciliationInputError, match="schema_version"):
        reconcile(invalid_snapshot, [], default_policy(), clock=clock)

    duplicates = [
        make_candidate(candidate_id="duplicate"),
        make_candidate(candidate_id="duplicate"),
    ]
    with pytest.raises(ReconciliationInputError, match="duplicate candidate_id"):
        reconcile(make_snapshot(), duplicates, default_policy(), clock=clock)

    assert calls == 0


def test_reconcile_rejects_malformed_candidate_without_writing_artifact(
    tmp_path: Path,
) -> None:
    candidate = make_candidate(observed_at="not-a-time")

    with pytest.raises(ReconciliationInputError, match="observed_at"):
        reconcile(make_snapshot(), [candidate], default_policy(), clock=fixed_clock)

    assert tuple(tmp_path.iterdir()) == ()


def test_reconcile_copies_candidate_collection_without_mutating_it() -> None:
    candidates = synthetic_five_status_candidate_set()
    original_ids = [candidate.candidate_id for candidate in candidates]

    reconcile(make_snapshot(), candidates, default_policy(), clock=fixed_clock)

    assert [candidate.candidate_id for candidate in candidates] == original_ids


def test_reconcile_normalizes_clock_to_utc_once() -> None:
    calls = 0

    def clock() -> str:
        nonlocal calls
        calls += 1
        return "2026-09-14T08:00:00+08:00"

    result = reconcile(make_snapshot(), [], default_policy(), clock=clock)

    assert calls == 1
    assert result.reconciled_at == CAPTURED_AT


def test_reconcile_core_imports_no_external_io_modules() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in reconcile_source_files()
    ).lower()
    for forbidden in (
        "import subprocess",
        "import socket",
        "import requests",
        "import openai",
        "import mcp",
        "import git",
    ):
        assert forbidden not in source

    orchestration = Path(
        "src/agent_memory_os/reconcile/reconciler.py"
    ).read_text(encoding="utf-8").lower()
    assert "from pathlib import" not in orchestration
    assert "open(" not in orchestration
