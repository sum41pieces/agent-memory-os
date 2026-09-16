from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from agent_memory_os.evidence.models import (
    ChangedFileRecord,
    CollectorEvidence,
    CommitRecord,
    DiscoveredCommand,
    DocsEvidence,
    EvidenceSnapshot,
    EvidenceStatus,
    EvidenceValue,
    GitEvidence,
    RecentChangesEvidence,
    RecordedTestResult,
    RepositoryEvidence,
    ShadowGuardReport,
    TestsEvidence as SnapshotTestsEvidence,
    diff_snapshots,
    snapshot_from_json,
    snapshot_to_json,
    utc_now_iso,
)


def test_evidence_status_has_explicit_wire_values() -> None:
    assert EvidenceStatus.KNOWN.value == "known"
    assert EvidenceStatus.UNKNOWN.value == "unknown"
    assert EvidenceStatus.UNAVAILABLE.value == "unavailable"


def test_known_requires_non_null_value_and_source() -> None:
    with pytest.raises(ValueError, match="KNOWN evidence requires a value"):
        EvidenceValue(status=EvidenceStatus.KNOWN, value=None, source="git:HEAD")
    with pytest.raises(ValueError, match="source must be non-empty"):
        EvidenceValue(status=EvidenceStatus.KNOWN, value="abc", source="")


def test_known_rejects_reason() -> None:
    with pytest.raises(ValueError, match="KNOWN evidence cannot have a reason"):
        EvidenceValue(
            status=EvidenceStatus.KNOWN,
            value="abc",
            reason="ambiguous",
            source="git:HEAD",
        )


@pytest.mark.parametrize(
    "status",
    [EvidenceStatus.UNKNOWN, EvidenceStatus.UNAVAILABLE],
)
def test_non_known_forbids_value_and_requires_reason(status: EvidenceStatus) -> None:
    with pytest.raises(ValueError, match="non-known evidence forbids value"):
        EvidenceValue(status=status, value="guess", reason="weak", source="docs:STATE.md")
    with pytest.raises(ValueError, match="non-known evidence requires a reason"):
        EvidenceValue(status=status, value=None, source="docs:STATE.md")


def test_non_known_json_omits_value_instead_of_using_null() -> None:
    item = EvidenceValue.unknown(
        reason="insufficient evidence",
        source="docs:STATE.md",
    )
    assert item.to_dict() == {
        "reason": "insufficient evidence",
        "source": "docs:STATE.md",
        "status": "unknown",
    }
    assert "value" not in item.to_dict()


def test_empty_collection_and_zero_are_known_values() -> None:
    assert EvidenceValue.known([], source="git:tag --points-at HEAD").value == []
    assert EvidenceValue.known(0, source="git:status").value == 0


def known(value, source: str = "synthetic:test"):
    return EvidenceValue.known(value, source=source)


def unavailable(source: str = "synthetic:test"):
    return EvidenceValue.unavailable(reason="not present", source=source)


def make_snapshot(*, captured_at: str = "2026-09-14T00:00:00+00:00") -> EvidenceSnapshot:
    commit = CommitRecord(
        sha=known("a" * 40, "git:log"),
        short_sha=known("a" * 12, "git:log"),
        authored_at=known("2026-09-13T00:00:00+00:00", "git:log"),
        subject=known("synthetic commit", "git:log"),
    )
    changed = ChangedFileRecord(
        path=known("README.md", "git:diff --name-status"),
        status=known("M", "git:diff --name-status"),
    )
    command = DiscoveredCommand(
        name=known("test", "package.json"),
        command=known("pytest", "package.json"),
        kind=known("package_script", "package.json"),
    )
    recorded = RecordedTestResult(
        summary=known("7 passed", "docs:docs/STATE.md"),
        execution_status=known("recorded_not_executed", "docs:docs/STATE.md"),
        source_document=known("docs/STATE.md", "docs:docs/STATE.md"),
    )
    guard = ShadowGuardReport(
        head_before=known("a" * 40, "shadow_guard:before"),
        head_after=known("a" * 40, "shadow_guard:after"),
        index_hash_before=known("b" * 64, "shadow_guard:before"),
        index_hash_after=known("b" * 64, "shadow_guard:after"),
        status_hash_before=known("c" * 64, "shadow_guard:before"),
        status_hash_after=known("c" * 64, "shadow_guard:after"),
        tracked_manifest_before=known("d" * 64, "shadow_guard:before"),
        tracked_manifest_after=known("d" * 64, "shadow_guard:after"),
        head_unchanged=known(True, "shadow_guard:comparison"),
        index_unchanged=known(True, "shadow_guard:comparison"),
        status_unchanged=known(True, "shadow_guard:comparison"),
        tracked_content_unchanged=known(True, "shadow_guard:comparison"),
        changed_components=known([], "shadow_guard:comparison"),
        verdict=known("SHADOW_COLLECTION_PASS", "shadow_guard:comparison"),
    )
    return EvidenceSnapshot(
        schema_version=known("1.0", "collector:schema"),
        project_id=known("synthetic-interview-agent", "cli:--project-id"),
        captured_at=known(captured_at, "collector:clock"),
        source_mode=known("shadow_read_only", "collector:mode"),
        repository=RepositoryEvidence(
            path=known(
                r"C:\Users\demo\projects\interview-agent-finals",
                "cli:--project",
            ),
            exists=known(True, "filesystem:project-root"),
            project_name=known(
                "synthetic-interview-agent-示例",
                "filesystem:project-root",
            ),
        ),
        git=GitEvidence(
            is_repository=known(True, "git:rev-parse --is-inside-work-tree"),
            branch=known("main", "git:branch --show-current"),
            head_sha=known("a" * 40, "git:rev-parse HEAD"),
            head_short=known("a" * 12, "git:rev-parse --short=12 HEAD"),
            tags_at_head=known([], "git:tag --points-at HEAD"),
            remote_count=known(0, "git-common-dir/config"),
            staged_count=known(0, "git:diff --cached --name-only -z"),
            modified_count=known(1, "git:diff --name-only -z"),
            untracked_count=known(0, "git:ls-files --others --exclude-standard -z"),
            recent_commits=known([commit], "git:log -n 10"),
        ),
        docs=DocsEvidence(
            discovered=known(["README.md"], "docs:discovery"),
            agents=unavailable("docs:AGENTS.md"),
            project_context=unavailable("docs:docs/PROJECT_CONTEXT.md"),
            state=unavailable("docs:docs/STATE.md"),
            next=unavailable("docs:docs/NEXT.md"),
            decisions=unavailable("docs:docs/DECISIONS.md"),
            session_log=unavailable("docs:docs/SESSION_LOG.md"),
            readme=unavailable("docs:README.md"),
        ),
        tests=SnapshotTestsEvidence(
            discovered_test_roots=known(["tests"], "tests:discovery"),
            python_test_files=known(["tests/test_example.py"], "tests:discovery"),
            frontend_test_files=known([], "tests:discovery"),
            discovered_commands=known([command], "tests:package.json"),
            last_recorded_results=known([recorded], "tests:recorded-results"),
        ),
        recent_changes=RecentChangesEvidence(
            changed_files=known([changed], "git:diff --name-status"),
            diff_stat=known("1 file changed", "git:diff --stat"),
            recent_commit_summary=known(["aaaaaaaaaaaa synthetic commit"], "git:log -n 10"),
        ),
        collector=CollectorEvidence(
            warnings=known([], "collector:warnings"),
            errors=known([], "collector:errors"),
            evidence_sources=known(
                ["git:rev-parse HEAD", "docs:README.md", "package.json"],
                "collector:source-inventory",
            ),
            shadow_guard=known(guard, "shadow_guard:comparison"),
        ),
    )


def evidence_nodes_without_source(value, path: str = "") -> list[str]:
    missing: list[str] = []
    if isinstance(value, dict):
        wire_status = value.get("status")
        if (
            isinstance(wire_status, str)
            and wire_status in {"known", "unknown", "unavailable"}
            and "source" not in value
        ):
            missing.append(path)
        for key, child in value.items():
            missing.extend(evidence_nodes_without_source(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            missing.extend(evidence_nodes_without_source(child, f"{path}/{index}"))
    return missing


def test_utc_capture_time_has_explicit_timezone() -> None:
    parsed = datetime.fromisoformat(utc_now_iso())
    assert parsed.utcoffset() == timedelta(0)


def test_snapshot_shape_contains_required_sections() -> None:
    data = make_snapshot().to_dict()
    assert set(data) == {
        "schema_version",
        "project_id",
        "captured_at",
        "source_mode",
        "repository",
        "git",
        "docs",
        "tests",
        "recent_changes",
        "collector",
    }
    assert set(data["collector"]) == {
        "warnings",
        "errors",
        "evidence_sources",
        "shadow_guard",
    }


def test_canonical_json_is_deterministic_and_preserves_unicode() -> None:
    snapshot = make_snapshot()
    first = snapshot_to_json(snapshot)
    second = snapshot_to_json(snapshot)
    assert first == second
    assert "synthetic-interview-agent-示例" in first
    assert "\\u793a" not in first
    assert first.endswith("\n")
    assert json.loads(first) == snapshot.to_dict()


def test_snapshot_from_json_round_trips_canonical_snapshot() -> None:
    original = make_snapshot()
    encoded = snapshot_to_json(original)
    loaded = snapshot_from_json(encoded)
    assert loaded == original
    assert snapshot_to_json(loaded) == encoded


def test_snapshot_from_json_rejects_unexpected_root_key() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    payload["unexpected"] = True
    with pytest.raises(ValueError, match=r"unexpected keys at \$"):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_missing_root_key() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    del payload["collector"]
    with pytest.raises(ValueError, match=r"missing keys at \$: \['collector'\]"):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_invalid_evidence_status() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    payload["schema_version"]["status"] = "certain"
    with pytest.raises(
        ValueError,
        match=r"invalid EvidenceStatus at \$\.schema_version\.status: 'certain'",
    ):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_known_value_with_reason() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    payload["schema_version"]["reason"] = "conflicting source"
    with pytest.raises(
        ValueError,
        match=r"unexpected keys at \$\.schema_version: \['reason'\]",
    ):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_non_known_value_with_value() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    payload["docs"]["agents"]["value"] = "AGENTS.md"
    with pytest.raises(
        ValueError,
        match=r"unexpected keys at \$\.docs\.agents: \['value'\]",
    ):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_wrong_scalar_type() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    payload["git"]["remote_count"]["value"] = "0"
    with pytest.raises(
        ValueError,
        match=r"expected int at \$\.git\.remote_count\.value",
    ):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_malformed_nested_record() -> None:
    payload = json.loads(snapshot_to_json(make_snapshot()))
    del payload["git"]["recent_commits"]["value"][0]["subject"]
    with pytest.raises(
        ValueError,
        match=(
            r"missing keys at \$\.git\.recent_commits\.value\[0\]: "
            r"\['subject'\]"
        ),
    ):
        snapshot_from_json(json.dumps(payload))


def test_snapshot_from_json_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="evidence root must be an object"):
        snapshot_from_json("[]")


def test_snapshot_from_json_reports_invalid_json() -> None:
    with pytest.raises(
        ValueError,
        match="invalid evidence JSON: Expecting property name enclosed in double quotes",
    ):
        snapshot_from_json("{")


def test_every_serialized_evidence_value_has_source() -> None:
    assert evidence_nodes_without_source(make_snapshot().to_dict()) == []


def test_snapshot_diff_can_ignore_capture_time() -> None:
    snapshot = make_snapshot()
    later = replace(
        snapshot,
        captured_at=known("2026-09-14T00:01:00+00:00", "collector:clock"),
    )
    assert diff_snapshots(snapshot, later, ignore_paths={"/captured_at"}) == []


def test_snapshot_diff_reports_stable_path() -> None:
    snapshot = make_snapshot()
    changed_repository = replace(
        snapshot.repository,
        project_name=known("renamed", "filesystem:project-root"),
    )
    changed = replace(snapshot, repository=changed_repository)
    differences = diff_snapshots(snapshot, changed)
    assert [difference.path for difference in differences] == [
        "/repository/project_name/value"
    ]
