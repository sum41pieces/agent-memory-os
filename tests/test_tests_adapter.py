from pathlib import Path
import subprocess

import pytest

from agent_memory_os.adapters.tests_adapter import TestsAdapter as ProjectTestsAdapter
from agent_memory_os.evidence.models import EvidenceStatus


@pytest.fixture
def synthetic_project() -> Path:
    return Path(__file__).parents[1] / "examples/synthetic-project"


def test_tests_adapter_discovers_without_executing(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_project: Path,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("test discovery must not launch subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    adapter = ProjectTestsAdapter(synthetic_project)
    evidence = adapter.collect()

    assert evidence.python_test_files.value == ["tests/test_example.py"]
    assert evidence.frontend_test_files.value == [
        "web/example.test.ts",
        "web/example.test.tsx",
    ]
    assert "tests" in evidence.discovered_test_roots.value
    assert {item.name.value for item in evidence.discovered_commands.value} == {
        "pytest",
        "test",
        "test:unit",
    }
    assert all(item.command.source for item in evidence.discovered_commands.value)
    result = evidence.last_recorded_results.value[0]
    assert result.execution_status.value == "recorded_not_executed"
    assert result.summary.source == "docs:docs/STATE.md"
    assert result.source_document.value == "docs/STATE.md"
    assert "package.json:package.json" in adapter.evidence_sources


def test_generated_and_dependency_directories_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_real.py").write_text("pass\n", encoding="utf-8")
    for directory in ("node_modules", ".venv", "dist", "build", ".git"):
        hidden = tmp_path / directory
        hidden.mkdir()
        (hidden / "hidden.test.ts").write_text("hidden\n", encoding="utf-8")
    evidence = ProjectTestsAdapter(tmp_path).collect()
    assert evidence.python_test_files.value == ["tests/test_real.py"]
    assert evidence.frontend_test_files.value == []


def test_bounded_discovery_is_unknown_instead_of_partial(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_one.py").write_text("pass\n", encoding="utf-8")
    (tests / "test_two.py").write_text("pass\n", encoding="utf-8")
    evidence = ProjectTestsAdapter(tmp_path, max_files=1).collect()
    assert evidence.python_test_files.status is EvidenceStatus.UNKNOWN
    assert evidence.discovered_test_roots.status is EvidenceStatus.UNKNOWN
    assert evidence.discovered_commands.status is EvidenceStatus.UNKNOWN
    assert "file limit" in evidence.python_test_files.reason


def test_invalid_package_json_becomes_warning_without_execution(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{invalid", encoding="utf-8")
    adapter = ProjectTestsAdapter(tmp_path)
    evidence = adapter.collect()
    assert evidence.discovered_commands.status is EvidenceStatus.UNKNOWN
    assert "package.json parse failed" in evidence.discovered_commands.reason
    assert any("package.json parse failed" in warning for warning in adapter.warnings)


def test_truncated_package_json_is_unknown_not_partial(tmp_path: Path) -> None:
    scripts = ','.join(f'"test:{index}": "echo {index}"' for index in range(50000))
    (tmp_path / "package.json").write_text(
        '{"scripts": {' + scripts + "}}",
        encoding="utf-8",
    )
    evidence = ProjectTestsAdapter(tmp_path).collect()
    assert evidence.discovered_commands.status is EvidenceStatus.UNKNOWN
    assert "read limit" in evidence.discovered_commands.reason


def test_only_state_and_session_log_supply_recorded_results(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "NEXT.md").write_text("Tests: 99 passed\n", encoding="utf-8")
    (docs / "SESSION_LOG.md").write_text(
        "Vitest result: 3 failed in a historical run.\n",
        encoding="utf-8",
    )
    results = ProjectTestsAdapter(tmp_path).collect().last_recorded_results.value
    assert len(results) == 1
    assert results[0].summary.value.startswith("Vitest result: 3 failed")
    assert results[0].source_document.value == "docs/SESSION_LOG.md"
