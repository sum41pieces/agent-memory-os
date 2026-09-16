import tomllib
from pathlib import Path


def test_package_version_and_phase_boundary() -> None:
    import agent_memory_os

    assert agent_memory_os.__version__ == "0.3.0"
    root = Path(__file__).parents[1]
    assert not (root / "src/agent_memory_os/current_state").exists()
    assert not (root / "src/agent_memory_os/dashboard").exists()


def test_readme_reports_public_alpha_truthfully() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Persistent, auditable project state" in readme
    assert "Alpha / Work in Progress" in readme
    assert "Coding agents often lose important project state across sessions." in readme
    assert "Memory is not authority." in readme
    assert "recorded_not_executed" in readme
    assert "SHADOW_COLLECTION_PASS" in readme


def test_readme_separates_implemented_in_progress_and_planned() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    implemented, remainder = readme.split("### In progress", 1)
    in_progress, planned = remainder.split("### Planned", 1)
    for name in (
        "Evidence Collector",
        "Read-only Shadow Project Policy",
        "State Reconciler",
    ):
        assert name in implemented
    assert "Current State Model" in in_progress
    for name in (
        "Context Compiler",
        "Authority Gate",
        "Checkpoint Writer",
        "Eval Harness",
        "Dashboard",
    ):
        assert name in planned
    assert "does not imply permission" in readme


def test_public_metadata_and_runtime_artifact_ignore() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["description"] == (
        "Read-only evidence collection and deterministic state reconciliation "
        "for Agent Memory OS"
    )
    assert project["project"].get("license") == {"text": "MIT"}
    rules = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/artifacts/" in rules
    assert "/runtime-output/" in rules


def test_mit_license_uses_approved_public_identity() -> None:
    license_path = Path("LICENSE")
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8")
    assert text.startswith("MIT License\n")
    assert "Copyright (c) 2026 sum41pieces" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in text
    assert "<AUTHOR_NAME>" not in text
    assert "@" not in text


def test_internal_plans_and_real_artifact_test_are_absent() -> None:
    root = Path(__file__).parents[1]
    assert not (root / "docs/superpowers").exists()
    assert not (root / "tests/test_reconciliation_real_artifact.py").exists()
