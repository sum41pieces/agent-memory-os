from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import agent_memory_os.evidence.collector as collector_module

from agent_memory_os.adapters.git_adapter import GitAdapter, ShadowIntegrityState
from agent_memory_os.evidence.collector import (
    EvidenceCollector,
    compare_integrity_states,
    main,
)
from agent_memory_os.safety.shadow_policy import PolicyViolation, ShadowProjectPolicy


def run_fixture_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=True,
        shell=False,
    )


@pytest.fixture
def git_synthetic_project(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "examples/synthetic-project"
    root = tmp_path / "synthetic-interview-agent"
    shutil.copytree(source, root)
    run_fixture_git(root, "init", "-b", "main")
    run_fixture_git(root, "add", ".")
    run_fixture_git(
        root,
        "-c",
        "user.name=AgentMemoryOSTest",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "synthetic evidence baseline",
    )
    return root


def test_collector_aggregates_synthetic_evidence_and_provenance(
    git_synthetic_project: Path,
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    artifacts = product / "artifacts"
    artifacts.mkdir(parents=True)
    output = artifacts / "synthetic.evidence.json"

    result = EvidenceCollector(product_root=product).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )

    assert result.verdict == "SHADOW_COLLECTION_PASS"
    assert result.exit_code == 0
    assert result.safety_violation is False
    assert result.snapshot is not None
    assert result.snapshot.project_id.value == "synthetic"
    assert result.snapshot.repository.project_name.value == "synthetic-interview-agent"
    assert result.snapshot.git.is_repository.value is True
    assert result.snapshot.docs.state.value.title.value == "Synthetic State"
    assert result.snapshot.tests.last_recorded_results.value[0].execution_status.value == (
        "recorded_not_executed"
    )
    sources = result.snapshot.collector.evidence_sources.value
    assert "git:rev-parse HEAD" in sources
    assert "docs:docs/STATE.md" in sources
    assert "package.json:package.json" in sources
    guard = result.snapshot.collector.shadow_guard.value
    assert guard.head_unchanged.value is True
    assert guard.index_unchanged.value is True
    assert guard.status_unchanged.value is True
    assert guard.tracked_content_unchanged.value is True
    assert guard.head_before.source == "git:rev-parse HEAD"
    assert guard.index_hash_before.source.startswith("git-index:")
    assert guard.status_hash_before.source.startswith("git:status ")
    assert guard.tracked_manifest_before.source.startswith("git:ls-files -z")
    assert "docs:AGENTS.md" in sources
    assert any(source.startswith("git-config-safety:") for source in sources)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["project_id"]["value"] == "synthetic"
    assert "synthetic-interview-agent" in output.read_text(encoding="utf-8")


def test_output_outside_product_artifacts_is_rejected_before_collection(
    git_synthetic_project: Path,
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    (product / "artifacts").mkdir(parents=True)
    output = tmp_path / "outside.json"
    result = EvidenceCollector(product_root=product).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )
    assert result.exit_code == 2
    assert result.console_lines[-1] == "SHADOW_COLLECTION_FAIL"
    assert not output.exists()


def test_collector_rejects_artifact_write_inside_inspected_project_before_guard(
    git_synthetic_project: Path,
) -> None:
    output = git_synthetic_project / "artifacts/evidence.json"
    result = EvidenceCollector(product_root=git_synthetic_project).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )
    assert result.exit_code == 2
    assert result.console_lines[0].startswith("POLICY_VIOLATION=")
    assert not output.exists()


def test_non_git_project_cannot_claim_zero_write_proof(tmp_path: Path) -> None:
    product = tmp_path / "product"
    (product / "artifacts").mkdir(parents=True)
    project = tmp_path / "not-git"
    project.mkdir()
    output = product / "artifacts/non-git.json"
    result = EvidenceCollector(product_root=product).collect(
        project=project,
        project_id="not-git",
        output=output,
    )
    assert result.verdict == "SHADOW_COLLECTION_FAIL"
    assert result.exit_code == 3
    assert result.safety_violation is False
    assert not output.exists()


def test_integrity_comparison_names_only_changed_components() -> None:
    before = ShadowIntegrityState(
        head="a",
        index_hash="b",
        status_hash="c",
        tracked_manifest_hash="d",
        tracked_count=1,
        proof_capable=True,
        issues=(),
    )
    after = replace(before, index_hash="changed", tracked_manifest_hash="changed")
    comparison = compare_integrity_states(before, after)
    assert comparison.changed_components == ("INDEX", "TRACKED_CONTENT")
    assert comparison.proof_capable is True


def test_missing_postflight_measurement_is_unavailable_not_a_change() -> None:
    before = ShadowIntegrityState(
        head="a",
        index_hash="b",
        status_hash="c",
        tracked_manifest_hash="d",
        tracked_count=1,
        proof_capable=True,
        issues=(),
    )
    after = replace(
        before,
        status_hash=None,
        proof_capable=False,
        issues=("status timed out",),
    )
    comparison = compare_integrity_states(before, after)
    assert comparison.changed_components == ()
    assert comparison.proof_capable is False


def test_definite_change_is_retained_when_another_measurement_is_unavailable() -> None:
    before = ShadowIntegrityState(
        head="a",
        index_hash="b",
        status_hash="c",
        tracked_manifest_hash="d",
        tracked_count=1,
        proof_capable=True,
        issues=(),
    )
    after = replace(
        before,
        index_hash="changed",
        status_hash=None,
        proof_capable=False,
        issues=("status timed out",),
    )
    comparison = compare_integrity_states(before, after)
    assert comparison.changed_components == ("INDEX",)
    assert comparison.proof_capable is False


def test_unexpected_collection_error_still_runs_postflight_once(
    git_synthetic_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = tmp_path / "product"
    output = product / "artifacts/error.json"
    capture_count = 0

    class CountingAdapter(GitAdapter):
        def capture_integrity_state(self) -> ShadowIntegrityState:
            nonlocal capture_count
            capture_count += 1
            return super().capture_integrity_state()

    def fail_docs(_self):
        raise OSError("injected document read failure")

    monkeypatch.setattr(collector_module.DocsAdapter, "collect", fail_docs)
    result = EvidenceCollector(
        product_root=product,
        git_adapter_factory=CountingAdapter,
    ).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )
    assert capture_count == 2
    assert result.exit_code == 3
    assert result.safety_violation is False
    assert "collection failed" in result.issues[0]
    assert not output.exists()


def test_output_policy_is_revalidated_after_guard_before_write(
    git_synthetic_project: Path,
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    output = product / "artifacts/revalidated.json"

    class RejectSecondValidationPolicy(ShadowProjectPolicy):
        def __init__(self) -> None:
            object.__setattr__(self, "calls", 0)

        def validate_collection_request(self, **kwargs) -> None:
            object.__setattr__(self, "calls", self.calls + 1)
            if self.calls == 2:
                raise PolicyViolation("destination changed before artifact write")
            super().validate_collection_request(**kwargs)

    policy = RejectSecondValidationPolicy()
    result = EvidenceCollector(product_root=product, policy=policy).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )
    assert policy.calls == 2
    assert result.exit_code == 2
    assert not output.exists()


def test_postflight_exception_is_proof_unavailable_without_artifact(
    git_synthetic_project: Path,
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    output = product / "artifacts/postflight-error.json"

    class FailingPostflightAdapter(GitAdapter):
        def __init__(self, root: Path, policy: ShadowProjectPolicy) -> None:
            super().__init__(root, policy)
            self.capture_count = 0

        def capture_integrity_state(self) -> ShadowIntegrityState:
            self.capture_count += 1
            if self.capture_count == 2:
                raise OSError("injected postflight failure")
            return super().capture_integrity_state()

    result = EvidenceCollector(
        product_root=product,
        git_adapter_factory=FailingPostflightAdapter,
    ).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )
    assert result.exit_code == 3
    assert result.safety_violation is False
    assert "post-collection guard failed" in result.issues[0]
    assert not output.exists()


class SequencedIntegrityGitAdapter(GitAdapter):
    def __init__(
        self,
        project_root: Path,
        policy: ShadowProjectPolicy,
        states: tuple[ShadowIntegrityState, ShadowIntegrityState],
    ) -> None:
        super().__init__(project_root, policy)
        self._states = iter(states)

    def capture_integrity_state(self) -> ShadowIntegrityState:
        return next(self._states)


@pytest.mark.parametrize(
    ("component", "changes"),
    [
        ("HEAD", {"head": "changed"}),
        ("INDEX", {"index_hash": "changed"}),
        ("STATUS", {"status_hash": "changed"}),
        ("TRACKED_CONTENT", {"tracked_manifest_hash": "changed"}),
    ],
)
def test_guard_change_fails_fast_without_artifact_or_repair(
    component: str,
    changes: dict[str, str],
    git_synthetic_project: Path,
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    artifacts = product / "artifacts"
    artifacts.mkdir(parents=True)
    output = artifacts / "must-not-exist.json"
    baseline = GitAdapter(
        git_synthetic_project,
        ShadowProjectPolicy(),
    ).capture_integrity_state()
    after = replace(baseline, **changes)

    def factory(root: Path, policy: ShadowProjectPolicy):
        return SequencedIntegrityGitAdapter(root, policy, (baseline, after))

    result = EvidenceCollector(
        product_root=product,
        git_adapter_factory=factory,
    ).collect(
        project=git_synthetic_project,
        project_id="synthetic",
        output=output,
    )

    assert result.verdict == "SHADOW_COLLECTION_FAIL"
    assert result.exit_code == 4
    assert result.safety_violation is True
    assert result.changed_components == (component,)
    assert result.console_lines[0] == "SHADOW_SAFETY_VIOLATION"
    assert result.console_lines[1] == f"CHANGED_COMPONENT={component}"
    assert result.console_lines[-1] == "SHADOW_COLLECTION_FAIL"
    assert not output.exists()
    actual = GitAdapter(
        git_synthetic_project,
        ShadowProjectPolicy(),
    ).capture_integrity_state()
    assert actual == baseline


def test_cli_prints_pass_for_synthetic_project(
    git_synthetic_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = tmp_path / "product"
    (product / "artifacts").mkdir(parents=True)
    output = product / "artifacts/cli.json"
    exit_code = main(
        [
            "--project",
            str(git_synthetic_project),
            "--project-id",
            "synthetic",
            "--output",
            str(output),
        ],
        product_root=product,
    )
    assert exit_code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        "SHADOW_COLLECTION_PASS"
    )
