import os
from pathlib import Path
import subprocess

import pytest

from agent_memory_os.safety.shadow_policy import (
    ALLOWED_GIT_SUBCOMMANDS,
    PolicyViolation,
    ShadowProjectPolicy,
)


def test_shadow_defaults_allow_only_read_and_analyze() -> None:
    policy = ShadowProjectPolicy()
    assert policy.read is True
    assert policy.analyze is True
    assert policy.write is False
    assert policy.destructive is False


@pytest.mark.parametrize(
    "subcommand",
    ["add", "commit", "restore", "checkout", "reset", "clean"],
)
def test_forbidden_git_commands_are_rejected(subcommand: str) -> None:
    with pytest.raises(PolicyViolation, match="not allowed"):
        ShadowProjectPolicy().validate_git_argv(["git", subcommand])


@pytest.mark.parametrize(
    "option",
    [
        "--output=leak",
        "--output",
        "--exec=touch",
        "--exec",
        "--ext-diff",
        "--textconv",
        "--no-index",
    ],
)
def test_dangerous_git_options_are_rejected(option: str) -> None:
    with pytest.raises(PolicyViolation, match="option is not allowed"):
        ShadowProjectPolicy().validate_git_argv(["git", "diff", option])


def test_git_invocation_must_be_argument_list() -> None:
    with pytest.raises(PolicyViolation, match="argument list"):
        ShadowProjectPolicy().validate_git_argv("git status")  # type: ignore[arg-type]


def test_read_only_git_allowlist_is_accepted() -> None:
    policy = ShadowProjectPolicy()
    for subcommand in sorted(ALLOWED_GIT_SUBCOMMANDS):
        with pytest.raises(PolicyViolation, match="argument shape is not allowed"):
            policy.validate_git_argv(["git", subcommand])


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "branch", "new-branch"],
        ["git", "branch", "-m", "renamed"],
        ["git", "branch", "-D", "old"],
        ["git", "tag", "new-tag"],
        ["git", "tag", "-d", "old-tag"],
    ],
)
def test_mutating_forms_of_allowed_subcommands_are_rejected(argv: list[str]) -> None:
    with pytest.raises(PolicyViolation, match="argument shape is not allowed"):
        ShadowProjectPolicy().validate_git_argv(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "rev-parse", "--is-inside-work-tree"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "--short=12", "HEAD"],
        ["git", "rev-parse", "--git-common-dir"],
        ["git", "rev-parse", "--git-path", "index"],
        ["git", "branch", "--show-current"],
        ["git", "tag", "--points-at", "HEAD"],
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
        ],
    ],
)
def test_collector_git_argument_shapes_are_accepted(argv: list[str]) -> None:
    ShadowProjectPolicy().validate_git_argv(argv)


def test_output_inside_artifacts_is_accepted(tmp_path: Path) -> None:
    artifacts = tmp_path / "product" / "artifacts"
    policy = ShadowProjectPolicy()
    policy.validate_collection_request(
        source_mode="shadow_read_only",
        output_path=artifacts / "sample.json",
        artifacts_root=artifacts,
        project_root=tmp_path / "shadow",
    )


def test_artifacts_directory_itself_is_not_a_valid_output(tmp_path: Path) -> None:
    artifacts = tmp_path / "product" / "artifacts"
    with pytest.raises(PolicyViolation, match="output must be a file below"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=artifacts,
            artifacts_root=artifacts,
            project_root=tmp_path / "shadow",
        )


def test_output_inside_collected_project_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "product"
    artifacts = project / "artifacts"
    with pytest.raises(PolicyViolation, match="must not be inside the collected project"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=artifacts / "sample.json",
            artifacts_root=artifacts,
            project_root=project,
        )


def test_product_and_collected_project_trees_must_be_disjoint(tmp_path: Path) -> None:
    product = tmp_path / "product"
    project = product / "nested-shadow"
    artifacts = product / "artifacts"
    with pytest.raises(PolicyViolation, match="must be disjoint"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=artifacts / "sample.json",
            artifacts_root=artifacts,
            project_root=project,
        )


def test_output_outside_artifacts_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyViolation, match="inside the product artifacts directory"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=tmp_path / "outside.json",
            artifacts_root=tmp_path / "product/artifacts",
            project_root=tmp_path / "shadow",
        )


def test_symlinked_artifacts_directory_cannot_escape_product_root(
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    outside = tmp_path / "outside"
    product.mkdir()
    outside.mkdir()
    artifacts = product / "artifacts"
    try:
        artifacts.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        escaped_artifacts = str(artifacts).replace("'", "''")
        escaped_outside = str(outside).replace("'", "''")
        created = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "New-Item -ItemType Junction "
                    f"-Path '{escaped_artifacts}' -Target '{escaped_outside}'"
                ),
            ],
            capture_output=True,
            check=False,
            shell=False,
        )
        if created.returncode != 0:
            pytest.skip("directory symlinks and junctions unavailable")
    with pytest.raises(PolicyViolation, match="artifacts directory must stay inside"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=artifacts / "sample.json",
            artifacts_root=artifacts,
            project_root=tmp_path / "shadow",
        )


@pytest.mark.parametrize(
    "flag",
    [
        "requested_write",
        "requested_destructive",
        "execute_tests",
        "execute_build",
        "start_service",
        "install_dependencies",
    ],
)
def test_collection_rejects_every_mutating_or_executing_intent(
    tmp_path: Path,
    flag: str,
) -> None:
    artifacts = tmp_path / "artifacts"
    kwargs = {flag: True}
    with pytest.raises(PolicyViolation, match="forbidden operation requested"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="shadow_read_only",
            output_path=artifacts / "sample.json",
            artifacts_root=artifacts,
            project_root=tmp_path / "shadow",
            **kwargs,
        )


def test_collection_rejects_non_shadow_mode(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    with pytest.raises(PolicyViolation, match="source mode"):
        ShadowProjectPolicy().validate_collection_request(
            source_mode="write_enabled",
            output_path=artifacts / "sample.json",
            artifacts_root=artifacts,
            project_root=tmp_path / "shadow",
        )
