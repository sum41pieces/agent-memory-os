"""Deny-by-default policy for read-only Shadow project collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {"status", "log", "diff", "show", "rev-parse", "branch", "tag", "ls-files"}
)
ALLOWED_GIT_ARGV = frozenset(
    {
        ("git", "rev-parse", "--is-inside-work-tree"),
        ("git", "rev-parse", "--show-toplevel"),
        ("git", "rev-parse", "--git-dir"),
        ("git", "rev-parse", "--git-common-dir"),
        ("git", "rev-parse", "--git-path", "index"),
        ("git", "rev-parse", "HEAD"),
        ("git", "rev-parse", "--short=12", "HEAD"),
        ("git", "branch", "--show-current"),
        ("git", "tag", "--points-at", "HEAD"),
        ("git", "ls-files", "-z"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        (
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
        ),
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--cached",
            "--name-only",
            "-z",
        ),
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--name-only",
            "-z",
        ),
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--stat",
            "HEAD",
        ),
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--name-status",
            "-z",
            "HEAD",
        ),
        (
            "git",
            "log",
            "-n",
            "10",
            "--date=iso-strict",
            "--pretty=format:%H%x1f%h%x1f%aI%x1f%s%x1e",
        ),
    }
)
FORBIDDEN_GIT_SUBCOMMANDS = frozenset(
    {"add", "commit", "restore", "checkout", "reset", "clean"}
)
FORBIDDEN_GIT_OPTIONS = frozenset(
    {"--output", "--exec", "--ext-diff", "--textconv", "--no-index"}
)


class PolicyViolation(RuntimeError):
    """Raised before an operation that violates Shadow authority."""


@dataclass(frozen=True)
class ShadowProjectPolicy:
    """Permanent default permissions for a Shadow project."""

    read: bool = True
    analyze: bool = True
    write: bool = False
    destructive: bool = False

    def validate_git_argv(self, argv: Sequence[str]) -> None:
        if not isinstance(argv, (list, tuple)):
            raise PolicyViolation("Git invocation must use an argument list")
        if len(argv) < 2 or argv[0] != "git":
            raise PolicyViolation("Git argument list must begin with git and a subcommand")
        if any(not isinstance(argument, str) or "\x00" in argument for argument in argv):
            raise PolicyViolation("Git arguments must be NUL-free strings")

        subcommand = argv[1]
        if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
            raise PolicyViolation(f"Git subcommand is not allowed: {subcommand}")

        for argument in argv[2:]:
            option_name = argument.split("=", 1)[0]
            if option_name in FORBIDDEN_GIT_OPTIONS:
                raise PolicyViolation(f"Git option is not allowed: {argument}")
        if tuple(argv) not in ALLOWED_GIT_ARGV:
            raise PolicyViolation(
                f"Git argument shape is not allowed: {' '.join(argv)}"
            )

    def validate_collection_request(
        self,
        *,
        source_mode: str,
        output_path: Path,
        artifacts_root: Path,
        project_root: Path,
        requested_write: bool = False,
        requested_destructive: bool = False,
        execute_tests: bool = False,
        execute_build: bool = False,
        start_service: bool = False,
        install_dependencies: bool = False,
    ) -> None:
        if source_mode != "shadow_read_only":
            raise PolicyViolation(f"unsupported source mode: {source_mode}")
        forbidden_intents = {
            "requested_write": requested_write,
            "requested_destructive": requested_destructive,
            "execute_tests": execute_tests,
            "execute_build": execute_build,
            "start_service": start_service,
            "install_dependencies": install_dependencies,
        }
        requested = sorted(name for name, enabled in forbidden_intents.items() if enabled)
        if requested:
            raise PolicyViolation(
                "forbidden operation requested: " + ", ".join(requested)
            )

        declared_product_root = artifacts_root.parent.resolve(strict=False)
        declared_artifacts = declared_product_root / artifacts_root.name
        resolved_artifacts = artifacts_root.resolve(strict=False)
        if (
            resolved_artifacts != declared_artifacts
            or not resolved_artifacts.is_relative_to(declared_product_root)
        ):
            raise PolicyViolation(
                "artifacts directory must stay inside the product root and cannot be redirected"
            )
        resolved_output = output_path.resolve(strict=False)
        if resolved_output == resolved_artifacts:
            raise PolicyViolation("output must be a file below the artifacts directory")
        if not resolved_output.is_relative_to(resolved_artifacts):
            raise PolicyViolation(
                "output must stay inside the product artifacts directory"
            )
        resolved_project = project_root.resolve(strict=False)
        if resolved_output.is_relative_to(resolved_project):
            raise PolicyViolation(
                "output must not be inside the collected project"
            )
        if (
            resolved_project.is_relative_to(declared_product_root)
            or declared_product_root.is_relative_to(resolved_project)
        ):
            raise PolicyViolation(
                "product and collected project trees must be disjoint"
            )
