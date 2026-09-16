"""Discover test evidence without executing project commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

from agent_memory_os.adapters.filesystem_adapter import (
    FilesystemAdapter,
    UnsafePathError,
)
from agent_memory_os.evidence.models import (
    DiscoveredCommand,
    EvidenceValue,
    RecordedTestResult,
    TestsEvidence,
)


SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
PYTEST_CONFIG_NAMES = frozenset({"pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"})
RECORDED_RESULT_PATHS = (Path("docs/STATE.md"), Path("docs/SESSION_LOG.md"))
RECORDED_RESULT_PATTERN = re.compile(
    r"(?:test|pytest|vitest|测试).*(?:passed|failed|skipped|pass|fail|通过|失败)",
    flags=re.IGNORECASE,
)


class TestsAdapter:
    """Find test files, configs, scripts, and recorded historical results."""

    def __init__(
        self,
        project_root: Path,
        *,
        max_files: int = 20_000,
        max_depth: int = 32,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.filesystem = FilesystemAdapter(self.project_root)
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.evidence_sources: set[str] = {"tests:discovery"}
        if max_files <= 0 or max_depth <= 0:
            raise ValueError("discovery limits must be positive")
        self.max_files = max_files
        self.max_depth = max_depth

    def collect(self) -> TestsEvidence:
        files, discovery_issues = self._walk_files()
        python_files = sorted(
            relative.as_posix()
            for relative in files
            if relative.name.startswith("test_") and relative.suffix == ".py"
        )
        frontend_files = sorted(
            relative.as_posix()
            for relative in files
            if relative.name.endswith((".test.ts", ".test.tsx"))
        )
        pytest_configs = [
            path for path in files if self._is_pytest_config(path)
        ]

        roots = {
            str(Path(path).parent).replace("\\", "/")
            for path in python_files
        }
        roots.update(
            "." if path.parent == Path(".") else path.parent.as_posix()
            for path in pytest_configs
        )

        commands = self._discover_commands(files, pytest_configs)
        recorded_results = self._discover_recorded_results()
        if discovery_issues:
            reason = "; ".join(discovery_issues)
            roots_value = EvidenceValue.unknown(
                reason=reason,
                source="tests:discovery",
            )
            python_value = EvidenceValue.unknown(
                reason=reason,
                source="tests:discovery",
            )
            frontend_value = EvidenceValue.unknown(
                reason=reason,
                source="tests:discovery",
            )
            commands = EvidenceValue.unknown(
                reason=reason,
                source="tests:commands",
            )
        else:
            roots_value = EvidenceValue.known(
                sorted(roots),
                source="tests:discovery",
            )
            python_value = EvidenceValue.known(
                python_files,
                source="tests:discovery",
            )
            frontend_value = EvidenceValue.known(
                frontend_files,
                source="tests:discovery",
            )
        return TestsEvidence(
            discovered_test_roots=roots_value,
            python_test_files=python_value,
            frontend_test_files=frontend_value,
            discovered_commands=commands,
            last_recorded_results=EvidenceValue.known(
                recorded_results,
                source="tests:recorded-results",
            ),
        )

    def _walk_files(self) -> tuple[list[Path], list[str]]:
        discovered: list[Path] = []
        issues: list[str] = []

        def walk_error(error: OSError) -> None:
            issue = f"test discovery read failed: {error}"
            self.warnings.append(issue)
            issues.append(issue)

        for current, directories, filenames in os.walk(
            self.project_root,
            topdown=True,
            followlinks=False,
            onerror=walk_error,
        ):
            current_path = Path(current)
            relative_current = current_path.relative_to(self.project_root)
            if len(relative_current.parts) >= self.max_depth:
                directories[:] = []
                issue = f"test discovery depth limit reached: {self.max_depth}"
                if issue not in issues:
                    self.warnings.append(issue)
                    issues.append(issue)
            safe_directories: list[str] = []
            for directory in sorted(directories):
                if directory in SKIPPED_DIRECTORIES:
                    continue
                candidate = current_path / directory
                try:
                    resolved = candidate.resolve(strict=False)
                except OSError:
                    continue
                if resolved.is_relative_to(self.project_root):
                    safe_directories.append(directory)
            directories[:] = safe_directories
            for filename in sorted(filenames):
                candidate = current_path / filename
                try:
                    resolved = candidate.resolve(strict=False)
                except OSError:
                    continue
                if not resolved.is_relative_to(self.project_root):
                    self.warnings.append(
                        f"skipped file outside project root: {candidate}"
                    )
                    continue
                if len(discovered) >= self.max_files:
                    issue = f"test discovery file limit reached: {self.max_files}"
                    if issue not in issues:
                        self.warnings.append(issue)
                        issues.append(issue)
                    directories[:] = []
                    break
                discovered.append(candidate.relative_to(self.project_root))
            if len(discovered) >= self.max_files:
                break
        return sorted(discovered, key=lambda path: path.as_posix()), issues

    def _is_pytest_config(self, relative_path: Path) -> bool:
        if relative_path.name not in PYTEST_CONFIG_NAMES:
            return False
        if relative_path.name == "pytest.ini":
            return True
        try:
            result = self.filesystem.read_text(relative_path, max_bytes=256 * 1024)
        except (OSError, UnsafePathError):
            return False
        lowered = result.text.casefold()
        if relative_path.name == "pyproject.toml":
            return "[tool.pytest" in lowered
        if relative_path.name == "setup.cfg":
            return "[tool:pytest]" in lowered
        return "[pytest]" in lowered or "pytest" in lowered

    def _discover_commands(
        self,
        files: list[Path],
        pytest_configs: list[Path],
    ) -> EvidenceValue[list[DiscoveredCommand]]:
        commands: list[DiscoveredCommand] = []
        issues: list[str] = []
        for config in sorted(pytest_configs, key=lambda path: path.as_posix()):
            source = f"pytest-config:{config.as_posix()}"
            self.evidence_sources.add(source)
            commands.append(
                DiscoveredCommand(
                    name=EvidenceValue.known("pytest", source=source),
                    command=EvidenceValue.known("pytest", source=source),
                    kind=EvidenceValue.known("pytest_config", source=source),
                )
            )

        for relative_path in (path for path in files if path.name == "package.json"):
            source = f"package.json:{relative_path.as_posix()}"
            self.evidence_sources.add(source)
            try:
                read = self.filesystem.read_text(
                    relative_path,
                    max_bytes=1024 * 1024,
                )
                if read.truncated:
                    raise ValueError("package.json exceeds the 1 MiB read limit")
                if read.decode_replaced:
                    raise ValueError("package.json is not valid UTF-8")
                data = json.loads(read.text)
                scripts = data.get("scripts", {})
                if not isinstance(scripts, dict):
                    raise ValueError("scripts is not an object")
            except (OSError, UnsafePathError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                issue = (
                    f"package.json parse failed: {relative_path.as_posix()}: {error}"
                )
                self.warnings.append(issue)
                issues.append(issue)
                continue
            for name, command in sorted(scripts.items()):
                if "test" not in str(name).casefold() or not isinstance(command, str):
                    continue
                commands.append(
                    DiscoveredCommand(
                        name=EvidenceValue.known(str(name), source=source),
                        command=EvidenceValue.known(command, source=source),
                        kind=EvidenceValue.known("package_script", source=source),
                    )
                )
        if issues:
            return EvidenceValue.unknown(
                reason="; ".join(issues),
                source="tests:commands",
            )
        return EvidenceValue.known(commands, source="tests:commands")

    def _discover_recorded_results(self) -> list[RecordedTestResult]:
        records: list[RecordedTestResult] = []
        for relative_path in RECORDED_RESULT_PATHS:
            absolute = self.project_root / relative_path
            if not absolute.is_file():
                continue
            source = f"docs:{relative_path.as_posix()}"
            self.evidence_sources.add(source)
            try:
                read = self.filesystem.read_text(relative_path, max_bytes=64 * 1024)
            except (OSError, UnsafePathError) as error:
                self.warnings.append(
                    f"recorded test result read failed: {relative_path.as_posix()}: {error}"
                )
                continue
            for line in read.text.splitlines():
                summary = line.strip()
                if not summary or not RECORDED_RESULT_PATTERN.search(summary):
                    continue
                records.append(
                    RecordedTestResult(
                        summary=EvidenceValue.known(summary[:500], source=source),
                        execution_status=EvidenceValue.known(
                            "recorded_not_executed",
                            source=source,
                        ),
                        source_document=EvidenceValue.known(
                            relative_path.as_posix(),
                            source=source,
                        ),
                    )
                )
        return records
