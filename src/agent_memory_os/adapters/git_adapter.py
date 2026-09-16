"""Policy-gated, read-only Git evidence collection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

from agent_memory_os.adapters.filesystem_adapter import FilesystemAdapter
from agent_memory_os.evidence.models import (
    ChangedFileRecord,
    CommitRecord,
    EvidenceValue,
    GitEvidence,
)
from agent_memory_os.safety.shadow_policy import ShadowProjectPolicy


@dataclass(frozen=True)
class GitCommandResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def stdout_text(self) -> tuple[str, bool]:
        return _decode_utf8(self.stdout)

    def stderr_text(self) -> tuple[str, bool]:
        return _decode_utf8(self.stderr)


@dataclass(frozen=True)
class ShadowIntegrityState:
    head: str | None
    index_hash: str | None
    status_hash: str | None
    tracked_manifest_hash: str | None
    tracked_count: int | None
    proof_capable: bool
    issues: tuple[str, ...]
    component_sources: tuple[tuple[str, str], ...] = ()


def _decode_utf8(data: bytes) -> tuple[str, bool]:
    try:
        return data.decode("utf-8", errors="strict"), False
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), True


class GitAdapter:
    """Run only approved Git commands and translate failures into evidence."""

    def __init__(
        self,
        project_root: Path,
        policy: ShadowProjectPolicy,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.project_root = project_root.resolve(strict=False)
        self.policy = policy
        self.timeout_seconds = timeout_seconds
        discovered_git = shutil.which("git")
        try:
            resolved_git = (
                Path(discovered_git).resolve(strict=True)
                if discovered_git is not None
                else None
            )
        except OSError:
            resolved_git = None
        if (
            resolved_git is None
            or not resolved_git.is_absolute()
            or not resolved_git.is_file()
            or resolved_git.is_relative_to(self.project_root)
        ):
            self.git_executable: str | None = None
        else:
            self.git_executable = str(resolved_git)
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.evidence_sources: set[str] = set()

    def run(self, argv: list[str]) -> GitCommandResult:
        self.policy.validate_git_argv(argv)
        if self.git_executable is None:
            return GitCommandResult(
                tuple(argv),
                None,
                b"",
                b"trusted Git executable is unavailable",
            )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        environment.update(
            {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
        )
        safe_config = (
            ("core.fsmonitor", "false"),
            ("core.untrackedCache", "false"),
            ("core.hooksPath", os.devnull),
            ("protocol.allow", "never"),
            ("submodule.recurse", "false"),
            ("diff.external", ""),
            ("diff.trustExitCode", "false"),
        )
        environment["GIT_CONFIG_COUNT"] = str(len(safe_config))
        for index, (key, value) in enumerate(safe_config):
            environment[f"GIT_CONFIG_KEY_{index}"] = key
            environment[f"GIT_CONFIG_VALUE_{index}"] = value
        try:
            completed = subprocess.run(
                argv,
                executable=self.git_executable,
                cwd=self.project_root,
                env=environment,
                capture_output=True,
                check=False,
                shell=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, bytes) else b""
            stderr = error.stderr if isinstance(error.stderr, bytes) else b""
            return GitCommandResult(tuple(argv), None, stdout, stderr, timed_out=True)
        except OSError as error:
            return GitCommandResult(
                tuple(argv),
                None,
                b"",
                str(error).encode("utf-8", errors="replace"),
            )
        return GitCommandResult(
            tuple(argv),
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )

    def collect(self) -> GitEvidence:
        source = "git:rev-parse --is-inside-work-tree"
        self.evidence_sources.add(source)
        result = self.run(["git", "rev-parse", "--is-inside-work-tree"])
        if not result.ok or result.stdout.strip() != b"true":
            return self._non_repository_evidence(source)
        return GitEvidence(
            is_repository=EvidenceValue.known(True, source=source),
            branch=self._collect_branch(),
            head_sha=self._collect_required_text(
                ["git", "rev-parse", "HEAD"],
                "git:rev-parse HEAD",
            ),
            head_short=self._collect_required_text(
                ["git", "rev-parse", "--short=12", "HEAD"],
                "git:rev-parse --short=12 HEAD",
            ),
            tags_at_head=self._collect_tags(),
            remote_count=self._collect_remote_count(),
            staged_count=self._collect_path_count(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--ignore-submodules=all",
                    "--cached",
                    "--name-only",
                    "-z",
                ],
                "git:diff --cached --name-only -z",
            ),
            modified_count=self._collect_path_count(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--ignore-submodules=all",
                    "--name-only",
                    "-z",
                ],
                "git:diff --name-only -z",
            ),
            untracked_count=self._collect_path_count(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                "git:ls-files --others --exclude-standard -z",
            ),
            recent_commits=self._collect_recent_commits(),
        )

    def resolve_common_directory(self) -> Path:
        source = "git:rev-parse --git-common-dir"
        text = self._required_command_text(
            ["git", "rev-parse", "--git-common-dir"],
            source,
        )
        return self._resolve_git_reported_path(text)

    def resolve_index_path(self) -> Path:
        source = "git:rev-parse --git-path index"
        text = self._required_command_text(
            ["git", "rev-parse", "--git-path", "index"],
            source,
        )
        index_path = self._absolute_git_reported_path(text)
        git_directory = self.resolve_git_directory()
        self._validate_git_metadata_path(index_path, git_directory, "Git index")
        return index_path.resolve(strict=False)

    def resolve_git_directory(self) -> Path:
        source = "git:rev-parse --git-dir"
        text = self._required_command_text(
            ["git", "rev-parse", "--git-dir"],
            source,
        )
        return self._resolve_git_reported_path(text)

    def collect_diff_stat(self) -> EvidenceValue[str]:
        source = "git:diff --stat HEAD"
        return self._collect_required_text(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=all",
                "--stat",
                "HEAD",
            ],
            source,
            strip=False,
        )

    def collect_changed_files(self) -> EvidenceValue[list[ChangedFileRecord]]:
        source = "git:diff --name-status -z HEAD"
        self.evidence_sources.add(source)
        result = self.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=all",
                "--name-status",
                "-z",
                "HEAD",
            ]
        )
        if not result.ok:
            reason = self._record_command_warning(source, result)
            return EvidenceValue.unavailable(reason=reason, source=source)
        text, replaced = result.stdout_text()
        if replaced:
            reason = f"changed-file output is not valid UTF-8: {source}"
            self.warnings.append(reason)
            return EvidenceValue.unknown(reason=reason, source=source)
        tokens = [token for token in text.split("\x00") if token]
        records: list[ChangedFileRecord] = []
        index = 0
        while index < len(tokens):
            status = tokens[index]
            index += 1
            if status.startswith(("R", "C")) and index + 1 < len(tokens):
                old_path, new_path = tokens[index], tokens[index + 1]
                path = f"{old_path} -> {new_path}"
                index += 2
            elif index < len(tokens):
                path = tokens[index]
                index += 1
            else:
                reason = f"incomplete name-status record from {source}"
                self.warnings.append(reason)
                return EvidenceValue.unknown(reason=reason, source=source)
            records.append(
                ChangedFileRecord(
                    path=EvidenceValue.known(path, source=source),
                    status=EvidenceValue.known(status, source=source),
                )
            )
        return EvidenceValue.known(records, source=source)

    def capture_integrity_state(self) -> ShadowIntegrityState:
        issues: list[str] = []
        component_sources: list[tuple[str, str]] = []
        repository = self.run(["git", "rev-parse", "--is-inside-work-tree"])
        self.evidence_sources.add("git:rev-parse --is-inside-work-tree")
        if not repository.ok or repository.stdout.strip() != b"true":
            return ShadowIntegrityState(
                None,
                None,
                None,
                None,
                None,
                False,
                ("project is not a Git repository",),
            )

        config_issues = self._unsafe_repository_config_issues()
        if config_issues:
            return ShadowIntegrityState(
                None,
                None,
                None,
                None,
                None,
                False,
                tuple(config_issues),
            )

        top_level_issues: list[str] = []
        top_level = self._integrity_text(
            ["git", "rev-parse", "--show-toplevel"],
            "git:rev-parse --show-toplevel",
            top_level_issues,
        )
        if top_level is None:
            return ShadowIntegrityState(
                None,
                None,
                None,
                None,
                None,
                False,
                tuple(top_level_issues),
            )
        if Path(top_level).resolve(strict=False) != self.project_root:
            return ShadowIntegrityState(
                None,
                None,
                None,
                None,
                None,
                False,
                ("project path is not the Git worktree root",),
            )

        head_issues: list[str] = []
        head = self._integrity_text(
            ["git", "rev-parse", "HEAD"],
            "git:rev-parse HEAD",
            head_issues,
        )
        if head is None:
            unborn_marker = self._unborn_head_marker()
            if unborn_marker is None:
                issues.extend(head_issues)
            else:
                head = unborn_marker
        component_sources.append(("HEAD", "git:rev-parse HEAD"))

        index_hash: str | None = None
        try:
            index_path = self.resolve_index_path()
            component_sources.append(("INDEX", f"git-index:{index_path}"))
            if index_path.is_file():
                index_hash = FilesystemAdapter.hash_file(index_path)
            elif not index_path.exists():
                index_hash = "MISSING_INDEX"
            else:
                issues.append(f"Git index is unavailable: {index_path}")
        except (OSError, RuntimeError) as error:
            issues.append(f"Git index hash unavailable: {error}")

        status_source = (
            "git:status --porcelain=v1 -z --untracked-files=all "
            "--ignore-submodules=all"
        )
        self.evidence_sources.add(status_source)
        component_sources.append(("STATUS", status_source))
        status_result = self.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=all",
            ]
        )
        status_hash: str | None = None
        if status_result.ok:
            status_hash = hashlib.sha256(status_result.stdout).hexdigest()
        else:
            issues.append(self._command_failure_reason(status_source, status_result))

        tracked_source = "git:ls-files -z"
        self.evidence_sources.add(tracked_source)
        component_sources.append(
            (
                "TRACKED_CONTENT",
                "git:ls-files -z + filesystem:tracked-manifest-sha256",
            )
        )
        tracked_result = self.run(["git", "ls-files", "-z"])
        manifest_hash: str | None = None
        tracked_count: int | None = None
        if tracked_result.ok:
            tracked_text, replaced = tracked_result.stdout_text()
            if replaced:
                issues.append("tracked path list was not valid UTF-8")
            tracked_paths = [path for path in tracked_text.split("\x00") if path]
            manifest = FilesystemAdapter(self.project_root).hash_tracked_manifest(
                tracked_paths
            )
            manifest_hash = manifest.aggregate_sha256
            tracked_count = manifest.tracked_count
            issues.extend(manifest.issues)
        else:
            issues.append(self._command_failure_reason(tracked_source, tracked_result))

        return ShadowIntegrityState(
            head=head,
            index_hash=index_hash,
            status_hash=status_hash,
            tracked_manifest_hash=manifest_hash,
            tracked_count=tracked_count,
            proof_capable=(
                not issues
                and head is not None
                and index_hash is not None
                and status_hash is not None
                and manifest_hash is not None
            ),
            issues=tuple(issues),
            component_sources=tuple(component_sources),
        )

    def _collect_branch(self) -> EvidenceValue[str]:
        source = "git:branch --show-current"
        value = self._collect_required_text(
            ["git", "branch", "--show-current"],
            source,
        )
        if value.status.value == "known" and value.value == "":
            return EvidenceValue.known("DETACHED", source=source)
        return value

    def _collect_tags(self) -> EvidenceValue[list[str]]:
        source = "git:tag --points-at HEAD"
        self.evidence_sources.add(source)
        result = self.run(["git", "tag", "--points-at", "HEAD"])
        if not result.ok:
            reason = self._record_command_warning(source, result)
            return EvidenceValue.unavailable(reason=reason, source=source)
        text, replaced = result.stdout_text()
        if replaced:
            reason = f"commit log output is not valid UTF-8: {source}"
            self.warnings.append(reason)
            return EvidenceValue.unknown(reason=reason, source=source)
        return EvidenceValue.known(sorted(text.splitlines()), source=source)

    def _collect_remote_count(self) -> EvidenceValue[int]:
        source = "git-common-dir/config"
        try:
            common_directory = self.resolve_common_directory()
            config_path = common_directory / "config"
            self._validate_git_metadata_path(
                config_path,
                common_directory,
                "Git config",
            )
            source = f"git-common-dir/config:{config_path}"
            self.evidence_sources.add(source)
            with config_path.open("rb") as stream:
                data = stream.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                reason = "repository config exceeds the 1 MiB read limit"
                self.warnings.append(reason)
                return EvidenceValue.unknown(reason=reason, source=source)
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            reason = "repository config is not valid UTF-8"
            self.warnings.append(reason)
            return EvidenceValue.unknown(reason=reason, source=source)
        except (OSError, RuntimeError) as error:
            reason = f"repository config unavailable: {error}"
            self.warnings.append(reason)
            return EvidenceValue.unavailable(reason=reason, source=source)

        if re.search(r"(?im)^\s*\[\s*include(?:if)?(?:\s|\])", text):
            reason = "repository config contains include directives"
            self.warnings.append(reason)
            return EvidenceValue.unknown(reason=reason, source=source)
        remote_names = {
            match.group(1)
            for match in re.finditer(
                r'^\s*\[\s*remote\s+"([^"]+)"\s*\]\s*$',
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        }
        return EvidenceValue.known(len(remote_names), source=source)

    def _unsafe_repository_config_issues(self) -> list[str]:
        """Reject local configuration that can execute helpers or read elsewhere."""

        try:
            common_directory = self.resolve_common_directory()
            git_directory = self.resolve_git_directory()
        except (OSError, RuntimeError) as error:
            return [f"unsafe Git config: metadata location unavailable: {error}"]

        candidates = [(common_directory / "config", common_directory)]
        worktree_config = git_directory / "config.worktree"
        if worktree_config != candidates[0][0]:
            candidates.append((worktree_config, git_directory))

        issues: list[str] = []
        for config_path, allowed_root in candidates:
            source = f"git-config-safety:{config_path}"
            self.evidence_sources.add(source)
            if not config_path.exists():
                continue
            try:
                self._validate_git_metadata_path(
                    config_path,
                    allowed_root,
                    "Git config",
                )
                with config_path.open("rb") as stream:
                    data = stream.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise RuntimeError("config exceeds the 1 MiB safety limit")
                text = data.decode("utf-8", errors="strict")
            except (OSError, RuntimeError, UnicodeDecodeError) as error:
                issues.append(f"unsafe Git config: {config_path}: {error}")
                continue
            issues.extend(self._scan_unsafe_git_config(text, config_path))
        return issues

    def _unborn_head_marker(self) -> str | None:
        """Return a stable marker only for a readable symbolic unborn HEAD."""

        try:
            git_directory = self.resolve_git_directory()
            head_path = git_directory / "HEAD"
            self._validate_git_metadata_path(head_path, git_directory, "Git HEAD")
            with head_path.open("rb") as stream:
                data = stream.read(4097)
            if len(data) > 4096:
                return None
            text = data.decode("ascii", errors="strict").strip()
        except (OSError, RuntimeError, UnicodeDecodeError):
            return None
        match = re.fullmatch(r"ref:\s*(refs/[^\s]+)", text)
        if match is None:
            return None
        return f"UNBORN:{match.group(1)}"

    @staticmethod
    def _scan_unsafe_git_config(text: str, config_path: Path) -> list[str]:
        unsafe: list[str] = []
        section = ""
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            header = re.match(r"^\[\s*([^\]]+)\s*\]$", line)
            if header:
                section = header.group(1).strip().casefold()
                if section.startswith(("include", "includeif")):
                    unsafe.append(
                        f"unsafe Git config: include directive in {config_path}:"
                        f"{line_number}"
                    )
                continue
            key_match = re.match(r"^([A-Za-z][A-Za-z0-9-]*)\s*(?:=|\s)", line)
            if not key_match:
                continue
            key = key_match.group(1).casefold()
            dangerous = (
                section == "core"
                and key in {"fsmonitor", "attributesfile", "excludesfile"}
            ) or (
                section.startswith('filter "')
                and key in {"clean", "smudge", "process"}
            ) or (
                section.startswith('diff "')
                and key in {"command", "textconv"}
            ) or (
                section == "diff" and key == "external"
            )
            if dangerous:
                unsafe.append(
                    f"unsafe Git config: {section}.{key} in "
                    f"{config_path}:{line_number}"
                )
        return unsafe

    @staticmethod
    def _validate_git_metadata_path(
        path: Path,
        allowed_root: Path,
        label: str,
    ) -> None:
        root = allowed_root.resolve(strict=True)
        candidate = Path(os.path.abspath(path))
        if not candidate.is_relative_to(root):
            raise RuntimeError(f"{label} escapes its Git metadata root: {candidate}")
        current = root
        for part in candidate.relative_to(root).parts:
            current = current / part
            try:
                metadata = os.lstat(current)
            except FileNotFoundError:
                continue
            attributes = getattr(metadata, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag):
                raise RuntimeError(f"{label} uses a link or reparse point: {current}")

    def _collect_path_count(
        self,
        argv: list[str],
        source: str,
    ) -> EvidenceValue[int]:
        self.evidence_sources.add(source)
        result = self.run(argv)
        if not result.ok:
            reason = self._record_command_warning(source, result)
            return EvidenceValue.unavailable(reason=reason, source=source)
        text, replaced = result.stdout_text()
        if replaced:
            self.warnings.append(f"UTF-8 replacement used for {source}")
        count = len([path for path in text.split("\x00") if path])
        return EvidenceValue.known(count, source=source)

    def _collect_recent_commits(self) -> EvidenceValue[list[CommitRecord]]:
        source = "git:log -n 10"
        self.evidence_sources.add(source)
        result = self.run(
            [
                "git",
                "log",
                "-n",
                "10",
                "--date=iso-strict",
                "--pretty=format:%H%x1f%h%x1f%aI%x1f%s%x1e",
            ]
        )
        if not result.ok:
            reason = self._record_command_warning(source, result)
            return EvidenceValue.unavailable(reason=reason, source=source)
        text, replaced = result.stdout_text()
        if replaced:
            reason = f"commit log output is not valid UTF-8: {source}"
            self.warnings.append(reason)
            return EvidenceValue.unknown(reason=reason, source=source)
        records: list[CommitRecord] = []
        for raw_record in text.split("\x1e"):
            record = raw_record.strip("\r\n")
            if not record:
                continue
            parts = record.split("\x1f", 3)
            if len(parts) != 4:
                reason = f"unparseable commit record from {source}"
                self.warnings.append(reason)
                return EvidenceValue.unknown(reason=reason, source=source)
            sha, short_sha, authored_at, subject = parts
            records.append(
                CommitRecord(
                    sha=EvidenceValue.known(sha, source=source),
                    short_sha=EvidenceValue.known(short_sha, source=source),
                    authored_at=EvidenceValue.known(authored_at, source=source),
                    subject=EvidenceValue.known(subject, source=source),
                )
            )
        return EvidenceValue.known(records, source=source)

    def _collect_required_text(
        self,
        argv: list[str],
        source: str,
        *,
        strip: bool = True,
    ) -> EvidenceValue[str]:
        self.evidence_sources.add(source)
        result = self.run(argv)
        if not result.ok:
            reason = self._record_command_warning(source, result)
            return EvidenceValue.unavailable(reason=reason, source=source)
        text, replaced = result.stdout_text()
        if replaced:
            self.warnings.append(f"UTF-8 replacement used for {source}")
        return EvidenceValue.known(text.strip() if strip else text, source=source)

    def _required_command_text(self, argv: list[str], source: str) -> str:
        self.evidence_sources.add(source)
        result = self.run(argv)
        if not result.ok:
            raise RuntimeError(self._record_command_warning(source, result))
        text, replaced = result.stdout_text()
        if replaced:
            self.warnings.append(f"UTF-8 replacement used for {source}")
        value = text.strip()
        if not value:
            raise RuntimeError(f"empty output from {source}")
        return value

    def _integrity_text(
        self,
        argv: list[str],
        source: str,
        issues: list[str],
    ) -> str | None:
        self.evidence_sources.add(source)
        result = self.run(argv)
        if not result.ok:
            issues.append(self._command_failure_reason(source, result))
            return None
        text, replaced = result.stdout_text()
        if replaced:
            issues.append(f"UTF-8 replacement used for {source}")
        value = text.strip()
        if not value:
            issues.append(f"empty output from {source}")
            return None
        return value

    def _resolve_git_reported_path(self, text: str) -> Path:
        return self._absolute_git_reported_path(text).resolve(strict=False)

    def _absolute_git_reported_path(self, text: str) -> Path:
        path = Path(text)
        if not path.is_absolute():
            path = self.project_root / path
        return Path(os.path.abspath(path))

    def _record_command_warning(
        self,
        source: str,
        result: GitCommandResult,
    ) -> str:
        reason = self._command_failure_reason(source, result)
        self.warnings.append(reason)
        return reason

    @staticmethod
    def _command_failure_reason(source: str, result: GitCommandResult) -> str:
        if result.timed_out:
            return f"Git command timed out: {source}"
        stderr, _ = result.stderr_text()
        detail = " ".join(stderr.strip().split())[:300]
        if detail:
            return f"Git command failed: {source}: {detail}"
        return f"Git command failed: {source} (exit {result.returncode})"

    @staticmethod
    def _non_repository_evidence(source: str) -> GitEvidence:
        def missing(field_source: str) -> EvidenceValue:
            return EvidenceValue.unavailable(
                reason="project is not a Git repository",
                source=field_source,
            )

        return GitEvidence(
            is_repository=EvidenceValue.known(False, source=source),
            branch=missing("git:branch --show-current"),
            head_sha=missing("git:rev-parse HEAD"),
            head_short=missing("git:rev-parse --short=12 HEAD"),
            tags_at_head=missing("git:tag --points-at HEAD"),
            remote_count=missing("git-common-dir/config"),
            staged_count=missing("git:diff --cached --name-only -z"),
            modified_count=missing("git:diff --name-only -z"),
            untracked_count=missing(
                "git:ls-files --others --exclude-standard -z"
            ),
            recent_commits=missing("git:log -n 10"),
        )
