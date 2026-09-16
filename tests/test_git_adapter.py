from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from agent_memory_os.adapters.git_adapter import GitAdapter, GitCommandResult
from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.safety.shadow_policy import PolicyViolation, ShadowProjectPolicy


def run_setup_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        check=True,
        shell=False,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "中文 Git 项目"
    root.mkdir()
    run_setup_git(root, "init", "-b", "main")
    (root / "README.md").write_text("# Synthetic\n", encoding="utf-8")
    (root / "staged.txt").write_text("baseline\n", encoding="utf-8")
    (root / "中文.txt").write_text("证据\n", encoding="utf-8")
    run_setup_git(root, "add", "README.md", "staged.txt", "中文.txt")
    run_setup_git(
        root,
        "-c",
        "user.name=AgentMemoryOSTest",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "synthetic baseline",
    )
    run_setup_git(root, "tag", "synthetic-v1")
    return root


def test_git_runner_uses_argv_no_shell_timeout_and_optional_locks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"true\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    GitAdapter(tmp_path, ShadowProjectPolicy()).run(
        ["git", "rev-parse", "--is-inside-work-tree"]
    )
    argv, kwargs = calls[0]
    assert isinstance(argv, list)
    assert kwargs.get("shell") is False
    assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert kwargs["env"]["GIT_PAGER"] == "cat"
    assert "GIT_CONFIG_NOSYSTEM" not in kwargs["env"]
    assert kwargs["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
    assert kwargs["env"]["GIT_CONFIG_COUNT"] == "7"
    assert kwargs["env"]["GIT_CONFIG_KEY_0"] == "core.fsmonitor"
    assert kwargs["env"]["GIT_CONFIG_VALUE_0"] == "false"
    assert kwargs["env"]["GIT_CONFIG_KEY_1"] == "core.untrackedCache"
    assert kwargs["env"]["GIT_CONFIG_VALUE_1"] == "false"
    assert kwargs["env"]["GIT_CONFIG_KEY_2"] == "core.hooksPath"
    assert kwargs["env"]["GIT_CONFIG_VALUE_2"] == os.devnull
    assert kwargs["env"]["GIT_CONFIG_KEY_3"] == "protocol.allow"
    assert kwargs["env"]["GIT_CONFIG_VALUE_3"] == "never"
    assert kwargs["env"]["GIT_CONFIG_KEY_4"] == "submodule.recurse"
    assert kwargs["env"]["GIT_CONFIG_VALUE_4"] == "false"
    assert kwargs["env"]["GIT_CONFIG_KEY_5"] == "diff.external"
    assert kwargs["env"]["GIT_CONFIG_VALUE_5"] == ""
    assert kwargs["env"]["GIT_CONFIG_KEY_6"] == "diff.trustExitCode"
    assert kwargs["env"]["GIT_CONFIG_VALUE_6"] == "false"
    assert kwargs["executable"] == str(
        Path(shutil.which("git")).resolve(strict=True)  # type: ignore[arg-type]
    )
    assert kwargs["timeout"] > 0
    assert kwargs["cwd"] == tmp_path.resolve()


def test_git_runner_scrubs_inherited_git_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run(argv, **kwargs):
        captured_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout=b"true\n", stderr=b"")

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker-controlled"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-work-tree"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    GitAdapter(tmp_path, ShadowProjectPolicy()).run(
        ["git", "rev-parse", "--is-inside-work-tree"]
    )
    assert "GIT_DIR" not in captured_environment
    assert "GIT_WORK_TREE" not in captured_environment


def test_shadow_local_git_executable_is_rejected_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "shadow"
    project.mkdir()
    (project / "git.exe").write_bytes(b"not a real executable")
    monkeypatch.chdir(project)
    monkeypatch.setattr(shutil, "which", lambda _name: "git.exe")
    called = False

    def forbidden_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Shadow-local Git must never execute")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    result = GitAdapter(project, ShadowProjectPolicy()).run(
        ["git", "rev-parse", "--is-inside-work-tree"]
    )
    assert result.ok is False
    assert b"trusted Git executable is unavailable" in result.stderr
    assert called is False


def test_git_runner_enforces_policy_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PolicyViolation):
        GitAdapter(tmp_path, ShadowProjectPolicy()).run(["git", "checkout", "main"])
    assert called is False


def test_non_git_directory_returns_known_false(tmp_path: Path) -> None:
    evidence = GitAdapter(tmp_path, ShadowProjectPolicy()).collect()
    assert evidence.is_repository.value is False
    assert evidence.is_repository.source == "git:rev-parse --is-inside-work-tree"
    assert evidence.head_sha.status is EvidenceStatus.UNAVAILABLE
    assert evidence.remote_count.status is EvidenceStatus.UNAVAILABLE


def test_repository_collection_reports_git_evidence(git_repo: Path) -> None:
    (git_repo / "README.md").write_text("# Modified\n", encoding="utf-8")
    (git_repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    (git_repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    run_setup_git(git_repo, "add", "staged.txt")
    run_setup_git(git_repo, "remote", "add", "origin", ".")

    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    evidence = adapter.collect()

    assert evidence.is_repository.value is True
    assert evidence.branch.value == "main"
    assert len(evidence.head_sha.value) == 40
    assert len(evidence.head_short.value) == 12
    assert evidence.tags_at_head.value == ["synthetic-v1"]
    assert evidence.remote_count.value == 1
    assert evidence.staged_count.value == 1
    assert evidence.modified_count.value == 1
    assert evidence.untracked_count.value == 1
    assert len(evidence.recent_commits.value) == 1
    assert evidence.recent_commits.value[0].subject.value == "synthetic baseline"
    assert "files changed" in adapter.collect_diff_stat().value
    changed = adapter.collect_changed_files().value
    assert {item.path.value for item in changed} == {"README.md", "staged.txt"}
    assert {item.status.value for item in changed} == {"M"}
    assert "git:rev-parse HEAD" in adapter.evidence_sources


def test_remote_count_is_unknown_when_repo_config_has_include(git_repo: Path) -> None:
    config_path = git_repo / ".git/config"
    with config_path.open("a", encoding="utf-8") as stream:
        stream.write("\n[include]\n\tpath = ../extra.gitconfig\n")
    evidence = GitAdapter(git_repo, ShadowProjectPolicy()).collect()
    assert evidence.remote_count.status is EvidenceStatus.UNKNOWN
    assert "include" in evidence.remote_count.reason.lower()


@pytest.mark.parametrize(
    "config_text",
    [
        '[include]\n\tpath = ../outside.config\n',
        '[core]\n\tfsmonitor = dangerous-helper.exe\n',
        '[filter "danger"]\n\tprocess = dangerous-helper.exe\n',
        '[filter "danger"]\n\tclean = dangerous-helper.exe\n',
        '[diff "danger"]\n\tcommand = dangerous-helper.exe\n',
        '[core]\n\tattributesFile = ../outside.attributes\n',
        '[core]\n\texcludesFile = ../outside.ignore\n',
    ],
)
def test_integrity_guard_rejects_execution_or_external_read_config_before_status(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_text: str,
) -> None:
    config_path = git_repo / ".git/config"
    with config_path.open("a", encoding="utf-8") as stream:
        stream.write("\n" + config_text)
    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    calls: list[tuple[str, ...]] = []
    original_run = adapter.run

    def recording_run(argv: list[str]):
        calls.append(tuple(argv))
        return original_run(argv)

    monkeypatch.setattr(adapter, "run", recording_run)
    state = adapter.capture_integrity_state()
    assert state.proof_capable is False
    assert any("unsafe Git config" in issue for issue in state.issues)
    assert not any(argv[1] == "status" for argv in calls)


def test_common_directory_and_index_path_support_linked_worktree(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "linked-worktree"
    run_setup_git(git_repo, "worktree", "add", "-b", "linked-test", str(worktree))
    adapter = GitAdapter(worktree, ShadowProjectPolicy())
    assert adapter.resolve_common_directory() == (git_repo / ".git").resolve()
    index_path = adapter.resolve_index_path()
    assert index_path.is_file()
    assert index_path != (git_repo / ".git/index").resolve()


def test_index_path_must_stay_inside_worktree_git_directory(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    original = adapter._required_command_text

    def redirected(argv: list[str], source: str) -> str:
        if argv == ["git", "rev-parse", "--git-path", "index"]:
            return str(tmp_path / "outside-index")
        return original(argv, source)

    monkeypatch.setattr(adapter, "_required_command_text", redirected)
    with pytest.raises(RuntimeError, match="escapes its Git metadata root"):
        adapter.resolve_index_path()


def test_integrity_capture_is_repeatable_and_does_not_change_index(
    git_repo: Path,
) -> None:
    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    before = adapter.capture_integrity_state()
    after = adapter.capture_integrity_state()
    assert before == after
    assert before.proof_capable is True
    assert before.head is not None
    assert before.index_hash is not None
    assert before.status_hash is not None
    assert before.tracked_manifest_hash is not None


def test_unborn_repository_has_stable_proof_capable_absence_markers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unborn"
    root.mkdir()
    run_setup_git(root, "init", "-b", "main")
    adapter = GitAdapter(root, ShadowProjectPolicy())
    before = adapter.capture_integrity_state()
    after = adapter.capture_integrity_state()
    assert before == after
    assert before.proof_capable is True
    assert before.head == "UNBORN:refs/heads/main"
    assert before.index_hash == "MISSING_INDEX"


def test_integrity_capture_detects_tracked_content_change(git_repo: Path) -> None:
    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    before = adapter.capture_integrity_state()
    (git_repo / "中文.txt").write_text("变化\n", encoding="utf-8")
    after = adapter.capture_integrity_state()
    assert before.head == after.head
    assert before.index_hash == after.index_hash
    assert before.status_hash != after.status_hash
    assert before.tracked_manifest_hash != after.tracked_manifest_hash


def test_git_adapter_uses_only_policy_allowlisted_subcommands(git_repo: Path) -> None:
    adapter = GitAdapter(git_repo, ShadowProjectPolicy())
    adapter.collect()
    adapter.capture_integrity_state()
    assert all(
        source.split(":", 1)[1].split(" ", 1)[0]
        in {
            "branch",
            "diff",
            "log",
            "ls-files",
            "rev-parse",
            "status",
            "tag",
        }
        for source in adapter.evidence_sources
        if source.startswith("git:")
    )


def test_incomplete_changed_file_record_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GitAdapter(tmp_path, ShadowProjectPolicy())
    monkeypatch.setattr(
        adapter,
        "run",
        lambda _argv: GitCommandResult(("git", "diff"), 0, b"M\0", b""),
    )
    evidence = adapter.collect_changed_files()
    assert evidence.status is EvidenceStatus.UNKNOWN
    assert "incomplete" in evidence.reason


def test_malformed_commit_record_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GitAdapter(tmp_path, ShadowProjectPolicy())
    monkeypatch.setattr(
        adapter,
        "run",
        lambda _argv: GitCommandResult(("git", "log"), 0, b"malformed\x1e", b""),
    )
    evidence = adapter._collect_recent_commits()
    assert evidence.status is EvidenceStatus.UNKNOWN
    assert "unparseable" in evidence.reason
