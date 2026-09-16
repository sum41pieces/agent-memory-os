import os
from pathlib import Path
import subprocess

import pytest

from agent_memory_os.adapters.filesystem_adapter import (
    FilesystemAdapter,
    UnsafePathError,
)


def test_bounded_text_read_preserves_chinese_and_reports_truncation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "中文项目"
    root.mkdir()
    (root / "README.md").write_text("# 标题\n" + "内容" * 100, encoding="utf-8")
    result = FilesystemAdapter(root).read_text(Path("README.md"), max_bytes=32)
    assert result.text.startswith("# 标题")
    assert result.truncated is True


def test_read_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(UnsafePathError):
        FilesystemAdapter(root).read_text(Path("../outside.txt"), max_bytes=64)


def test_read_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(UnsafePathError):
        FilesystemAdapter(root).read_text(outside, max_bytes=64)


def test_hash_file_returns_sha256(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "a.txt"
    target.write_bytes(b"agent-memory-os")
    assert FilesystemAdapter(root).hash_file(target) == (
        "486d2a7eb19bd15cc4be764dee5a6fc2b6ff4b05c0db06c1b70e57332ccb731e"
    )


def test_manifest_digest_changes_when_tracked_bytes_change(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "a.txt"
    target.write_bytes(b"one")
    adapter = FilesystemAdapter(root)
    first = adapter.hash_tracked_manifest(["a.txt"])
    target.write_bytes(b"two")
    second = adapter.hash_tracked_manifest(["a.txt"])
    assert first.aggregate_sha256 != second.aggregate_sha256
    assert first.tracked_count == second.tracked_count == 1
    assert first.proof_capable is True
    assert second.proof_capable is True


def test_manifest_is_deterministic_for_path_order(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_bytes(b"a")
    (root / "b.txt").write_bytes(b"b")
    adapter = FilesystemAdapter(root)
    first = adapter.hash_tracked_manifest(["b.txt", "a.txt"])
    second = adapter.hash_tracked_manifest(["a.txt", "b.txt"])
    assert first == second


def test_missing_tracked_path_has_stable_proof_marker(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    manifest = FilesystemAdapter(root).hash_tracked_manifest(["deleted.txt"])
    assert manifest.proof_capable is True
    assert manifest.tracked_count == 1
    assert manifest.entries[0].kind == "missing"


def test_unsafe_git_path_prevents_proof(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    manifest = FilesystemAdapter(root).hash_tracked_manifest(["../outside.txt"])
    assert manifest.proof_capable is False
    assert manifest.issues == ("unsafe tracked path: ../outside.txt",)


def test_directory_in_tracked_paths_prevents_proof(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "folder").mkdir(parents=True)
    manifest = FilesystemAdapter(root).hash_tracked_manifest(["folder"])
    assert manifest.proof_capable is False
    assert manifest.entries[0].kind == "unsupported"


def test_manifest_rejects_intermediate_link_without_reading_outside(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("must not be hashed", encoding="utf-8")
    linked = root / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        created = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "New-Item -ItemType Junction "
                    f"-Path '{str(linked).replace(chr(39), chr(39) * 2)}' "
                    f"-Target '{str(outside).replace(chr(39), chr(39) * 2)}'"
                ),
            ],
            capture_output=True,
            check=False,
            shell=False,
        )
        if created.returncode != 0:
            pytest.skip("directory links are unavailable")

    manifest = FilesystemAdapter(root).hash_tracked_manifest(
        ["linked/secret.txt"]
    )
    assert manifest.proof_capable is False
    assert manifest.entries[0].kind == "unsafe"
    assert "intermediate link" in manifest.issues[0]
