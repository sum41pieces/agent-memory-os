"""Contained filesystem reads and deterministic SHA-256 manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable


class UnsafePathError(ValueError):
    """Raised when a requested source path escapes its declared root."""


@dataclass(frozen=True)
class TextReadResult:
    path: Path
    text: str
    truncated: bool
    decode_replaced: bool


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    kind: str
    size: int
    sha256: str


@dataclass(frozen=True)
class TrackedManifest:
    aggregate_sha256: str
    tracked_count: int
    proof_capable: bool
    issues: tuple[str, ...]
    entries: tuple[ManifestEntry, ...]


class FilesystemAdapter:
    """Read files without allowing path traversal outside a declared root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def resolve_inside(self, path: Path) -> Path:
        candidate = path.resolve(strict=False) if path.is_absolute() else (
            self.root / path
        ).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise UnsafePathError(str(path))
        return candidate

    def read_text(self, relative_path: Path, *, max_bytes: int) -> TextReadResult:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        path = self.resolve_inside(relative_path)
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        payload = data[:max_bytes]
        try:
            text = payload.decode("utf-8", errors="strict")
            decode_replaced = False
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            decode_replaced = True
        return TextReadResult(
            path=path,
            text=text,
            truncated=truncated,
            decode_replaced=decode_replaced,
        )

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def hash_tracked_manifest(self, git_paths: Iterable[str]) -> TrackedManifest:
        entries: list[ManifestEntry] = []
        issues: list[str] = []
        normalized_paths = sorted(
            {path.replace("\\", "/") for path in git_paths}
        )

        for normalized in normalized_paths:
            pure = PurePosixPath(normalized)
            if pure.is_absolute() or ".." in pure.parts or not normalized:
                issues.append(f"unsafe tracked path: {normalized}")
                entries.append(
                    ManifestEntry(normalized, "unsafe", -1, "")
                )
                continue

            platform_relative = Path(*pure.parts)
            candidate = Path(os.path.abspath(self.root / platform_relative))
            if not candidate.is_relative_to(self.root):
                issues.append(f"unsafe tracked path: {normalized}")
                entries.append(
                    ManifestEntry(normalized, "unsafe", -1, "")
                )
                continue

            unsafe_link = self._intermediate_link(candidate)
            if unsafe_link is not None:
                issues.append(
                    f"unsafe tracked path (intermediate link): {normalized}: "
                    f"{unsafe_link}"
                )
                entries.append(ManifestEntry(normalized, "unsafe", -1, ""))
                continue

            try:
                final_stat = os.lstat(candidate)
            except FileNotFoundError:
                entries.append(ManifestEntry(normalized, "missing", 0, ""))
                continue
            except OSError as error:
                issues.append(f"unreadable tracked path: {normalized}: {error}")
                entries.append(ManifestEntry(normalized, "unreadable", -1, ""))
                continue

            if stat.S_ISLNK(final_stat.st_mode):
                try:
                    link_bytes = os.fsencode(os.readlink(candidate))
                except OSError as error:
                    issues.append(f"unreadable tracked path: {normalized}: {error}")
                    entries.append(
                        ManifestEntry(normalized, "unreadable", -1, "")
                    )
                    continue
                entries.append(
                    ManifestEntry(
                        normalized,
                        "symlink",
                        len(link_bytes),
                        hashlib.sha256(link_bytes).hexdigest(),
                    )
                )
            elif self._is_reparse_point(final_stat):
                issues.append(f"unsafe tracked path (reparse point): {normalized}")
                entries.append(ManifestEntry(normalized, "unsafe", -1, ""))
            elif stat.S_ISREG(final_stat.st_mode):
                try:
                    resolved_candidate = candidate.resolve(strict=True)
                    if not resolved_candidate.is_relative_to(self.root):
                        raise UnsafePathError(normalized)
                    size = final_stat.st_size
                    digest = self.hash_file(resolved_candidate)
                except OSError as error:
                    issues.append(f"unreadable tracked path: {normalized}: {error}")
                    entries.append(
                        ManifestEntry(normalized, "unreadable", -1, "")
                    )
                except UnsafePathError:
                    issues.append(f"unsafe tracked path: {normalized}")
                    entries.append(ManifestEntry(normalized, "unsafe", -1, ""))
                else:
                    entries.append(
                        ManifestEntry(normalized, "file", size, digest)
                    )
            else:
                issues.append(f"unsupported tracked path: {normalized}")
                entries.append(
                    ManifestEntry(normalized, "unsupported", -1, "")
                )

        aggregate = hashlib.sha256()
        for entry in entries:
            aggregate.update(entry.path.encode("utf-8"))
            aggregate.update(b"\x00")
            aggregate.update(entry.kind.encode("ascii"))
            aggregate.update(b"\x00")
            aggregate.update(str(entry.size).encode("ascii"))
            aggregate.update(b"\x00")
            aggregate.update(entry.sha256.encode("ascii"))
            aggregate.update(b"\x00")

        return TrackedManifest(
            aggregate_sha256=aggregate.hexdigest(),
            tracked_count=len(entries),
            proof_capable=not issues,
            issues=tuple(issues),
            entries=tuple(entries),
        )

    def _intermediate_link(self, candidate: Path) -> Path | None:
        relative = candidate.relative_to(self.root)
        current = self.root
        for part in relative.parts[:-1]:
            current = current / part
            try:
                metadata = os.lstat(current)
            except FileNotFoundError:
                return None
            except OSError:
                return current
            if stat.S_ISLNK(metadata.st_mode) or self._is_reparse_point(metadata):
                return current
        return None

    @staticmethod
    def _is_reparse_point(metadata: os.stat_result) -> bool:
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = getattr(metadata, "st_file_attributes", 0)
        return bool(attributes & flag)
