"""Fixed-allowlist, bounded project document discovery."""

from __future__ import annotations

from pathlib import Path
import re

from agent_memory_os.adapters.filesystem_adapter import (
    FilesystemAdapter,
    UnsafePathError,
)
from agent_memory_os.evidence.models import (
    DocsEvidence,
    DocumentRecord,
    EvidenceValue,
)


DOCUMENTS = {
    "agents": Path("AGENTS.md"),
    "readme": Path("README.md"),
    "project_context": Path("docs/PROJECT_CONTEXT.md"),
    "state": Path("docs/STATE.md"),
    "next": Path("docs/NEXT.md"),
    "decisions": Path("docs/DECISIONS.md"),
    "session_log": Path("docs/SESSION_LOG.md"),
}


class DocsAdapter:
    """Discover and summarize only the approved project documents."""

    def __init__(
        self,
        project_root: Path,
        *,
        max_bytes: int = 64 * 1024,
        summary_chars: int = 500,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.filesystem = FilesystemAdapter(self.project_root)
        self.max_bytes = max_bytes
        self.summary_chars = summary_chars
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.evidence_sources: set[str] = {"docs:discovery"}

    def collect(self) -> DocsEvidence:
        discovered: list[str] = []
        records: dict[str, EvidenceValue[DocumentRecord]] = {}
        for name, relative_path in DOCUMENTS.items():
            source = f"docs:{relative_path.as_posix()}"
            self.evidence_sources.add(source)
            try:
                absolute_path = self.filesystem.resolve_inside(relative_path)
            except UnsafePathError as error:
                records[name] = EvidenceValue.unavailable(
                    reason=f"unsafe document path: {error}",
                    source=source,
                )
                continue
            if not absolute_path.is_file():
                records[name] = EvidenceValue.unavailable(
                    reason="document does not exist",
                    source=source,
                )
                continue
            discovered.append(relative_path.as_posix())
            records[name] = self._read_document(relative_path, source)

        return DocsEvidence(
            discovered=EvidenceValue.known(
                sorted(discovered),
                source="docs:discovery",
            ),
            agents=records["agents"],
            project_context=records["project_context"],
            state=records["state"],
            next=records["next"],
            decisions=records["decisions"],
            session_log=records["session_log"],
            readme=records["readme"],
        )

    def _read_document(
        self,
        relative_path: Path,
        source: str,
    ) -> EvidenceValue[DocumentRecord]:
        try:
            result = self.filesystem.read_text(
                relative_path,
                max_bytes=self.max_bytes,
            )
        except (OSError, UnsafePathError) as error:
            message = f"document read failed: {relative_path.as_posix()}: {error}"
            self.warnings.append(message)
            return EvidenceValue.unavailable(reason=message, source=source)

        if result.truncated:
            self.warnings.append(
                f"document truncated at {self.max_bytes} bytes: "
                f"{relative_path.as_posix()}"
            )
        if result.decode_replaced:
            self.warnings.append(
                f"UTF-8 replacement used for {relative_path.as_posix()}"
            )

        heading_match = re.search(
            r"(?m)^\s{0,3}#\s+(.+?)\s*$",
            result.text,
        )
        if heading_match:
            title = EvidenceValue.known(heading_match.group(1).strip(), source=source)
        else:
            title = EvidenceValue.unavailable(
                reason="Markdown heading not present",
                source=source,
            )

        updated_match = re.search(
            r"(?im)^\s*last_updated\s*:\s*['\"]?([^'\"#\r\n]+)",
            result.text,
        )
        if updated_match:
            last_updated = EvidenceValue.known(
                updated_match.group(1).strip(),
                source=source,
            )
        else:
            last_updated = EvidenceValue.unavailable(
                reason="last_updated not present",
                source=source,
            )

        summary = re.sub(r"\s+", " ", result.text).strip()[: self.summary_chars]
        record = DocumentRecord(
            path=EvidenceValue.known(relative_path.as_posix(), source=source),
            title=title,
            summary=EvidenceValue.known(summary, source=source),
            last_updated=last_updated,
        )
        return EvidenceValue.known(record, source=source)
