"""Evidence Collector orchestration and command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

from agent_memory_os.adapters.docs_adapter import DocsAdapter
from agent_memory_os.adapters.git_adapter import GitAdapter, ShadowIntegrityState
from agent_memory_os.adapters.tests_adapter import TestsAdapter
from agent_memory_os.evidence.models import (
    CollectorEvidence,
    EvidenceSnapshot,
    EvidenceStatus,
    EvidenceValue,
    RecentChangesEvidence,
    RepositoryEvidence,
    ShadowGuardReport,
    snapshot_to_json,
    utc_now_iso,
)
from agent_memory_os.safety.shadow_policy import PolicyViolation, ShadowProjectPolicy


GitAdapterFactory = Callable[[Path, ShadowProjectPolicy], GitAdapter]


@dataclass(frozen=True)
class IntegrityComparison:
    changed_components: tuple[str, ...]
    proof_capable: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class CollectionResult:
    verdict: str
    exit_code: int
    snapshot: EvidenceSnapshot | None
    output_path: Path | None
    safety_violation: bool
    changed_components: tuple[str, ...]
    issues: tuple[str, ...]
    console_lines: tuple[str, ...]


def compare_integrity_states(
    before: ShadowIntegrityState,
    after: ShadowIntegrityState,
) -> IntegrityComparison:
    """Compare only the four approved Shadow mutation indicators."""

    fields = (
        ("HEAD", before.head, after.head),
        ("INDEX", before.index_hash, after.index_hash),
        ("STATUS", before.status_hash, after.status_hash),
        (
            "TRACKED_CONTENT",
            before.tracked_manifest_hash,
            after.tracked_manifest_hash,
        ),
    )
    changed = tuple(
        name
        for name, left, right in fields
        if left is not None and right is not None and left != right
    )
    issues = tuple(
        [f"before: {issue}" for issue in before.issues]
        + [f"after: {issue}" for issue in after.issues]
    )
    return IntegrityComparison(
        changed_components=changed,
        proof_capable=before.proof_capable and after.proof_capable,
        issues=issues,
    )


class EvidenceCollector:
    """Collect evidence only between equal, proof-capable integrity guards."""

    def __init__(
        self,
        *,
        product_root: Path,
        policy: ShadowProjectPolicy | None = None,
        git_adapter_factory: GitAdapterFactory = GitAdapter,
    ) -> None:
        self.product_root = product_root.resolve(strict=False)
        self.artifacts_root = self.product_root / "artifacts"
        self.policy = policy or ShadowProjectPolicy()
        self.git_adapter_factory = git_adapter_factory

    def collect(
        self,
        *,
        project: Path,
        project_id: str,
        output: Path,
    ) -> CollectionResult:
        resolved_project = project.resolve(strict=False)
        resolved_output = (
            output.resolve(strict=False)
            if output.is_absolute()
            else (self.product_root / output).resolve(strict=False)
        )
        try:
            self.policy.validate_collection_request(
                source_mode="shadow_read_only",
                output_path=resolved_output,
                artifacts_root=self.artifacts_root,
                project_root=resolved_project,
            )
        except PolicyViolation as error:
            return self._failure(
                exit_code=2,
                issues=(str(error),),
                lines=(f"POLICY_VIOLATION={error}", "SHADOW_COLLECTION_FAIL"),
            )

        captured_at = utc_now_iso()
        git_adapter = self.git_adapter_factory(resolved_project, self.policy)
        before = git_adapter.capture_integrity_state()
        if not before.proof_capable:
            issues = before.issues or ("pre-collection proof is unavailable",)
            return self._failure(
                exit_code=3,
                issues=issues,
                lines=tuple(
                    [f"PROOF_UNAVAILABLE={issue}" for issue in issues]
                    + ["SHADOW_COLLECTION_FAIL"]
                ),
            )

        collection_error: Exception | None = None
        docs_adapter: DocsAdapter | None = None
        tests_adapter: TestsAdapter | None = None
        try:
            git_evidence = git_adapter.collect()
            changed_files = git_adapter.collect_changed_files()
            diff_stat = git_adapter.collect_diff_stat()
            docs_adapter = DocsAdapter(resolved_project)
            docs_evidence = docs_adapter.collect()
            tests_adapter = TestsAdapter(resolved_project)
            tests_evidence = tests_adapter.collect()
        except Exception as error:
            collection_error = error
        finally:
            try:
                after = git_adapter.capture_integrity_state()
            except Exception as error:
                after = ShadowIntegrityState(
                    head=None,
                    index_hash=None,
                    status_hash=None,
                    tracked_manifest_hash=None,
                    tracked_count=None,
                    proof_capable=False,
                    issues=(f"post-collection guard failed: {error}",),
                )

        comparison = compare_integrity_states(before, after)
        if comparison.changed_components:
            return self._failure(
                exit_code=4,
                safety_violation=True,
                changed_components=comparison.changed_components,
                issues=comparison.issues,
                lines=tuple(
                    ["SHADOW_SAFETY_VIOLATION"]
                    + [
                        f"CHANGED_COMPONENT={component}"
                        for component in comparison.changed_components
                    ]
                    + ["SHADOW_COLLECTION_FAIL"]
                ),
            )
        if not comparison.proof_capable:
            issues = comparison.issues or ("post-collection proof is unavailable",)
            return self._failure(
                exit_code=3,
                issues=issues,
                lines=tuple(
                    [f"PROOF_UNAVAILABLE={issue}" for issue in issues]
                    + ["SHADOW_COLLECTION_FAIL"]
                ),
            )
        if collection_error is not None:
            issue = f"collection failed: {collection_error}"
            return self._failure(
                exit_code=3,
                issues=(issue,),
                lines=(f"COLLECTION_FAILED={collection_error}", "SHADOW_COLLECTION_FAIL"),
            )

        assert docs_adapter is not None
        assert tests_adapter is not None
        guard = self._build_guard(before, after)
        warnings = sorted(
            set(
                git_adapter.warnings
                + docs_adapter.warnings
                + tests_adapter.warnings
            )
        )
        errors = sorted(
            set(git_adapter.errors + docs_adapter.errors + tests_adapter.errors)
        )
        recent_commit_summary = self._recent_commit_summary(git_evidence.recent_commits)
        collector_evidence = CollectorEvidence(
            warnings=EvidenceValue.known(warnings, source="collector:warnings"),
            errors=EvidenceValue.known(errors, source="collector:errors"),
            evidence_sources=EvidenceValue.known(
                [],
                source="collector:source-inventory",
            ),
            shadow_guard=EvidenceValue.known(
                guard,
                source="shadow_guard:comparison",
            ),
        )
        snapshot = EvidenceSnapshot(
            schema_version=EvidenceValue.known("1.0.0", source="collector:schema"),
            project_id=EvidenceValue.known(project_id, source="cli:--project-id"),
            captured_at=EvidenceValue.known(captured_at, source="collector:clock"),
            source_mode=EvidenceValue.known(
                "shadow_read_only",
                source="collector:mode",
            ),
            repository=RepositoryEvidence(
                path=EvidenceValue.known(str(resolved_project), source="cli:--project"),
                exists=EvidenceValue.known(
                    resolved_project.is_dir(),
                    source="filesystem:project-root",
                ),
                project_name=EvidenceValue.known(
                    resolved_project.name,
                    source="filesystem:project-root",
                ),
            ),
            git=git_evidence,
            docs=docs_evidence,
            tests=tests_evidence,
            recent_changes=RecentChangesEvidence(
                changed_files=changed_files,
                diff_stat=diff_stat,
                recent_commit_summary=recent_commit_summary,
            ),
            collector=collector_evidence,
        )
        inventory = sorted(
            _all_evidence_sources(snapshot.to_dict())
            | git_adapter.evidence_sources
            | docs_adapter.evidence_sources
            | tests_adapter.evidence_sources
        )
        snapshot = replace(
            snapshot,
            collector=replace(
                snapshot.collector,
                evidence_sources=EvidenceValue.known(
                    inventory,
                    source="collector:source-inventory",
                ),
            ),
        )

        try:
            self.policy.validate_collection_request(
                source_mode="shadow_read_only",
                output_path=resolved_output,
                artifacts_root=self.artifacts_root,
                project_root=resolved_project,
            )
        except PolicyViolation as error:
            return self._failure(
                exit_code=2,
                snapshot=snapshot,
                issues=(str(error),),
                lines=(f"POLICY_VIOLATION={error}", "SHADOW_COLLECTION_FAIL"),
            )

        try:
            self._write_snapshot_after_pass(snapshot, resolved_output)
        except OSError as error:
            return self._failure(
                exit_code=3,
                snapshot=snapshot,
                issues=(f"artifact write failed: {error}",),
                lines=(f"ARTIFACT_WRITE_FAILED={error}", "SHADOW_COLLECTION_FAIL"),
            )

        lines = (
            "HEAD_UNCHANGED=true",
            "INDEX_UNCHANGED=true",
            "STATUS_UNCHANGED=true",
            "TRACKED_CONTENT_UNCHANGED=true",
            f"EVIDENCE_JSON={resolved_output}",
            "SHADOW_COLLECTION_PASS",
        )
        return CollectionResult(
            verdict="SHADOW_COLLECTION_PASS",
            exit_code=0,
            snapshot=snapshot,
            output_path=resolved_output,
            safety_violation=False,
            changed_components=(),
            issues=(),
            console_lines=lines,
        )

    @staticmethod
    def _build_guard(
        before: ShadowIntegrityState,
        after: ShadowIntegrityState,
    ) -> ShadowGuardReport:
        comparison_source = "shadow_guard:comparison"
        if None in (
            before.head,
            after.head,
            before.index_hash,
            after.index_hash,
            before.status_hash,
            after.status_hash,
            before.tracked_manifest_hash,
            after.tracked_manifest_hash,
        ):
            raise ValueError("proof-capable guard contains a missing digest")
        before_sources = dict(before.component_sources)
        after_sources = dict(after.component_sources)
        return ShadowGuardReport(
            head_before=EvidenceValue.known(
                before.head,
                source=before_sources.get("HEAD", "git:rev-parse HEAD"),
            ),
            head_after=EvidenceValue.known(
                after.head,
                source=after_sources.get("HEAD", "git:rev-parse HEAD"),
            ),
            index_hash_before=EvidenceValue.known(
                before.index_hash,
                source=before_sources.get("INDEX", "git:index"),
            ),
            index_hash_after=EvidenceValue.known(
                after.index_hash,
                source=after_sources.get("INDEX", "git:index"),
            ),
            status_hash_before=EvidenceValue.known(
                before.status_hash,
                source=before_sources.get("STATUS", "git:status"),
            ),
            status_hash_after=EvidenceValue.known(
                after.status_hash,
                source=after_sources.get("STATUS", "git:status"),
            ),
            tracked_manifest_before=EvidenceValue.known(
                before.tracked_manifest_hash,
                source=before_sources.get(
                    "TRACKED_CONTENT",
                    "git:ls-files -z + filesystem:tracked-manifest-sha256",
                ),
            ),
            tracked_manifest_after=EvidenceValue.known(
                after.tracked_manifest_hash,
                source=after_sources.get(
                    "TRACKED_CONTENT",
                    "git:ls-files -z + filesystem:tracked-manifest-sha256",
                ),
            ),
            head_unchanged=EvidenceValue.known(True, source=comparison_source),
            index_unchanged=EvidenceValue.known(True, source=comparison_source),
            status_unchanged=EvidenceValue.known(True, source=comparison_source),
            tracked_content_unchanged=EvidenceValue.known(
                True,
                source=comparison_source,
            ),
            changed_components=EvidenceValue.known([], source=comparison_source),
            verdict=EvidenceValue.known(
                "SHADOW_COLLECTION_PASS",
                source=comparison_source,
            ),
        )

    @staticmethod
    def _recent_commit_summary(recent_commits: EvidenceValue) -> EvidenceValue[list[str]]:
        if recent_commits.status is not EvidenceStatus.KNOWN:
            if recent_commits.status is EvidenceStatus.UNKNOWN:
                return EvidenceValue.unknown(
                    reason=recent_commits.reason or "recent commits are unknown",
                    source=recent_commits.source,
                )
            return EvidenceValue.unavailable(
                reason=recent_commits.reason or "recent commits are unavailable",
                source=recent_commits.source,
            )
        summaries = [
            f"{record.short_sha.value} {record.subject.value}"
            for record in recent_commits.value
        ]
        return EvidenceValue.known(summaries, source=recent_commits.source)

    @staticmethod
    def _failure(
        *,
        exit_code: int,
        issues: tuple[str, ...],
        lines: tuple[str, ...],
        snapshot: EvidenceSnapshot | None = None,
        safety_violation: bool = False,
        changed_components: tuple[str, ...] = (),
    ) -> CollectionResult:
        return CollectionResult(
            verdict="SHADOW_COLLECTION_FAIL",
            exit_code=exit_code,
            snapshot=snapshot,
            output_path=None,
            safety_violation=safety_violation,
            changed_components=changed_components,
            issues=issues,
            console_lines=lines,
        )

    @staticmethod
    def _write_snapshot_after_pass(snapshot: EvidenceSnapshot, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output.name}.",
                suffix=".tmp",
                dir=output.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(snapshot_to_json(snapshot))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, output)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _all_evidence_sources(value: object) -> set[str]:
    sources: set[str] = set()
    if isinstance(value, dict):
        wire_status = value.get("status")
        if (
            isinstance(wire_status, str)
            and wire_status in {"known", "unknown", "unavailable"}
        ):
            source = value.get("source")
            if isinstance(source, str) and source:
                sources.add(source)
        for child in value.values():
            sources.update(_all_evidence_sources(child))
    elif isinstance(value, list):
        for child in value:
            sources.update(_all_evidence_sources(child))
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect read-only evidence with Shadow mutation proof."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    product_root: Path | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    root = product_root or Path(__file__).resolve().parents[3]
    result = EvidenceCollector(product_root=root).collect(
        project=arguments.project,
        project_id=arguments.project_id,
        output=arguments.output,
    )
    for line in result.console_lines:
        print(line)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
