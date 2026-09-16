from pathlib import Path

from agent_memory_os.adapters.docs_adapter import DocsAdapter
from agent_memory_os.evidence.models import EvidenceStatus


def test_docs_adapter_extracts_bounded_fixed_documents(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (tmp_path / "AGENTS.md").write_text("# Rules\nRead only.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# 项目说明\n简介。\n", encoding="utf-8")
    (docs_dir / "STATE.md").write_text(
        "# Current State\nlast_updated: 2026-09-14\n" + "内容" * 5000,
        encoding="utf-8",
    )
    (docs_dir / "EXTRA.md").write_text("# Must not be read\n", encoding="utf-8")

    adapter = DocsAdapter(tmp_path, max_bytes=4096, summary_chars=240)
    evidence = adapter.collect()

    assert evidence.discovered.value == ["AGENTS.md", "README.md", "docs/STATE.md"]
    assert evidence.state.value.title.value == "Current State"
    assert len(evidence.state.value.summary.value) <= 240
    assert evidence.state.value.last_updated.value == "2026-09-14"
    assert evidence.state.source == "docs:docs/STATE.md"
    assert evidence.state.value.title.source == "docs:docs/STATE.md"
    assert evidence.readme.value.title.value == "项目说明"
    assert evidence.agents.value.title.value == "Rules"
    assert evidence.readme.value.last_updated.status is EvidenceStatus.UNAVAILABLE
    assert evidence.next.status is EvidenceStatus.UNAVAILABLE
    assert "docs:docs/EXTRA.md" not in adapter.evidence_sources
    assert any("truncated" in warning for warning in adapter.warnings)


def test_missing_docs_are_explicitly_unavailable(tmp_path: Path) -> None:
    evidence = DocsAdapter(tmp_path).collect()
    assert evidence.discovered.value == []
    for item in (
        evidence.project_context,
        evidence.state,
        evidence.next,
        evidence.decisions,
        evidence.session_log,
        evidence.readme,
        evidence.agents,
    ):
        assert item.status is EvidenceStatus.UNAVAILABLE
        assert item.reason == "document does not exist"
        assert item.source.startswith("docs:")


def test_missing_heading_and_last_updated_are_not_guessed(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("plain body only", encoding="utf-8")
    record = DocsAdapter(tmp_path).collect().readme.value
    assert record.title.status is EvidenceStatus.UNAVAILABLE
    assert record.title.reason == "Markdown heading not present"
    assert record.last_updated.status is EvidenceStatus.UNAVAILABLE
    assert record.last_updated.reason == "last_updated not present"


def test_invalid_utf8_becomes_warning_not_guess(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_bytes(b"# title\n\xff")
    adapter = DocsAdapter(tmp_path)
    record = adapter.collect().readme.value
    assert "�" in record.summary.value
    assert any("UTF-8 replacement" in warning for warning in adapter.warnings)
