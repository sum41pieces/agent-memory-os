import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEMO = ROOT / "demo"
REQUIRED_FILES = (
    DEMO / "index.html",
    DEMO / "styles.css",
    DEMO / "app.js",
    DEMO / "README.md",
)
PUBLIC_FIXTURES = (
    ROOT / "examples" / "sanitized-interview-agent" / "synthetic-reconciliation.json",
    ROOT / "examples" / "sanitized-interview-agent" / "synthetic-evidence.json",
)


def read_demo_file(name: str) -> str:
    path = DEMO / name
    assert path.is_file(), f"missing demo file: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_demo_files_exist() -> None:
    for path in REQUIRED_FILES:
        assert path.is_file(), f"missing demo file: {path.relative_to(ROOT)}"


def test_html_references_local_assets_and_synthetic_labels() -> None:
    html = read_demo_file("index.html")
    assert 'href="styles.css"' in html
    assert 'src="app.js"' in html
    assert "SYNTHETIC DEMO" in html
    assert "NO PRIVATE PROJECT DATA" in html
    assert "Agent Memory OS" in html
    assert "Reconciliation Explorer" in html
    assert "Memory is not authority." in html
    assert "Authority Gate — Planned" in html
    assert "Dashboard — Implemented" not in html
    assert "Current State Model — Implemented" not in html
    assert "https://" not in html


def test_javascript_reads_public_synthetic_wire_data() -> None:
    script = read_demo_file("app.js")
    assert (
        "../examples/sanitized-interview-agent/synthetic-reconciliation.json"
        in script
    )
    assert "../examples/sanitized-interview-agent/synthetic-evidence.json" in script
    assert "fetch(" in script
    for status in ("active", "superseded", "conflicted", "pending", "deprecated"):
        assert f'"{status}"' in script
    for field in (
        "fact_id",
        "candidate_ids",
        "evidence_refs",
        "activation_witness_candidate_ids",
        "relation_ids",
        "resolution_method",
        "valid_from",
        "resolved_at",
        "reason",
    ):
        assert field in script
    assert "from_fact_id" in script
    assert "to_fact_id" in script
    assert "selected_value" in script
    assert "requires_human_review" in script
    assert "Not available" in script


def test_javascript_handles_filtering_theme_and_http_requirement() -> None:
    script = read_demo_file("app.js")
    assert "ALL" in script
    assert 'location.protocol === "file:"' in script
    assert "This demo must be served over HTTP." in script
    assert "prefers-color-scheme" in script
    assert "localStorage" in script
    assert "addEventListener" in script
    assert "React" not in script
    assert "Vue" not in script
    assert "npm" not in script


def test_public_fixture_contract_supports_demo_views() -> None:
    reconciliation = json.loads(PUBLIC_FIXTURES[0].read_text(encoding="utf-8"))
    evidence = json.loads(PUBLIC_FIXTURES[1].read_text(encoding="utf-8"))
    statuses = ("active", "superseded", "conflicted", "pending", "deprecated")
    facts = [fact for status in statuses for fact in reconciliation[status]]
    by_id = {fact["fact_id"]: fact for fact in facts}

    assert evidence["project_id"]["value"] == reconciliation["project_id"]
    assert {status: len(reconciliation[status]) for status in statuses} == {
        "active": 2,
        "superseded": 2,
        "conflicted": 2,
        "pending": 1,
        "deprecated": 1,
    }
    assert all(
        fact["status"] == status.upper()
        for status in statuses
        for fact in reconciliation[status]
    )
    assert all(
        relation["from_fact_id"] in by_id
        for relation in reconciliation["relations"]
    )
    assert all(
        relation["to_fact_id"] in by_id
        for relation in reconciliation["relations"]
    )
    assert any(
        relation["relation_type"] == "SUPERSEDES"
        and by_id[relation["from_fact_id"]]["predicate"] == "project-path"
        for relation in reconciliation["relations"]
    )
    assert any(
        relation["relation_type"] == "CONFLICTS"
        for relation in reconciliation["relations"]
    )
    assert all(
        fact["requires_human_review"]
        for fact in reconciliation["conflicted"]
    )


def test_demo_docs_describe_local_read_only_use() -> None:
    readme = read_demo_file("README.md")
    assert "py -3.11 -m http.server 8000" in readme
    assert "http://localhost:8000/demo/" in readme
    assert "read-only" in readme.lower()
    assert "synthetic" in readme.lower()


def test_demo_files_contain_no_private_identifiers() -> None:
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*REQUIRED_FILES, *PUBLIC_FIXTURES)
    )
    prohibited = (
        "C:" + "\\Users\\" + "me",
        "D:" + "\\obsidan",
        "Codex" + "MemoryVault",
        "agent-memory-" + "runtime",
        "repository-" + "backups",
        "\u7075\u7280",
        "AI" + "\u9762\u8bd5\u7cfb\u7edf",
        "1078297937" + "@" + "qq.com",
        "private" + " artifact",
        "agent-memory-os" + "@example.invalid",
    )
    for value in prohibited:
        assert value.casefold() not in content.casefold()


def test_root_readme_keeps_capability_boundaries_and_demo_placeholder() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Live demo" in readme
    assert "Portfolio demo viewer available under /demo." in readme
    assert "GitHub Pages URL will be added after review." in readme
    implemented, remainder = readme.split("### In progress", 1)
    in_progress, planned = remainder.split("### Planned", 1)
    assert "Current State Model" not in implemented
    assert "Current State Model" in in_progress
    assert "Dashboard" in planned
