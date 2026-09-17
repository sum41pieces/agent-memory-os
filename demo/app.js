"use strict";

const RECONCILIATION_URL =
  "../examples/sanitized-interview-agent/synthetic-reconciliation.json";
const EVIDENCE_URL =
  "../examples/sanitized-interview-agent/synthetic-evidence.json";
const STATUSES = ["active", "superseded", "conflicted", "pending", "deprecated"];
const ALL = "ALL";

const state = {
  reconciliation: null,
  evidence: null,
  facts: [],
  filter: ALL,
};

const elements = {
  filters: document.querySelector("#status-filters"),
  factGrid: document.querySelector("#fact-grid"),
  visibleCount: document.querySelector("#visible-count"),
  loadState: document.querySelector("#load-state"),
  projectName: document.querySelector("#project-name"),
  pathStory: document.querySelector("#path-story"),
  conflictStory: document.querySelector("#conflict-story"),
  pendingStory: document.querySelector("#pending-story"),
  deprecatedStory: document.querySelector("#deprecated-story"),
  drawer: document.querySelector("#provenance-drawer"),
  drawerTitle: document.querySelector("#drawer-title"),
  drawerBody: document.querySelector("#drawer-body"),
  drawerClose: document.querySelector("#drawer-close"),
  themeToggle: document.querySelector("#theme-toggle"),
  themeLabel: document.querySelector("#theme-label"),
};

function valueOf(wireValue, fallback = "Not available") {
  if (!wireValue || wireValue.status !== "known" || wireValue.value === null) {
    return fallback;
  }
  return wireValue.value;
}

function humanize(value) {
  return String(value || "Not available")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value) {
  if (value === true) return "Enabled";
  if (value === false) return "Disabled";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None";
  if (value && typeof value === "object") return JSON.stringify(value, null, 2);
  return value === undefined || value === null || value === "" ? "Not available" : String(value);
}

function shortId(value) {
  if (!value) return "Not available";
  const parts = String(value).split(":");
  const tail = parts.at(-1);
  return tail.length > 14 ? `${tail.slice(0, 12)}…` : tail;
}

function displayConfidence(confidence) {
  const value = valueOf(confidence, "Not available");
  return typeof value === "number" ? `${Math.round(value * 100)}%` : displayValue(value);
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function appendDetail(list, label, content, options = {}) {
  const row = createElement("div");
  row.append(createElement("dt", "", label));
  const value = createElement("dd");

  if (options.codes) {
    value.className = "code-list";
    const items = Array.isArray(content) ? [...content] : [content];
    if (!items.length) items.push("None");
    items.forEach((item) => value.append(createElement("code", "", displayValue(item))));
  } else {
    value.textContent = displayValue(content);
  }

  row.append(value);
  list.append(row);
}

function statusColor(status) {
  return `var(--${String(status).toLowerCase()}, var(--muted))`;
}

function renderFilters() {
  elements.filters.replaceChildren();
  const filters = [ALL, ...STATUSES];

  filters.forEach((filter) => {
    const count = filter === ALL
      ? state.facts.length
      : state.reconciliation[filter].length;
    const button = createElement("button", "status-filter");
    button.type = "button";
    button.dataset.filter = filter;
    button.classList.toggle("selected", state.filter === filter);
    button.setAttribute("aria-pressed", String(state.filter === filter));
    button.append(
      createElement("span", "filter-name", filter),
      createElement("span", "filter-count", String(count)),
    );
    button.addEventListener("click", () => {
      state.filter = filter;
      renderFilters();
      renderFacts();
      elements.filters.querySelector(`[data-filter="${filter}"]`)?.focus();
    });
    elements.filters.append(button);
  });
}

function renderFacts() {
  const visibleFacts = state.filter === ALL
    ? state.facts
    : state.reconciliation[state.filter];
  elements.visibleCount.textContent = String(visibleFacts.length);
  elements.factGrid.replaceChildren();

  if (!visibleFacts.length) {
    elements.factGrid.append(createElement("p", "empty-state", "No facts match this filter."));
    return;
  }

  visibleFacts.forEach((fact) => {
    const card = createElement("button", "fact-card");
    card.type = "button";
    card.style.setProperty("--status-color", statusColor(fact.status));
    card.setAttribute(
      "aria-label",
      `${fact.status}: ${humanize(fact.predicate)}, ${displayValue(valueOf(fact.selected_value))}. Subject ${humanize(fact.subject)}. View provenance.`,
    );

    const top = createElement("span", "fact-card-top");
    top.append(
      createElement("span", "fact-status", fact.status),
      createElement("span", "fact-index", shortId(fact.fact_id)),
    );

    const metadata = createElement("span", "fact-metadata");
    metadata.append(
      createElement("span", "", `Subject · ${humanize(fact.subject)}`),
      createElement("span", "", `Confidence · ${displayConfidence(fact.confidence)}`),
      createElement(
        "span",
        fact.requires_human_review ? "review-required" : "",
        `Human review · ${fact.requires_human_review ? "Required" : "Not required"}`,
      ),
    );

    card.append(
      top,
      createElement("span", "fact-predicate", humanize(fact.predicate)),
      createElement("strong", "fact-value", displayValue(valueOf(fact.selected_value))),
      metadata,
      createElement("span", "fact-reason", fact.reason || "No resolution reason recorded."),
    );
    card.addEventListener("click", () => openProvenance(fact));
    elements.factGrid.append(card);
  });
}

function relationNode(label, fact) {
  const node = createElement("div", "relation-node");
  node.append(createElement("span", "relation-label", label));
  if (fact) {
    const status = createElement("span", "fact-status", fact.status);
    status.style.setProperty("--status-color", statusColor(fact.status));
    node.append(status);
  }
  node.append(
    createElement("strong", "", fact ? displayValue(valueOf(fact.selected_value)) : "Not available"),
  );
  return node;
}

function relationMarker(desktopText, mobileText) {
  const marker = createElement("div", "relation-marker");
  marker.append(
    createElement("span", "desktop-marker", desktopText),
    createElement("span", "mobile-marker", mobileText),
  );
  return marker;
}

function factById(factId) {
  return state.facts.find((fact) => fact.fact_id === factId);
}

function renderPathStory() {
  const relation = state.reconciliation.relations.find((item) => {
    const from = factById(item.from_fact_id);
    return item.relation_type === "SUPERSEDES" && from?.predicate === "project-path";
  });

  elements.pathStory.replaceChildren();
  if (!relation) {
    elements.pathStory.append(createElement("p", "empty-state", "Project path relation not available."));
    return;
  }

  const currentFact = factById(relation.from_fact_id);
  const formerFact = factById(relation.to_fact_id);
  const chain = createElement("div", "relation-chain");
  chain.append(
    relationNode("Earlier value", formerFact),
    relationMarker(`${relation.relation_type} BY →`, `${relation.relation_type} BY ↓`),
    relationNode("Selected value", currentFact),
  );
  elements.pathStory.append(chain, createElement("p", "relation-note", relation.relation_reason));
}

function renderConflictStory() {
  const relation = state.reconciliation.relations.find(
    (item) => item.relation_type === "CONFLICTS",
  );

  elements.conflictStory.replaceChildren();
  if (!relation) {
    elements.conflictStory.append(createElement("p", "empty-state", "Conflict relation not available."));
    return;
  }

  const leftFact = factById(relation.from_fact_id);
  const rightFact = factById(relation.to_fact_id);
  const pair = createElement("div", "conflict-pair");
  pair.append(
    relationNode("Candidate A", leftFact),
    relationMarker(`← ${relation.relation_type} →`, `↕ ${relation.relation_type}`),
    relationNode("Candidate B", rightFact),
  );
  const reviewRequired = leftFact?.requires_human_review || rightFact?.requires_human_review;
  elements.conflictStory.append(
    pair,
    createElement(
      "p",
      "relation-note",
      `${relation.relation_reason}. Human review is ${reviewRequired ? "required" : "not required"}; the viewer does not choose a winner.`,
    ),
  );
}

function renderSingleStory(target, fact) {
  target.replaceChildren();
  if (!fact) {
    target.append(createElement("p", "empty-state", "Fact not available."));
    return;
  }

  const details = createElement("dl", "story-meta");
  appendDetail(details, "Predicate", humanize(fact.predicate));
  appendDetail(details, "Value", valueOf(fact.selected_value));
  appendDetail(details, "Reason", fact.reason);
  appendDetail(details, "Valid from", valueOf(fact.valid_from));
  target.append(details);
}

function renderStories() {
  renderPathStory();
  renderConflictStory();
  renderSingleStory(elements.pendingStory, state.reconciliation.pending[0]);
  renderSingleStory(elements.deprecatedStory, state.reconciliation.deprecated[0]);
}

function openProvenance(fact) {
  elements.drawerTitle.textContent = humanize(fact.predicate);
  elements.drawerBody.replaceChildren();

  const summary = createElement("div", "drawer-summary");
  const status = createElement("span", "fact-status", fact.status);
  status.style.setProperty("--status-color", statusColor(fact.status));
  summary.append(status, createElement("h3", "", displayValue(valueOf(fact.selected_value))));

  const details = createElement("dl", "detail-list");
  appendDetail(details, "Fact ID", fact.fact_id, { codes: true });
  appendDetail(details, "Resolution", humanize(fact.resolution_method));
  appendDetail(details, "Reason", fact.reason);
  appendDetail(details, "Human review", fact.requires_human_review ? "Required" : "Not required");
  appendDetail(details, "Valid from", valueOf(fact.valid_from));
  appendDetail(details, "Resolved at", valueOf(fact.resolved_at));
  appendDetail(details, "Candidates", fact.candidate_ids || [], { codes: true });
  appendDetail(details, "Activation witness", fact.activation_witness_candidate_ids || [], { codes: true });
  appendDetail(details, "Relations", fact.relation_ids || [], { codes: true });
  appendDetail(details, "Evidence refs", fact.evidence_refs || [], { codes: true });
  appendDetail(details, "Selected value source", fact.selected_value?.source);

  const relatedRelations = state.reconciliation.relations.filter(
    (relation) => relation.from_fact_id === fact.fact_id || relation.to_fact_id === fact.fact_id,
  );
  appendDetail(
    details,
    "Relation detail",
    relatedRelations.map((relation) => `${relation.relation_type} · ${shortId(relation.relation_id)}`),
    { codes: true },
  );

  elements.drawerBody.append(summary, details);
  elements.drawer.showModal();
}

function initializeTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem("demo-theme");
  } catch (error) {
    saved = null;
  }
  if (saved === "light" || saved === "dark") {
    document.documentElement.dataset.theme = saved;
  }
  updateThemeLabel();
}

function updateThemeLabel() {
  const explicit = document.documentElement.dataset.theme;
  const effective = explicit || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  elements.themeLabel.textContent = explicit ? `Theme: ${effective}` : `Theme: system (${effective})`;
}

function toggleTheme() {
  const explicit = document.documentElement.dataset.theme;
  const current = explicit || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("demo-theme", next);
  } catch (error) {
    // Theme switching remains functional when storage is disabled.
  }
  updateThemeLabel();
}

async function loadDemo() {
  if (location.protocol === "file:") {
    throw new Error("This demo must be served over HTTP. See demo/README.md for the local command.");
  }

  const [reconciliationResponse, evidenceResponse] = await Promise.all([
    fetch(RECONCILIATION_URL),
    fetch(EVIDENCE_URL),
  ]);

  if (!reconciliationResponse.ok || !evidenceResponse.ok) {
    throw new Error("The public synthetic data could not be loaded.");
  }

  state.reconciliation = await reconciliationResponse.json();
  state.evidence = await evidenceResponse.json();
  state.facts = STATUSES.flatMap((status) => state.reconciliation[status] || []);

  elements.projectName.textContent = valueOf(state.evidence.project_id, state.reconciliation.project_id);
  renderFilters();
  renderFacts();
  renderStories();
  elements.loadState.classList.add("ready");
  elements.loadState.lastChild.textContent = ` Synthetic data loaded · ${state.facts.length} facts · ${state.reconciliation.relations.length} relations`;
}

function showLoadError(error) {
  elements.loadState.classList.add("error");
  elements.loadState.lastChild.textContent = ` ${error.message}`;
  elements.projectName.textContent = "Data unavailable";
}

elements.themeToggle.addEventListener("click", toggleTheme);
elements.drawerClose.addEventListener("click", () => elements.drawer.close());
elements.drawer.addEventListener("click", (event) => {
  if (event.target === elements.drawer) elements.drawer.close();
});
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", updateThemeLabel);

initializeTheme();
loadDemo().catch(showLoadError);
