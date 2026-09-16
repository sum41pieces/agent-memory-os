from dataclasses import FrozenInstanceError

import pytest

from agent_memory_os.reconcile.models import ReconciliationInputError
from agent_memory_os.reconcile.providers import (
    InMemoryCandidateProvider,
    MemoryCandidateProvider,
)

from reconciliation_helpers import make_candidate


def test_in_memory_provider_is_importable() -> None:
    assert InMemoryCandidateProvider is not None


@pytest.mark.parametrize("project_id", ["", "  ", "bad\x00project"])
def test_provider_rejects_invalid_project_id(project_id: str) -> None:
    with pytest.raises(ReconciliationInputError, match="project_id"):
        InMemoryCandidateProvider(project_id, [])


def test_provider_rejects_duplicate_candidate_ids() -> None:
    candidates = [
        make_candidate(candidate_id="duplicate"),
        make_candidate(candidate_id="duplicate", value="different"),
    ]

    with pytest.raises(ReconciliationInputError, match="duplicate"):
        InMemoryCandidateProvider("project", candidates)


def test_provider_returns_candidates_in_deterministic_id_order() -> None:
    provider = InMemoryCandidateProvider(
        "project",
        [
            make_candidate(candidate_id="z"),
            make_candidate(candidate_id="a"),
            make_candidate(candidate_id="m"),
        ],
    )

    assert tuple(item.candidate_id for item in provider.candidates_for("project")) == (
        "a",
        "m",
        "z",
    )


def test_provider_filters_by_exact_project_id() -> None:
    candidate = make_candidate(candidate_id="c1")
    provider = InMemoryCandidateProvider("project", [candidate])

    assert provider.candidates_for("other-project") == ()
    assert provider.candidates_for("project") == (candidate,)


def test_provider_defensively_copies_caller_candidate_list() -> None:
    first = make_candidate(candidate_id="first")
    candidates = [first]
    provider = InMemoryCandidateProvider("project", candidates)

    candidates.append(make_candidate(candidate_id="later"))

    assert provider.candidates_for("project") == (first,)


def test_provider_is_frozen_and_satisfies_protocol() -> None:
    provider: MemoryCandidateProvider = InMemoryCandidateProvider("project", [])

    with pytest.raises(FrozenInstanceError):
        provider.project_id = "changed"
