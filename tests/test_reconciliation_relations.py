"""Canonical relation identity and immutable payload tests."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields, replace
import re
from threading import Event, Thread

import pytest

import agent_memory_os.reconcile.models as reconciliation_models
from agent_memory_os.evidence.models import EvidenceStatus
from agent_memory_os.reconcile.models import (
    FactDecision,
    ReconciledFact,
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationWarning,
    ReconciliationStatus,
    RelationGraph,
    RelationRecord,
    RelationType,
    ResolutionMethod,
    SourceType,
    UnresolvedCandidate,
    WarningCode,
)
from agent_memory_os.reconcile.reconciler import (
    _validate_supersedes_acyclic,
    materialize_relation_graph as _materialize_authenticated_graph,
    validate_relation_graph,
    validate_result_fields,
)
from agent_memory_os.reconcile.rules import (
    RelationRequest,
    group_candidates,
    make_relation_id,
    resolve_non_known_groups,
    resolve_predicate,
)


def materialize_relation_graph(facts, requests):
    """Route synthetic provisional facts through the private Task 19 builder."""

    snapshot = tuple(facts)
    if snapshot and all(type(fact) is FactDecision for fact in snapshot):
        return _materialize_authenticated_graph(snapshot, requests)
    return _test_only_materialize_graph(snapshot, tuple(requests))
from reconciliation_helpers import (
    NOW,
    _test_only_copy,
    _test_only_graph,
    _test_only_materialize_graph,
    corrupt_graph_missing_superseded_by_index,
    decisions_for,
    default_policy,
    graph_with_supersedes_edges,
    known,
    make_candidate,
    make_current_candidate,
    make_historical_candidate,
    provisional_fact,
    provisional_facts,
    supersedes_request,
    symmetric_conflict_requests,
    unknown,
)


def relation(
    relation_type: RelationType = RelationType.SUPERSEDES,
    from_fact_id: str = "fact:v1:new",
    to_fact_id: str = "fact:v1:old",
) -> RelationRecord:
    return RelationRecord.create(
        relation_type,
        from_fact_id,
        to_fact_id,
        "synthetic reason",
        ("candidate",),
        ("evidence",),
    )


def test_relation_identity_excludes_reason_and_provenance_payload() -> None:
    first = RelationRecord.create(
        RelationType.SUPERSEDES,
        "fact:v1:new",
        "fact:v1:old",
        "reason one",
        ("c1",),
        ("e1",),
    )
    enriched = RelationRecord.create(
        RelationType.SUPERSEDES,
        "fact:v1:new",
        "fact:v1:old",
        "reason two",
        ("c1", "c2"),
        ("e1", "e2"),
    )

    assert first.relation_id == enriched.relation_id
    assert first.relation_reason != enriched.relation_reason
    assert first.candidate_ids != enriched.candidate_ids
    assert first.evidence_refs != enriched.evidence_refs


def test_relation_identity_matches_exact_canonical_v1_hash() -> None:
    record = relation()

    assert record.relation_id == (
        "relation:v1:"
        "05c9d5f92247017a632a44fbf1059b4536002cc84c0809d5369ad0727ffae79a"
    )
    assert re.fullmatch(r"relation:v1:[0-9a-f]{64}", record.relation_id)
    assert record.relation_id == make_relation_id(
        RelationType.SUPERSEDES,
        "fact:v1:new",
        "fact:v1:old",
    )


def test_relation_identity_distinguishes_type_and_ordered_endpoints() -> None:
    records = (
        relation(),
        relation(RelationType.CONFLICTS),
        relation(from_fact_id="fact:v1:other"),
        relation(to_fact_id="fact:v1:other"),
        relation(
            from_fact_id="fact:v1:old",
            to_fact_id="fact:v1:new",
        ),
    )

    assert len({record.relation_id for record in records}) == len(records)


def test_relation_immutable_payload_is_sorted_deduplicated_and_copied() -> None:
    candidate_ids = ["z", "a", "z"]
    evidence_refs = ["evidence:z", "evidence:a", "evidence:z"]

    record = RelationRecord.create(
        RelationType.CONFLICTS,
        "fact:v1:left",
        "fact:v1:right",
        "synthetic conflict",
        candidate_ids,
        evidence_refs,
    )
    candidate_ids[:] = ["mutated"]
    evidence_refs.append("evidence:mutated")

    assert record.candidate_ids == ("a", "z")
    assert record.evidence_refs == ("evidence:a", "evidence:z")
    assert type(record.candidate_ids) is tuple
    assert type(record.evidence_refs) is tuple


def test_relation_immutable_payload_accepts_exact_tuple_inputs() -> None:
    record = RelationRecord.create(
        RelationType.SUPERSEDES,
        "fact:v1:new",
        "fact:v1:old",
        "synthetic reason",
        ("z", "a", "z"),
        ("evidence:z", "evidence:a", "evidence:z"),
    )

    assert record.candidate_ids == ("a", "z")
    assert record.evidence_refs == ("evidence:a", "evidence:z")


def test_relation_immutable_payload_snapshots_list_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_ids: list[object] = ["race-candidate"]
    validation_started = Event()
    caller_mutated = Event()
    original_validate = reconciliation_models._validate_required_text

    def mutate_caller_list() -> None:
        assert validation_started.wait(timeout=2)
        candidate_ids[0] = []
        caller_mutated.set()

    def synchronized_validate(value: object, field_name: str) -> None:
        if value == "race-candidate" and field_name == "candidate_ids":
            validation_started.set()
            assert caller_mutated.wait(timeout=2)
        original_validate(value, field_name)

    monkeypatch.setattr(
        reconciliation_models,
        "_validate_required_text",
        synchronized_validate,
    )
    mutator = Thread(target=mutate_caller_list)
    mutator.start()
    try:
        record = RelationRecord.create(
            RelationType.SUPERSEDES,
            "fact:v1:new",
            "fact:v1:old",
            "synthetic reason",
            candidate_ids,
            ["evidence"],
        )
    finally:
        mutator.join(timeout=2)

    assert not mutator.is_alive()
    assert candidate_ids == [[]]
    assert record.candidate_ids == ("race-candidate",)


def test_relation_immutable_payload_and_record_are_frozen() -> None:
    record = relation()

    with pytest.raises(FrozenInstanceError):
        record.relation_reason = "changed"
    with pytest.raises(TypeError):
        record.candidate_ids[0] = "changed"


def test_relation_to_dict_is_deterministic_and_json_compatible() -> None:
    record = RelationRecord.create(
        RelationType.CONFLICTS,
        "fact:v1:left",
        "fact:v1:right",
        "synthetic conflict",
        ["right", "left", "right"],
        ["evidence:right", "evidence:left", "evidence:right"],
    )

    assert record.to_dict() == {
        "relation_id": record.relation_id,
        "relation_type": "CONFLICTS",
        "from_fact_id": "fact:v1:left",
        "to_fact_id": "fact:v1:right",
        "relation_reason": "synthetic conflict",
        "candidate_ids": ["left", "right"],
        "evidence_refs": ["evidence:left", "evidence:right"],
    }
    assert record.to_dict() == record.to_dict()


def test_relation_to_dict_returns_mutation_isolated_payload_lists() -> None:
    record = relation()
    serialized = record.to_dict()
    candidate_ids = serialized["candidate_ids"]
    evidence_refs = serialized["evidence_refs"]
    assert isinstance(candidate_ids, list)
    assert isinstance(evidence_refs, list)

    candidate_ids.append("mutated")
    evidence_refs.clear()

    assert record.candidate_ids == ("candidate",)
    assert record.evidence_refs == ("evidence",)
    assert record.to_dict()["candidate_ids"] == ["candidate"]
    assert record.to_dict()["evidence_refs"] == ["evidence"]


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("from_fact_id", ""),
        ("from_fact_id", "  "),
        ("from_fact_id", "fact\x00new"),
        ("to_fact_id", ""),
        ("to_fact_id", "  "),
        ("to_fact_id", "fact\x00old"),
        ("relation_reason", ""),
        ("relation_reason", "  "),
        ("relation_reason", "reason\x00bad"),
        ("candidate_ids", ()),
        ("candidate_ids", []),
        ("candidate_ids", ("",)),
        ("candidate_ids", ("  ",)),
        ("candidate_ids", ("candidate\x00bad",)),
        ("evidence_refs", ()),
        ("evidence_refs", []),
        ("evidence_refs", ("",)),
        ("evidence_refs", ("  ",)),
        ("evidence_refs", ("evidence\x00bad",)),
    ),
)
def test_relation_create_rejects_empty_or_nul_required_fields(
    field_name: str,
    invalid: object,
) -> None:
    fields = {
        "relation_type": RelationType.SUPERSEDES,
        "from_fact_id": "fact:v1:new",
        "to_fact_id": "fact:v1:old",
        "relation_reason": "synthetic reason",
        "candidate_ids": ("candidate",),
        "evidence_refs": ("evidence",),
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        RelationRecord.create(**fields)


def test_relation_create_rejects_self_edge() -> None:
    with pytest.raises(ReconciliationInputError, match="self-edge"):
        RelationRecord.create(
            RelationType.SUPERSEDES,
            "fact:v1:same",
            "fact:v1:same",
            "synthetic reason",
            ("candidate",),
            ("evidence",),
        )


class _StringSubclass(str):
    pass


class _TupleSubclass(tuple):
    pass


class _ListSubclass(list):
    pass


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("relation_type", "SUPERSEDES"),
        ("from_fact_id", _StringSubclass("fact:v1:new")),
        ("to_fact_id", _StringSubclass("fact:v1:old")),
        ("relation_reason", _StringSubclass("synthetic reason")),
        ("candidate_ids", "candidate"),
        ("candidate_ids", {"candidate"}),
        ("candidate_ids", _TupleSubclass(("candidate",))),
        ("candidate_ids", _ListSubclass(["candidate"])),
        ("candidate_ids", (_StringSubclass("candidate"),)),
        ("evidence_refs", "evidence"),
        ("evidence_refs", {"evidence"}),
        ("evidence_refs", _TupleSubclass(("evidence",))),
        ("evidence_refs", _ListSubclass(["evidence"])),
        ("evidence_refs", (_StringSubclass("evidence"),)),
    ),
)
def test_relation_create_requires_exact_types(
    field_name: str,
    invalid: object,
) -> None:
    fields = {
        "relation_type": RelationType.SUPERSEDES,
        "from_fact_id": "fact:v1:new",
        "to_fact_id": "fact:v1:old",
        "relation_reason": "synthetic reason",
        "candidate_ids": ("candidate",),
        "evidence_refs": ("evidence",),
    }
    fields[field_name] = invalid

    with pytest.raises(ReconciliationInputError, match=field_name):
        RelationRecord.create(**fields)


def test_make_relation_id_requires_exact_valid_identity_fields() -> None:
    with pytest.raises(ReconciliationInputError, match="relation_type"):
        make_relation_id("SUPERSEDES", "fact:v1:new", "fact:v1:old")
    with pytest.raises(ReconciliationInputError, match="from_fact_id"):
        make_relation_id(
            RelationType.SUPERSEDES,
            _StringSubclass("fact:v1:new"),
            "fact:v1:old",
        )
    with pytest.raises(ReconciliationInputError, match="to_fact_id"):
        make_relation_id(
            RelationType.SUPERSEDES,
            "fact:v1:new",
            "",
        )
    with pytest.raises(ReconciliationInputError, match="self-edge"):
        make_relation_id(
            RelationType.SUPERSEDES,
            "fact:v1:same",
            "fact:v1:same",
        )


def test_relation_direct_constructor_cannot_forge_identity() -> None:
    with pytest.raises(TypeError, match="created by create"):
        RelationRecord(
            relation_id="relation:v1:forged",
            relation_type=RelationType.SUPERSEDES,
            from_fact_id="fact:v1:new",
            to_fact_id="fact:v1:old",
            relation_reason="synthetic reason",
            candidate_ids=("candidate",),
            evidence_refs=("evidence",),
        )


def test_relation_internal_constructor_rejects_wrong_token_and_id_mismatch() -> None:
    fields = {
        "relation_id": "relation:v1:forged",
        "relation_type": RelationType.SUPERSEDES,
        "from_fact_id": "fact:v1:new",
        "to_fact_id": "fact:v1:old",
        "relation_reason": "synthetic reason",
        "candidate_ids": ("candidate",),
        "evidence_refs": ("evidence",),
    }

    with pytest.raises(TypeError, match="created by create"):
        RelationRecord._from_canonical(object(), **fields)
    with pytest.raises(ReconciliationInputError, match="relation_id"):
        RelationRecord._from_canonical(
            reconciliation_models._RELATION_RECORD_CONSTRUCTION_TOKEN,
            **fields,
        )


def test_relation_subclass_cannot_use_factory() -> None:
    class ForgedRelationRecord(RelationRecord):
        pass

    with pytest.raises(ReconciliationInputError, match="exact RelationRecord"):
        ForgedRelationRecord.create(
            RelationType.SUPERSEDES,
            "fact:v1:new",
            "fact:v1:old",
            "synthetic reason",
            ("candidate",),
            ("evidence",),
        )


def test_atomic_graph_derives_bidirectional_supersession_indexes() -> None:
    graph = materialize_relation_graph(
        provisional_facts("new", "old"),
        (supersedes_request("new", "old"),),
    )

    assert graph.fact("new").supersedes_fact_ids == ("old",)
    assert graph.fact("old").superseded_by_fact_ids == ("new",)
    assert graph.fact("new").relation_ids == (
        graph.relations[0].relation_id,
    )
    assert graph.fact("old").relation_ids == (
        graph.relations[0].relation_id,
    )


def test_conflict_edges_and_derived_indexes_are_symmetric() -> None:
    graph = materialize_relation_graph(
        provisional_facts("left", "right"),
        symmetric_conflict_requests("left", "right"),
    )

    assert tuple(
        (relation.from_fact_id, relation.to_fact_id)
        for relation in graph.relations
    ) == (("left", "right"), ("right", "left"))
    assert graph.fact("left").conflict_fact_ids == ("right",)
    assert graph.fact("right").conflict_fact_ids == ("left",)
    assert graph.fact("left").conflict_candidate_ids == (
        "candidate:right",
    )
    assert graph.fact("right").conflict_candidate_ids == (
        "candidate:left",
    )


def test_graph_merges_duplicate_requests_by_fixed_method_priority() -> None:
    requests = (
        supersedes_request(
            "new",
            "old",
            reason="lexically first but lower priority",
            candidate_ids=("candidate:new",),
            evidence_refs=("evidence:new",),
            method=ResolutionMethod.CURRENT_EVIDENCE_OVER_HISTORICAL,
        ),
        supersedes_request(
            "new",
            "old",
            reason="explicit winner",
            candidate_ids=("candidate:old",),
            evidence_refs=("evidence:old",),
            method=ResolutionMethod.EXPLICIT_SUPERSEDES,
        ),
    )

    graph = materialize_relation_graph(provisional_facts("old", "new"), requests)

    assert len(graph.relations) == 1
    assert graph.relations[0].relation_reason == "explicit winner"
    assert graph.relations[0].candidate_ids == (
        "candidate:new",
        "candidate:old",
    )
    assert graph.relations[0].evidence_refs == (
        "evidence:new",
        "evidence:old",
    )


def test_graph_orders_facts_relations_and_all_derived_indexes() -> None:
    facts = provisional_facts("z", "a", "m")
    requests = (
        supersedes_request("z", "a"),
        *symmetric_conflict_requests("m", "z"),
        supersedes_request("m", "a"),
    )

    graph = materialize_relation_graph(facts, requests)

    assert tuple(fact.fact_id for fact in graph.facts) == ("a", "m", "z")
    assert tuple(
        (
            relation.relation_type.value,
            relation.from_fact_id,
            relation.to_fact_id,
        )
        for relation in graph.relations
    ) == (
        ("CONFLICTS", "m", "z"),
        ("CONFLICTS", "z", "m"),
        ("SUPERSEDES", "m", "a"),
        ("SUPERSEDES", "z", "a"),
    )
    assert graph.fact("a").superseded_by_fact_ids == ("m", "z")
    assert graph.fact("m").relation_ids == tuple(sorted(graph.fact("m").relation_ids))


def test_graph_rejects_unknown_endpoint_without_publishing_any_graph() -> None:
    with pytest.raises(ReconciliationInputError, match="endpoint"):
        materialize_relation_graph(
            provisional_facts("known"),
            (supersedes_request("known", "missing"),),
        )


def test_graph_rejects_incomplete_conflict_pair() -> None:
    request = symmetric_conflict_requests("left", "right")[0]

    with pytest.raises(ReconciliationInputError, match="symmetric"):
        materialize_relation_graph(provisional_facts("left", "right"), (request,))


def test_graph_rejects_conflict_pair_with_asymmetric_payload() -> None:
    forward, _ = symmetric_conflict_requests("left", "right")
    reverse = RelationRequest.conflicts(
        "right",
        "left",
        "different reason",
        ("candidate:left", "candidate:right"),
        ("evidence:left", "evidence:right"),
    )

    with pytest.raises(ReconciliationInputError, match="same reason and provenance"):
        materialize_relation_graph(
            provisional_facts("left", "right"),
            (forward, reverse),
        )


def test_graph_rejects_noncanonical_exact_relation_request() -> None:
    forged = object.__new__(RelationRequest)
    for field_name, value in {
        "relation_type": "CONFLICTS",
        "from_fact_id": "left",
        "to_fact_id": "right",
        "reason": "unresolved competing current facts",
        "candidate_ids": ("candidate:left", "candidate:right"),
        "evidence_refs": ("evidence:left", "evidence:right"),
        "method": ResolutionMethod.UNRESOLVED_CONFLICT,
    }.items():
        object.__setattr__(forged, field_name, value)

    with pytest.raises(ReconciliationInputError, match="relation_type"):
        materialize_relation_graph(
            provisional_facts("left", "right"),
            (forged,),
        )


def test_graph_rejects_duplicate_fact_ids_and_preset_derived_indexes() -> None:
    fact = provisional_fact("same")

    with pytest.raises(ReconciliationInputError, match="unique fact_id"):
        materialize_relation_graph((fact, fact), ())

    forged = provisional_fact("forged")
    object.__setattr__(forged, "relation_ids", ("some-relation",))
    with pytest.raises(ReconciliationInputError, match="derived indexes must be empty"):
        materialize_relation_graph((forged,), ())


def test_graph_and_fact_snapshot_caller_lists_and_deep_values() -> None:
    nested = {"outer": ["initial"]}
    copied_fact = provisional_fact("fact", selected_value=nested)
    facts = [copied_fact]
    requests: list[RelationRequest] = []

    graph = materialize_relation_graph(facts, requests)
    facts.clear()
    requests.append(supersedes_request("other", "fact"))
    nested["outer"].append("mutated")

    assert graph.fact("fact").selected_value.value["outer"] == ("initial",)
    assert type(graph.facts) is tuple
    assert type(graph.relations) is tuple
    with pytest.raises(FrozenInstanceError):
        graph.facts = ()


def test_graph_converts_canonical_decisions_with_clock_provenance() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="current"))[0]

    graph = materialize_relation_graph((decision,), ())
    fact = graph.fact(decision.fact_id)

    assert fact.resolved_at.value == "2026-09-14T00:00:00+00:00"
    assert fact.resolved_at.source == "reconciliation:clock"
    assert fact.candidate_ids == decision.candidate_ids
    assert fact.activation_witness_candidate_ids == decision.activation_witness_candidate_ids


def test_reconciled_fact_direct_constructor_rejects_ghost_indexes() -> None:
    base = provisional_fact("fact")
    forged_fields = {
        field.name: getattr(base, field.name) for field in fields(ReconciledFact)
    }
    forged_fields.update(
        relation_ids=("relation:v1:ghost",),
        supersedes_fact_ids=("fact:v1:ghost",),
    )

    with pytest.raises((TypeError, ReconciliationInputError)):
        ReconciledFact(**forged_fields)


def test_reconciled_fact_dataclasses_replace_cannot_forge_indexes() -> None:
    base = provisional_fact("fact")

    with pytest.raises((TypeError, ReconciliationInputError)):
        replace(base, conflict_fact_ids=("fact:v1:ghost",))


def test_relation_graph_direct_constructor_cannot_supply_inconsistent_lookup() -> None:
    graph = materialize_relation_graph(provisional_facts("fact"), ())
    different_fact = materialize_relation_graph(
        provisional_facts("different"),
        (),
    ).fact("different")

    with pytest.raises((TypeError, ReconciliationInputError)):
        RelationGraph(
            facts=graph.facts,
            relations=graph.relations,
            _facts_by_id={"fact": different_fact},
        )


def _status_partitions(
    *facts: ReconciledFact,
) -> dict[ReconciliationStatus, tuple[ReconciledFact, ...]]:
    return {
        status: tuple(fact for fact in facts if fact.status is status)
        for status in ReconciliationStatus
    }


def _validate_result(
    graph: RelationGraph,
    *,
    source_decisions: tuple[FactDecision, ...],
    partitions: dict[
        ReconciliationStatus, tuple[ReconciledFact, ...]
    ] | None = None,
    unresolved: tuple[object, ...] = (),
    warnings: tuple[ReconciliationWarning, ...] = (),
    human_review_required: bool = False,
) -> None:
    validate_result_fields(
        facts=graph.facts,
        relations=graph.relations,
        partitions=(
            _status_partitions(*graph.facts)
            if partitions is None
            else partitions
        ),
        unresolved=unresolved,
        warnings=warnings,
        human_review_required=human_review_required,
        source_decisions=source_decisions,
    )


def test_valid_relation_graph_and_result_fields_pass_invariant_validation() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="current"))[0]
    graph = materialize_relation_graph((decision,), ())

    validate_relation_graph(graph)
    _validate_result(graph, source_decisions=(decision,))


def test_relation_endpoint_invariant_fails_for_missing_fact() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    dangling = RelationRecord.create(
        RelationType.SUPERSEDES,
        "new",
        "missing",
        "synthetic reason",
        ("candidate:new",),
        ("evidence:new",),
    )
    corrupt = _test_only_graph(graph.facts, (dangling,))

    with pytest.raises(ReconciliationInvariantError, match="endpoint exists"):
        validate_relation_graph(corrupt)


def test_relation_self_edge_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    self_edge = _test_only_copy(
        graph.relations[0],
        to_fact_id="new",
    )
    corrupt = _test_only_graph(graph.facts, (self_edge,))

    with pytest.raises(ReconciliationInvariantError, match="self-edge"):
        validate_relation_graph(corrupt)


def test_relation_identity_invariant_recomputes_id() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    forged = _test_only_copy(
        graph.relations[0],
        relation_id="relation:v1:forged",
    )
    corrupt = _test_only_graph(graph.facts, (forged,))

    with pytest.raises(ReconciliationInvariantError, match="canonical identity"):
        validate_relation_graph(corrupt)


def test_duplicate_relation_edge_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt = _test_only_graph(
        graph.facts,
        (graph.relations[0], graph.relations[0]),
    )

    with pytest.raises(ReconciliationInvariantError, match="duplicate.*edge"):
        validate_relation_graph(corrupt)


def test_missing_inverse_fact_index_fails_invariant_validation() -> None:
    graph = corrupt_graph_missing_superseded_by_index()

    with pytest.raises(
        ReconciliationInvariantError,
        match="superseded_by_fact_ids",
    ):
        validate_relation_graph(graph)


def test_ghost_supersession_index_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    source = graph.fact("new")
    corrupt_source = _test_only_copy(
        source,
        supersedes_fact_ids=("ghost", "old"),
    )
    corrupt = _test_only_graph(
        tuple(
            corrupt_source if fact.fact_id == "new" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="supersedes_fact_ids",
    ):
        validate_relation_graph(corrupt)


def test_relation_reverse_index_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    source = graph.fact("new")
    corrupt_source = _test_only_copy(source, relation_ids=())
    corrupt = _test_only_graph(
        tuple(
            corrupt_source if fact.fact_id == "new" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )

    with pytest.raises(ReconciliationInvariantError, match="relation_ids"):
        validate_relation_graph(corrupt)


def _valid_conflict_graph() -> RelationGraph:
    facts = tuple(
        _test_only_copy(
            provisional_fact(
                fact_id,
                ReconciliationStatus.CONFLICTED,
                requires_human_review=True,
            ),
            reason="unresolved competing current facts",
        )
        for fact_id in ("left", "right")
    )
    return materialize_relation_graph(
        facts,
        symmetric_conflict_requests("left", "right"),
    )


def test_conflict_symmetry_invariant_requires_inverse_edge() -> None:
    graph = _valid_conflict_graph()
    corrupt = _test_only_graph(graph.facts, (graph.relations[0],))

    with pytest.raises(ReconciliationInvariantError, match="symmetric inverse"):
        validate_relation_graph(corrupt)


def test_conflict_symmetry_invariant_requires_equal_payload() -> None:
    graph = _valid_conflict_graph()
    reverse = _test_only_copy(
        graph.relations[1],
        relation_reason="different reason",
    )
    corrupt = _test_only_graph(graph.facts, (graph.relations[0], reverse))

    with pytest.raises(
        ReconciliationInvariantError,
        match="same reason and provenance",
    ):
        validate_relation_graph(corrupt)


def test_conflict_fact_and_candidate_index_invariant_fails() -> None:
    graph = _valid_conflict_graph()
    left = graph.fact("left")
    corrupt_left = _test_only_copy(left, conflict_candidate_ids=())
    corrupt = _test_only_graph(
        tuple(
            corrupt_left if fact.fact_id == "left" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="conflict_candidate_ids",
    ):
        validate_relation_graph(corrupt)


def test_active_fact_incoming_supersedes_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_old = _test_only_copy(
        graph.fact("old"),
        status=ReconciliationStatus.ACTIVE,
        activation_witness_candidate_ids=("candidate:old",),
    )
    corrupt = _test_only_graph(
        tuple(
            corrupt_old if fact.fact_id == "old" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )

    with pytest.raises(ReconciliationInvariantError, match="ACTIVE.*incoming"):
        validate_relation_graph(corrupt)


def test_superseded_fact_incoming_supersedes_invariant_fails() -> None:
    graph = materialize_relation_graph(
        (provisional_fact("orphan", ReconciliationStatus.SUPERSEDED),),
        (),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="SUPERSEDED.*incoming",
    ):
        validate_relation_graph(graph)


def test_conflicted_fact_edge_and_review_invariants_fail_independently() -> None:
    orphan = materialize_relation_graph(
        (
            provisional_fact(
                "orphan",
                ReconciliationStatus.CONFLICTED,
                requires_human_review=True,
            ),
        ),
        (),
    )
    with pytest.raises(ReconciliationInvariantError, match="CONFLICTED.*edge"):
        validate_relation_graph(orphan)

    graph = _valid_conflict_graph()
    corrupt_left = _test_only_copy(
        graph.fact("left"),
        requires_human_review=False,
    )
    corrupt = _test_only_graph(
        tuple(
            corrupt_left if fact.fact_id == "left" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )
    with pytest.raises(ReconciliationInvariantError, match="require human review"):
        validate_relation_graph(corrupt)


def test_deprecated_replacement_invariant_fails() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_new = _test_only_copy(
        graph.fact("new"),
        status=ReconciliationStatus.DEPRECATED,
        activation_witness_candidate_ids=(),
    )
    corrupt = _test_only_graph(
        tuple(
            corrupt_new if fact.fact_id == "new" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="DEPRECATED.*replacement",
    ):
        validate_relation_graph(corrupt)


def test_unknown_active_value_invariant_fails() -> None:
    graph = materialize_relation_graph(
        (provisional_fact("active", ReconciliationStatus.ACTIVE),),
        (),
    )
    corrupt_fact = _test_only_copy(
        graph.fact("active"),
        selected_value=unknown("missing value"),
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(ReconciliationInvariantError, match="KNOWN selected_value"):
        validate_relation_graph(corrupt)


def test_fact_and_relation_provenance_invariants_fail_independently() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_fact = _test_only_copy(graph.fact("new"), evidence_refs=())
    corrupt = _test_only_graph(
        tuple(
            corrupt_fact if fact.fact_id == "new" else fact
            for fact in graph.facts
        ),
        graph.relations,
    )
    with pytest.raises(ReconciliationInvariantError, match="fact.*provenance"):
        validate_relation_graph(corrupt)

    corrupt_relation = _test_only_copy(graph.relations[0], candidate_ids=())
    corrupt = _test_only_graph(graph.facts, (corrupt_relation,))
    with pytest.raises(ReconciliationInvariantError, match="relation.*provenance"):
        validate_relation_graph(corrupt)


def test_status_partition_invariant_requires_exact_membership() -> None:
    decision = decisions_for(
        make_historical_candidate(candidate_id="pending")
    )[0]
    graph = materialize_relation_graph((decision,), ())
    partitions = _status_partitions(*graph.facts)
    partitions[ReconciliationStatus.PENDING] = ()

    with pytest.raises(
        ReconciliationInvariantError,
        match="exactly one status partition",
    ):
        _validate_result(
            graph,
            source_decisions=(decision,),
            partitions=partitions,
        )


def test_supersession_cycle_fails_invariant_validation() -> None:
    graph = graph_with_supersedes_edges(
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
    )

    with pytest.raises(ReconciliationInvariantError, match="acyclic"):
        validate_relation_graph(graph)


def test_active_witness_invariant_requires_canonical_candidate_membership() -> None:
    graph = materialize_relation_graph(
        (provisional_fact("active", ReconciliationStatus.ACTIVE),),
        (),
    )
    corrupt_fact = _test_only_copy(
        graph.fact("active"),
        activation_witness_candidate_ids=("candidate:ghost",),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="activation witness.*candidate_ids",
    ):
        validate_relation_graph(_test_only_graph((corrupt_fact,), ()))


def test_same_value_internal_supersession_invariant_requires_replaced_candidate(
) -> None:
    fact = provisional_fact(
        "reactivated",
        ReconciliationStatus.ACTIVE,
        candidate_ids=("candidate:new", "candidate:old"),
    )
    graph = materialize_relation_graph((fact,), ())
    corrupt = _test_only_copy(
        graph.fact("reactivated"),
        resolution_method=ResolutionMethod.SAME_VALUE_REACTIVATION,
        superseded_candidate_ids=(),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="internally replaced candidate",
    ):
        validate_relation_graph(_test_only_graph((corrupt,), ()))


def test_fact_candidate_membership_invariant_rejects_lost_original_candidate() -> None:
    fact = provisional_fact(
        "reactivated",
        ReconciliationStatus.ACTIVE,
        candidate_ids=("candidate:new", "candidate:old"),
        superseded_candidate_ids=("candidate:old",),
        activation_witness_candidate_ids=("candidate:new",),
    )
    graph = materialize_relation_graph((fact,), ())
    corrupt = _test_only_copy(
        graph.fact("reactivated"),
        candidate_ids=("candidate:new",),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="candidate_ids.*merged candidates",
    ):
        validate_relation_graph(_test_only_graph((corrupt,), ()))


def test_human_review_formula_invariant_does_not_follow_pending_status() -> None:
    decision = decisions_for(
        make_historical_candidate(candidate_id="pending")
    )[0]
    graph = materialize_relation_graph((decision,), ())

    _validate_result(
        graph,
        source_decisions=(decision,),
        human_review_required=False,
    )
    with pytest.raises(
        ReconciliationInvariantError,
        match="human_review_required.*formula",
    ):
        _validate_result(
            graph,
            source_decisions=(decision,),
            human_review_required=True,
        )


def test_human_review_formula_invariant_includes_warning_and_unresolved() -> None:
    warning_decision = decisions_for(
        make_historical_candidate(candidate_id="warning-pending")
    )[0]
    warning_graph = materialize_relation_graph((warning_decision,), ())
    warning = ReconciliationWarning(
        code=WarningCode.INVALID_TEMPORAL_ORDER,
        message="synthetic review warning",
        candidate_ids=("candidate:pending",),
        evidence_refs=("synthetic:pending",),
        requires_human_review=True,
    )
    with pytest.raises(
        ReconciliationInvariantError,
        match="human_review_required.*formula",
    ):
        _validate_result(
            warning_graph,
            source_decisions=(warning_decision,),
            warnings=(warning,),
        )

    groups = group_candidates(
        "synthetic-project",
        (
            make_candidate(
                candidate_id="pending",
                value=unknown(
                    "synthetic unresolved candidate",
                    source="synthetic:pending:value",
                ),
                source_type=SourceType.HISTORICAL_MEMORY,
                source_ref="synthetic:pending",
            ),
        ),
    )
    decisions, unresolved, _ = resolve_non_known_groups(
        groups,
        NOW,
        default_policy(),
    )
    decision = decisions[0]
    graph = materialize_relation_graph((decision,), ())
    with pytest.raises(
        ReconciliationInvariantError,
        match="human_review_required.*formula",
    ):
        _validate_result(
            graph,
            source_decisions=(decision,),
            unresolved=unresolved,
        )


def test_active_witness_source_invariant_rejects_historical_swap() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value="same"),
        make_historical_candidate(candidate_id="historical", value="same"),
    )[0]
    graph = materialize_relation_graph((decision,), ())
    corrupt_fact = _test_only_copy(
        graph.facts[0],
        activation_witness_candidate_ids=("historical",),
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match="activation_witness_candidate_ids.*canonical source decision",
    ):
        _validate_result(corrupt, source_decisions=(decision,))


def test_internal_supersession_source_invariant_rejects_partial_loss() -> None:
    oldest = make_current_candidate(candidate_id="a", value="same")
    middle = make_current_candidate(
        candidate_id="b",
        value="same",
        supersedes=("a",),
    )
    newest = make_current_candidate(
        candidate_id="c",
        value="same",
        supersedes=("b",),
    )
    decision = decisions_for(oldest, middle, newest)[0]
    assert decision.superseded_candidate_ids == ("a", "b")
    graph = materialize_relation_graph((decision,), ())
    corrupt_fact = _test_only_copy(
        graph.facts[0],
        superseded_candidate_ids=("b",),
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match="superseded_candidate_ids.*canonical source decision",
    ):
        _validate_result(corrupt, source_decisions=(decision,))


def test_merged_candidate_source_invariant_rejects_corroborator_loss() -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="current", value="same"),
        make_historical_candidate(candidate_id="historical", value="same"),
    )[0]
    graph = materialize_relation_graph((decision,), ())
    corrupt_fact = _test_only_copy(
        graph.facts[0],
        candidate_ids=("current",),
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match="candidate_ids.*canonical source decision",
    ):
        _validate_result(corrupt, source_decisions=(decision,))


@pytest.mark.parametrize("corrupt_relation", (object(), "not-a-relation"))
def test_relation_record_type_invariant_fails_before_sorting(
    corrupt_relation: object,
) -> None:
    graph = materialize_relation_graph((provisional_fact("fact"),), ())
    corrupt = _test_only_graph(graph.facts, (corrupt_relation,))

    with pytest.raises(
        ReconciliationInvariantError,
        match="exact RelationRecord",
    ):
        validate_relation_graph(corrupt)


def test_relation_type_invariant_fails_before_sorting() -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_relation = _test_only_copy(
        graph.relations[0],
        relation_type="SUPERSEDES",
    )
    corrupt = _test_only_graph(graph.facts, (corrupt_relation,))

    with pytest.raises(
        ReconciliationInvariantError,
        match="relation_type.*RelationType",
    ):
        validate_relation_graph(corrupt)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("from_fact_id", 1),
        ("from_fact_id", _StringSubclass("new")),
        ("from_fact_id", ""),
        ("from_fact_id", "  "),
        ("from_fact_id", "new\x00bad"),
        ("to_fact_id", 1),
        ("to_fact_id", _StringSubclass("old")),
        ("to_fact_id", ""),
        ("to_fact_id", "  "),
        ("to_fact_id", "old\x00bad"),
    ),
)
def test_relation_endpoint_field_invariant_fails_before_sorting(
    field_name: str,
    invalid: object,
) -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_relation = _test_only_copy(
        graph.relations[0],
        **{field_name: invalid},
    )
    corrupt = _test_only_graph(
        graph.facts,
        (graph.relations[0], corrupt_relation),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match=f"{field_name}.*non-empty NUL-free string",
    ):
        validate_relation_graph(corrupt)


def test_materialization_preserves_canonical_conflict_outcome() -> None:
    outcome = resolve_predicate(
        decisions_for(
            make_current_candidate(candidate_id="left", value="left"),
            make_current_candidate(candidate_id="right", value="right"),
        ),
        default_policy(),
    )

    graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )

    assert {fact.status for fact in graph.facts} == {
        ReconciliationStatus.CONFLICTED
    }
    assert all(fact.requires_human_review for fact in graph.facts)
    assert all(not fact.activation_witness_candidate_ids for fact in graph.facts)


def _explicit_supersedes_outcome():
    return resolve_predicate(
        decisions_for(
            make_current_candidate(candidate_id="old", value="old"),
            make_current_candidate(
                candidate_id="new",
                value="new",
                supersedes=("old",),
            ),
        ),
        default_policy(),
    )


def test_materialization_preserves_canonical_explicit_superseded_outcome() -> None:
    outcome = _explicit_supersedes_outcome()

    graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )

    assert tuple(fact.status for fact in graph.facts) == tuple(
        decision.status for decision in outcome.decisions
    )
    superseded = next(
        fact
        for fact in graph.facts
        if fact.status is ReconciliationStatus.SUPERSEDED
    )
    assert superseded.activation_witness_candidate_ids == ()


@pytest.mark.parametrize(
    ("field_name", "invalid", "message"),
    (
        ("subject", "forged-subject", "subject"),
        ("selected_value", known("forged-value"), "selected_value"),
        ("status", ReconciliationStatus.PENDING, "status"),
    ),
)
def test_source_replay_invariant_rejects_forged_final_fact_payload(
    field_name: str,
    invalid: object,
    message: str,
) -> None:
    decision = decisions_for(make_current_candidate(candidate_id="current"))[0]
    graph = materialize_relation_graph((decision,), ())
    changes = {field_name: invalid}
    if field_name == "status":
        changes["activation_witness_candidate_ids"] = ()
    corrupt_fact = _test_only_copy(graph.facts[0], **changes)
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match=f"{message}.*canonical predicate replay",
    ):
        _validate_result(corrupt, source_decisions=(decision,))


def test_source_replay_invariant_rejects_forged_relation_payload() -> None:
    outcome = _explicit_supersedes_outcome()
    graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )
    statuses = {decision.fact_id: decision.status for decision in outcome.decisions}
    reasons = {decision.fact_id: decision.reason for decision in outcome.decisions}
    methods = {
        decision.fact_id: decision.resolution_method
        for decision in outcome.decisions
    }
    facts = tuple(
        _test_only_copy(
            fact,
            status=statuses[fact.fact_id],
            reason=reasons[fact.fact_id],
            resolution_method=methods[fact.fact_id],
            activation_witness_candidate_ids=(
                fact.activation_witness_candidate_ids
                if statuses[fact.fact_id] is ReconciliationStatus.ACTIVE
                else ()
            ),
        )
        for fact in graph.facts
    )
    forged_relation = _test_only_copy(
        graph.relations[0],
        relation_reason="forged relation reason",
    )
    corrupt = _test_only_graph(facts, (forged_relation,))

    with pytest.raises(
        ReconciliationInvariantError,
        match="relations.*canonical predicate replay",
    ):
        _validate_result(corrupt, source_decisions=outcome.decisions)


class _ExplodingSequence(Sequence):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        raise RuntimeError("synthetic iteration failure")


@pytest.mark.parametrize("invalid_facts", (object(), _ExplodingSequence()))
def test_result_facts_container_invariant_is_total(
    invalid_facts: object,
) -> None:
    with pytest.raises(
        ReconciliationInvariantError,
        match="facts.*finite sequence",
    ):
        validate_result_fields(
            facts=invalid_facts,
            relations=(),
            partitions={},
            unresolved=(),
            warnings=(),
            human_review_required=False,
            source_decisions=(),
        )


def test_exact_reconciled_fact_invariant_precedes_attribute_access() -> None:
    graph = _test_only_graph((object(),), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match="exact ReconciledFact",
    ):
        validate_relation_graph(graph)


def test_incomplete_relation_record_invariant_precedes_attribute_access() -> None:
    incomplete = object.__new__(RelationRecord)
    object.__setattr__(incomplete, "relation_type", RelationType.SUPERSEDES)
    graph = materialize_relation_graph((provisional_fact("fact"),), ())
    corrupt = _test_only_graph(graph.facts, (incomplete,))

    with pytest.raises(
        ReconciliationInvariantError,
        match="complete canonical fields",
    ):
        validate_relation_graph(corrupt)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("relation_id", _StringSubclass("relation:v1:forged")),
        ("relation_reason", _StringSubclass("reason")),
        ("relation_reason", ""),
        ("candidate_ids", _TupleSubclass(("candidate",))),
        ("candidate_ids", ("",)),
        ("evidence_refs", _TupleSubclass(("evidence",))),
        ("evidence_refs", ("evidence\x00bad",)),
    ),
)
def test_relation_field_invariant_is_total_and_exact(
    field_name: str,
    invalid: object,
) -> None:
    graph = graph_with_supersedes_edges(("new", "old"))
    corrupt_relation = _test_only_copy(
        graph.relations[0],
        **{field_name: invalid},
    )
    corrupt = _test_only_graph(graph.facts, (corrupt_relation,))

    with pytest.raises(
        ReconciliationInvariantError,
        match=field_name,
    ):
        validate_relation_graph(corrupt)


def test_malformed_warning_and_partition_invariants_are_total() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="current"))[0]
    graph = materialize_relation_graph((decision,), ())

    with pytest.raises(
        ReconciliationInvariantError,
        match="warnings.*ReconciliationWarning",
    ):
        validate_result_fields(
            facts=graph.facts,
            relations=graph.relations,
            partitions=_status_partitions(*graph.facts),
            unresolved=(),
            warnings=(object(),),
            human_review_required=False,
            source_decisions=(decision,),
        )
    with pytest.raises(
        ReconciliationInvariantError,
        match="partitions.*mapping",
    ):
        validate_result_fields(
            facts=graph.facts,
            relations=graph.relations,
            partitions=object(),
            unresolved=(),
            warnings=(),
            human_review_required=False,
            source_decisions=(decision,),
        )


def test_iterative_supersedes_dag_invariant_handles_1100_nodes() -> None:
    edges = tuple(
        (f"node-{index:04d}", f"node-{index + 1:04d}")
        for index in range(1099)
    )
    graph = graph_with_supersedes_edges(*edges)

    _validate_supersedes_acyclic(
        tuple(fact.fact_id for fact in graph.facts),
        graph.relations,
    )

    cyclic = graph_with_supersedes_edges(
        *edges,
        ("node-1099", "node-0500"),
    )
    with pytest.raises(ReconciliationInvariantError, match="acyclic"):
        _validate_supersedes_acyclic(
            tuple(fact.fact_id for fact in cyclic.facts),
            cyclic.relations,
        )
