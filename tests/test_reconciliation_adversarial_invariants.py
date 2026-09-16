"""Adversarial cohort, stage, and payload invariant tests."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, replace
from datetime import timedelta
import copy
import inspect
import pickle
from types import MappingProxyType

import pytest

from agent_memory_os.evidence.models import EvidenceStatus, EvidenceValue
import agent_memory_os.reconcile.models as reconciliation_models
import agent_memory_os.reconcile.reconciler as reconciliation_reconciler
import agent_memory_os.reconcile.rules as reconciliation_rules
from agent_memory_os.reconcile.models import (
    BaseFact,
    FactDecision,
    ReconciledFact,
    ReconciliationInputError,
    ReconciliationInvariantError,
    ReconciliationStatus,
    ReconciliationWarning,
    RelationGraph,
    RelationRecord,
    RelationType,
    WarningCode,
    _ReconciliationStage,
)
from agent_memory_os.reconcile.reconciler import (
    materialize_relation_graph,
    validate_relation_graph,
    validate_result_fields,
)
from agent_memory_os.reconcile.rules import (
    CandidateGroup,
    classify_group,
    group_candidates,
    resolve_predicate,
    resolve_cross_value_replacements,
)
from reconciliation_helpers import (
    NOW,
    default_policy,
    decisions_for,
    known,
    make_current_candidate,
    provisional_fact,
    _test_only_copy,
    _test_only_graph,
)


class _ExplodingSequence(Sequence[object]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        raise KeyError(f"unstable item {index}")


class _OneShotSequence(Sequence[object]):
    def __init__(self, values: tuple[object, ...]) -> None:
        self._values = values
        self.iterations = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> object:
        return self._values[index]

    def __iter__(self) -> Iterator[object]:
        self.iterations += 1
        if self.iterations > 1:
            raise RuntimeError("sequence iterated more than once")
        return iter(self._values)


def _two_value_groups(*, project_id: str = "synthetic-project"):
    return group_candidates(
        project_id,
        (
            make_current_candidate(candidate_id="left", value="left"),
            make_current_candidate(candidate_id="right", value="right"),
        ),
        snapshot_id="snapshot:v1:adversarial",
    )


def test_group_invocation_mints_shared_but_unreused_cohort() -> None:
    first = _two_value_groups()
    second = _two_value_groups()

    assert first[0]._cohort is first[1]._cohort
    assert first[0]._cohort.snapshot_id == second[0]._cohort.snapshot_id
    assert first[0]._cohort.invocation_token != second[0]._cohort.invocation_token


def test_predicate_resolution_rejects_different_invocation_tokens() -> None:
    first = _two_value_groups()
    second = _two_value_groups()
    policy = default_policy()
    decisions = (
        classify_group(first[0], NOW, policy),
        classify_group(second[1], NOW, policy),
    )

    with pytest.raises(ReconciliationInputError, match="cohort"):
        resolve_predicate(decisions, policy)


def test_predicate_resolution_rejects_empty_or_partial_membership() -> None:
    groups = _two_value_groups()
    policy = default_policy()
    decisions = tuple(classify_group(group, NOW, policy) for group in groups)

    with pytest.raises(ReconciliationInputError, match="non-empty|membership"):
        resolve_predicate((), policy)
    with pytest.raises(ReconciliationInputError, match="complete|membership"):
        resolve_predicate((decisions[0],), policy)

    outcome = resolve_predicate(decisions, policy)
    assert {decision.fact_id for decision in outcome.decisions} == {
        group.fact_id for group in groups
    }


def test_public_stage_sequences_are_guarded_and_snapshotted_once() -> None:
    policy = default_policy()
    groups = _two_value_groups()
    decisions = tuple(classify_group(group, NOW, policy) for group in groups)

    with pytest.raises(ReconciliationInputError, match="candidates.*sequence"):
        group_candidates("synthetic-project", _ExplodingSequence())
    with pytest.raises(ReconciliationInputError, match="decisions.*sequence"):
        resolve_predicate(_ExplodingSequence(), policy)
    with pytest.raises(ReconciliationInputError, match="facts.*sequence"):
        materialize_relation_graph(_ExplodingSequence(), ())
    with pytest.raises(ReconciliationInputError, match="requests.*sequence"):
        materialize_relation_graph(decisions, _ExplodingSequence())

    candidate_sequence = _OneShotSequence(
        (make_current_candidate(candidate_id="one-shot"),)
    )
    assert group_candidates("synthetic-project", candidate_sequence)
    assert candidate_sequence.iterations == 1

    decision_sequence = _OneShotSequence(decisions)
    outcome = resolve_predicate(decision_sequence, policy)
    assert outcome.decisions
    assert decision_sequence.iterations == 1

    fact_sequence = _OneShotSequence(outcome.decisions)
    request_sequence = _OneShotSequence(outcome.relation_requests)
    assert materialize_relation_graph(fact_sequence, request_sequence).facts
    assert fact_sequence.iterations == 1
    assert request_sequence.iterations == 1


def test_predicate_resolution_rejects_project_or_snapshot_cohort_mix() -> None:
    policy = default_policy()
    project_group = group_candidates(
        "other-project",
        (
            make_current_candidate(
                candidate_id="foreign-project",
                value="foreign-project",
            ),
        ),
        snapshot_id="snapshot:v1:adversarial",
    )[0]
    snapshot_group = group_candidates(
        "synthetic-project",
        (
            make_current_candidate(
                candidate_id="foreign-snapshot",
                value="foreign-snapshot",
            ),
        ),
        snapshot_id="snapshot:v1:other",
    )[0]
    baseline_group = group_candidates(
        "synthetic-project",
        (
            make_current_candidate(
                candidate_id="baseline",
                value="baseline",
            ),
        ),
        snapshot_id="snapshot:v1:adversarial",
    )[0]

    for foreign_group in (project_group, snapshot_group):
        decisions = (
            classify_group(baseline_group, NOW, policy),
            classify_group(foreign_group, NOW, policy),
        )
        with pytest.raises(ReconciliationInputError, match="cohort"):
            resolve_predicate(decisions, policy)


def test_predicate_resolution_rejects_policy_or_clock_cohort_mix() -> None:
    groups = _two_value_groups()
    policy = default_policy()
    changed_policy = replace(policy, active_confidence_threshold=0.75)

    with pytest.raises(ReconciliationInputError, match="cohort|policy"):
        resolve_predicate(
            (
                classify_group(groups[0], NOW, policy),
                classify_group(groups[1], NOW, changed_policy),
            ),
            policy,
        )
    with pytest.raises(ReconciliationInputError, match="cohort|clock"):
        resolve_predicate(
            (
                classify_group(groups[0], NOW, policy),
                classify_group(groups[1], NOW + timedelta(seconds=1), policy),
            ),
            policy,
        )


def test_materialize_rejects_cross_predicate_invocation_mix() -> None:
    setting = decisions_for(
        make_current_candidate(
            candidate_id="setting-candidate",
            predicate="setting",
        )
    )[0]
    owner = decisions_for(
        make_current_candidate(
            candidate_id="owner-candidate",
            predicate="owner",
        )
    )[0]

    with pytest.raises(ReconciliationInputError, match="cohort"):
        materialize_relation_graph((setting, owner), ())


def test_group_candidates_rejects_cross_group_duplicate_candidate_ids() -> None:
    duplicate_id_candidates = (
        make_current_candidate(
            candidate_id="duplicate",
            predicate="setting",
        ),
        make_current_candidate(
            candidate_id="duplicate",
            predicate="owner",
        ),
    )

    with pytest.raises(ReconciliationInputError, match="duplicate candidate_id"):
        group_candidates("synthetic-project", duplicate_id_candidates)


def test_group_candidates_default_snapshot_is_payload_derived() -> None:
    candidates = (
        make_current_candidate(candidate_id="left", value="left"),
        make_current_candidate(candidate_id="right", value="right"),
    )

    forward = group_candidates("synthetic-project", candidates)
    reverse = group_candidates("synthetic-project", tuple(reversed(candidates)))
    changed_payload = group_candidates(
        "synthetic-project",
        (
            make_current_candidate(candidate_id="left", value="changed"),
            make_current_candidate(candidate_id="right", value="right"),
        ),
    )

    assert forward[0]._cohort.snapshot_id == reverse[0]._cohort.snapshot_id
    assert forward[0]._cohort.candidate_id_namespace == (
        reverse[0]._cohort.candidate_id_namespace
    )
    assert forward[0]._cohort.invocation_token != reverse[0]._cohort.invocation_token
    assert forward[0]._cohort.snapshot_id != changed_payload[0]._cohort.snapshot_id
    assert forward[0]._cohort.candidate_id_namespace == (
        changed_payload[0]._cohort.candidate_id_namespace
    )


class _EvilEqual:
    def __eq__(self, other: object) -> bool:
        return True


class _IntSubclass(int):
    pass


class _EqualMapping(Mapping[str, object]):
    def __iter__(self) -> Iterator[str]:
        return iter(("key",))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        if key != "key":
            raise KeyError(key)
        return "value"

    def __eq__(self, other: object) -> bool:
        return True


class _EqualSequence(Sequence[object]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return "value"

    def __eq__(self, other: object) -> bool:
        return True


def _partitions(*facts: ReconciledFact):
    return {
        status: tuple(fact for fact in facts if fact.status is status)
        for status in ReconciliationStatus
    }


def _validate_with_source(graph, decisions, *, warnings=()) -> None:
    validate_result_fields(
        facts=graph.facts,
        relations=graph.relations,
        partitions=_partitions(*graph.facts),
        unresolved=(),
        warnings=warnings,
        human_review_required=any(
            warning.requires_human_review for warning in warnings
        ),
        source_decisions=decisions,
    )


@pytest.mark.parametrize(
    "forged_value",
    (
        _EvilEqual(),
        _IntSubclass(1),
        True,
        _EqualMapping(),
        _EqualSequence(),
    ),
)
def test_result_replay_rejects_noncanonical_known_value_types(
    forged_value: object,
) -> None:
    decision = decisions_for(
        make_current_candidate(candidate_id="candidate", value=1)
    )[0]
    graph = materialize_relation_graph((decision,), ())
    forged_evidence = EvidenceValue.known(
        forged_value,
        source=graph.facts[0].selected_value.source,
    )
    corrupt_fact = _test_only_copy(
        graph.facts[0],
        selected_value=forged_evidence,
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(ReconciliationInvariantError, match="selected_value"):
        _validate_with_source(corrupt, (decision,))


def test_result_replay_rejects_incomplete_evidence_shell() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    graph = materialize_relation_graph((decision,), ())
    shell = object.__new__(EvidenceValue)
    object.__setattr__(shell, "status", EvidenceStatus.KNOWN)
    corrupt_fact = _test_only_copy(graph.facts[0], confidence=shell)
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(ReconciliationInvariantError, match="confidence"):
        _validate_with_source(corrupt, (decision,))


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("status", _EvilEqual()),
        ("source", _IntSubclass(1)),
    ),
)
def test_result_replay_rejects_forged_evidence_status_or_source(
    field_name: str,
    forged_value: object,
) -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    graph = materialize_relation_graph((decision,), ())
    forged_evidence = object.__new__(EvidenceValue)
    for name in ("status", "value", "reason", "source"):
        object.__setattr__(
            forged_evidence,
            name,
            (
                forged_value
                if name == field_name
                else object.__getattribute__(graph.facts[0].confidence, name)
            ),
        )
    corrupt_fact = _test_only_copy(
        graph.facts[0],
        confidence=forged_evidence,
    )
    corrupt = _test_only_graph((corrupt_fact,), ())

    with pytest.raises(ReconciliationInvariantError, match="confidence"):
        _validate_with_source(corrupt, (decision,))


def test_decision_replay_rejects_corrupted_candidate_evidence_context() -> None:
    policy = default_policy()
    group = group_candidates(
        "synthetic-project",
        (make_current_candidate(candidate_id="candidate", value=1),),
        snapshot_id="snapshot:v1:explicit",
    )[0]
    decision = classify_group(group, NOW, policy)
    candidate = group.candidates[0]
    object.__setattr__(candidate, "value", known(True))

    with pytest.raises(ReconciliationInputError, match="context|cohort|evidence"):
        resolve_predicate((decision,), policy)


def test_result_warning_requires_nonempty_provenance() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    graph = materialize_relation_graph((decision,), ())
    warning = object.__new__(ReconciliationWarning)
    warning_values = {
        "code": WarningCode.INSUFFICIENT_CONFIDENCE,
        "message": "forged warning",
        "candidate_ids": (),
        "evidence_refs": (),
        "requires_human_review": False,
    }
    for field_name in fields(ReconciliationWarning):
        object.__setattr__(warning, field_name.name, warning_values[field_name.name])

    with pytest.raises(ReconciliationInvariantError, match="warning.*provenance"):
        _validate_with_source(graph, (decision,), warnings=(warning,))


def test_materialize_rejects_reconciled_fact_and_mixed_stages() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    fact = provisional_fact("provisional")
    graph = materialize_relation_graph((decision,), ())
    relation = RelationRecord.create(
        RelationType.SUPERSEDES,
        "fact-a",
        "fact-b",
        "explicit candidate supersedes relation",
        ("candidate-a", "candidate-b"),
        ("evidence-a", "evidence-b"),
    )

    with pytest.raises(ReconciliationInputError, match="FactDecision|stage"):
        materialize_relation_graph((fact,), ())
    with pytest.raises(ReconciliationInputError, match="FactDecision|stage"):
        materialize_relation_graph((decision, fact), ())
    with pytest.raises(ReconciliationInputError, match="FactDecision|stage"):
        materialize_relation_graph((graph,), ())
    with pytest.raises(ReconciliationInputError, match="RelationRequest"):
        materialize_relation_graph((decision,), (relation,))
    with pytest.raises(ReconciliationInputError, match="RelationRequest"):
        materialize_relation_graph((decision,), (graph,))
    assert not hasattr(reconciliation_reconciler, "_assemble_relation_graph")


def test_materialize_rejects_field_identical_unauthenticated_decision_copy() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    copied = _test_only_copy(decision)
    object.__setattr__(copied, "_candidate_context", decision._candidate_context)
    object.__setattr__(copied, "_cohort", decision._cohort)

    with pytest.raises(ReconciliationInputError, match="stage|authenticated"):
        materialize_relation_graph((copied,), ())
    with pytest.raises(
        ReconciliationInputError,
        match="canonical coordinator payload",
    ):
        reconciliation_rules._make_predicate_outcome((copied,), ())


def test_materialize_rejects_authenticated_decision_payload_forgery() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    object.__setattr__(decision, "reason", "forged payload")

    with pytest.raises(ReconciliationInputError, match="stage|context|payload"):
        materialize_relation_graph((decision,), ())


def test_stage_authority_exposes_no_legacy_mint_or_test_graph_backdoor() -> None:
    for name in (
        "_AUTHENTICATED_STAGE_MEMBERS",
        "_STAGE_AUTHORIZATION_CONSTRUCTION_TOKEN",
        "_mint_stage_authorization",
        "_claim_canonical_stage_builders",
    ):
        assert not hasattr(reconciliation_models, name)
    for name in (
        "_authenticate_relation_graph_for_tests",
        "_materialize_relation_graph_for_tests",
        "_make_canonical_relation_graph_builder",
    ):
        assert not hasattr(reconciliation_reconciler, name)
    assert not hasattr(
        reconciliation_rules,
        "_build_canonical_predicate_outcome",
    )
    for module in (
        reconciliation_models,
        reconciliation_rules,
        reconciliation_reconciler,
    ):
        assert not any(
            name.startswith("_authorize_") or "stage_builder" in name
            for name in vars(module)
        )
        for name, value in vars(module).items():
            if not callable(value):
                continue
            try:
                parameters = set(inspect.signature(value).parameters)
            except (TypeError, ValueError):
                continue
            assert not {
                "member",
                "payload",
                "cohort",
                "parent_fingerprint",
            }.issubset(parameters), name


def test_copied_replaced_or_pickled_stage_members_cannot_replay() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    for copier in (copy.copy, replace):
        try:
            copied = copier(decision)
        except TypeError:
            continue
        with pytest.raises(ReconciliationInputError, match="stage|authenticated"):
            materialize_relation_graph((copied,), ())

    try:
        copied = pickle.loads(pickle.dumps(decision))
    except (pickle.PickleError, TypeError):
        return
    with pytest.raises(ReconciliationInputError, match="stage|authenticated"):
        materialize_relation_graph((copied,), ())


def test_cross_value_replacement_requires_guarded_complete_membership() -> None:
    policy = default_policy()
    groups = _two_value_groups()
    decisions = tuple(classify_group(group, NOW, policy) for group in groups)

    with pytest.raises(ReconciliationInputError, match="non-empty|membership"):
        resolve_cross_value_replacements((), policy)
    with pytest.raises(ReconciliationInputError, match="complete|membership"):
        resolve_cross_value_replacements((decisions[0],), policy)
    with pytest.raises(ReconciliationInputError, match="decisions.*sequence"):
        resolve_cross_value_replacements(_ExplodingSequence(), policy)

    one_shot = _OneShotSequence(decisions)
    assert resolve_cross_value_replacements(one_shot, policy) == ()
    assert one_shot.iterations == 1


def test_materialized_model_constructors_remain_token_gated() -> None:
    for record_type in (BaseFact, ReconciledFact, RelationRecord, RelationGraph):
        with pytest.raises(TypeError):
            record_type()
    assert not hasattr(ReconciledFact, "_from_provisional")


def _single_fact_base_stage() -> tuple[RelationGraph, FactDecision, BaseFact]:
    decision = decisions_for(
        make_current_candidate(candidate_id="base-stage-candidate")
    )[0]
    graph = materialize_relation_graph((decision,), ())
    return graph, decision, graph.facts[0]._base_fact


def test_base_fact_pipeline_rejects_wrong_stage_type() -> None:
    graph, decision, base_fact = _single_fact_base_stage()

    with pytest.raises(ReconciliationInputError, match="FactDecision|stage"):
        materialize_relation_graph((base_fact,), ())
    with pytest.raises(ReconciliationInputError, match="BaseFact"):
        ReconciledFact._from_materialized(
            reconciliation_models._RECONCILED_FACT_CONSTRUCTION_TOKEN,
            decision,
            conflict_candidate_ids=(),
            relation_ids=(),
            supersedes_fact_ids=(),
            superseded_by_fact_ids=(),
            conflict_fact_ids=(),
        )
    with pytest.raises(ReconciliationInvariantError, match="ReconciledFact"):
        validate_relation_graph(_test_only_graph((base_fact,), ()))
    assert type(base_fact) is BaseFact
    assert type(decision) is FactDecision
    assert type(graph.facts[0]) is ReconciledFact


def test_base_fact_pipeline_rejects_mismatched_context() -> None:
    graph, outcome = _explicit_replacement_graph()
    fact = graph.facts[0]
    wrong_decision = next(
        decision
        for decision in outcome.decisions
        if decision.fact_id != fact.fact_id
    )
    object.__setattr__(
        fact._base_fact,
        "_source_decision",
        wrong_decision,
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="BaseFact|parent|stage",
    ):
        validate_relation_graph(graph)


def test_base_fact_pipeline_rejects_stale_stage_object() -> None:
    graph, decision, _ = _single_fact_base_stage()
    fresh_graph = materialize_relation_graph((decision,), ())
    object.__setattr__(
        fresh_graph.facts[0],
        "_base_fact",
        graph.facts[0]._base_fact,
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="BaseFact|parent|stage",
    ):
        validate_relation_graph(fresh_graph)


def test_base_fact_pipeline_rejects_mixed_stage_collection() -> None:
    _, decision, base_fact = _single_fact_base_stage()

    with pytest.raises(ReconciliationInputError, match="FactDecision|stage"):
        materialize_relation_graph((decision, base_fact), ())


def test_base_fact_pipeline_rejects_unregistered_copy() -> None:
    graph, _, base_fact = _single_fact_base_stage()
    with pytest.raises(TypeError, match="BaseFact|reconciliation"):
        copy.copy(base_fact)
    with pytest.raises(TypeError, match="BaseFact|init|argument"):
        replace(base_fact, reason="forged replacement")
    copied = _test_only_copy(base_fact)
    object.__setattr__(graph.facts[0], "_base_fact", copied)

    with pytest.raises(
        ReconciliationInvariantError,
        match="BaseFact|parent|stage",
    ):
        validate_relation_graph(graph)


def test_base_fact_pipeline_materializes_exact_parent_stage() -> None:
    graph, _, base_fact = _single_fact_base_stage()
    fact = graph.facts[0]

    assert base_fact._stage_auth.stage is _ReconciliationStage.BASE_FACT
    assert fact._stage_auth.previous_stage is _ReconciliationStage.BASE_FACT
    assert fact._stage_auth.parent_fingerprint == base_fact._stage_auth.fingerprint
    assert not hasattr(base_fact, "relation_ids")
    with pytest.raises(FrozenInstanceError):
        setattr(base_fact, "reason", "mutated")
    validate_relation_graph(graph)


def test_base_fact_pipeline_binds_relation_and_result_parent_chain() -> None:
    graph, outcome = _explicit_replacement_graph()
    base_auth_by_fact_id = {
        fact.fact_id: fact._base_fact._stage_auth for fact in graph.facts
    }

    for relation in graph.relations:
        assert (
            relation._stage_auth.previous_stage
            is _ReconciliationStage.BASE_FACT
        )
        assert relation._stage_auth.parent_fingerprint == (
            reconciliation_reconciler._relation_parent_fingerprint(
                base_auth_by_fact_id,
                relation.from_fact_id,
                relation.to_fact_id,
            )
        )
    _validate_with_source(graph, outcome.decisions)


def test_candidate_group_public_constructor_cannot_mint_group_stage() -> None:
    candidate = make_current_candidate(candidate_id="candidate")

    with pytest.raises((TypeError, ReconciliationInputError), match="internal|group"):
        CandidateGroup("synthetic-project", (candidate,))


def test_classify_group_rejects_field_identical_group_clone() -> None:
    group = group_candidates(
        "synthetic-project",
        (make_current_candidate(candidate_id="candidate"),),
    )[0]
    clone = _test_only_copy(group)

    with pytest.raises(ReconciliationInputError, match="group.*stage|authenticated"):
        classify_group(clone, NOW, default_policy())


@pytest.mark.parametrize(
    "corruption",
    ("empty", "wrong-key", "clone", "foreign-build"),
)
def test_graph_lookup_must_bind_canonical_registered_fact_members(
    corruption: str,
) -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    graph = materialize_relation_graph((decision,), ())
    fact = graph.facts[0]
    if corruption == "empty":
        lookup = {}
    elif corruption == "wrong-key":
        lookup = {"fact:v1:wrong": fact}
    elif corruption == "clone":
        lookup = {fact.fact_id: _test_only_copy(fact)}
    else:
        foreign = materialize_relation_graph((decision,), ()).facts[0]
        assert foreign._cohort.fingerprint == fact._cohort.fingerprint
        assert (
            foreign._stage_auth.graph_build_token
            != fact._stage_auth.graph_build_token
        )
        lookup = {fact.fact_id: foreign}
    object.__setattr__(graph, "_facts_by_id", MappingProxyType(lookup))

    with pytest.raises(ReconciliationInvariantError, match="lookup|stage|graph build"):
        validate_relation_graph(graph)


def test_status_partition_rejects_field_identical_unregistered_fact_clone() -> None:
    decision = decisions_for(make_current_candidate(candidate_id="candidate"))[0]
    graph = materialize_relation_graph((decision,), ())
    fact = graph.facts[0]
    clone = object.__new__(ReconciledFact)
    for field in fields(ReconciledFact):
        object.__setattr__(clone, field.name, object.__getattribute__(fact, field.name))
    partitions = _partitions(*graph.facts)
    partitions[fact.status] = (clone,)

    with pytest.raises(
        ReconciliationInvariantError,
        match="partition.*authenticated|canonical member|stage",
    ):
        validate_result_fields(
            facts=graph.facts,
            relations=graph.relations,
            partitions=partitions,
            unresolved=(),
            warnings=(),
            human_review_required=False,
            source_decisions=(decision,),
        )


def _explicit_replacement_graph():
    policy = default_policy()
    groups = group_candidates(
        "synthetic-project",
        (
            make_current_candidate(candidate_id="old", value="old"),
            make_current_candidate(
                candidate_id="new",
                value="new",
                supersedes=("old",),
            ),
        ),
        snapshot_id="snapshot:v1:replacement",
    )
    outcome = resolve_predicate(
        tuple(classify_group(group, NOW, policy) for group in groups),
        policy,
    )
    return materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    ), outcome


def test_validator_rejects_stale_source_replay_from_new_invocation() -> None:
    first = decisions_for(make_current_candidate(candidate_id="candidate"))
    second = decisions_for(make_current_candidate(candidate_id="candidate"))
    graph = materialize_relation_graph(first, ())

    with pytest.raises(ReconciliationInvariantError, match="cohort|stage"):
        _validate_with_source(graph, second)


def test_validator_rejects_fact_or_relation_from_another_graph_build() -> None:
    first_graph, outcome = _explicit_replacement_graph()
    second_graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )
    assert first_graph._cohort.fingerprint == second_graph._cohort.fingerprint
    assert (
        first_graph._stage_auth.graph_build_token
        != second_graph._stage_auth.graph_build_token
    )
    foreign_fact_graph = object.__new__(type(first_graph))
    foreign_facts = (second_graph.facts[0], *first_graph.facts[1:])
    object.__setattr__(foreign_fact_graph, "facts", foreign_facts)
    object.__setattr__(foreign_fact_graph, "relations", first_graph.relations)
    object.__setattr__(
        foreign_fact_graph,
        "_facts_by_id",
        MappingProxyType({fact.fact_id: fact for fact in foreign_facts}),
    )
    object.__setattr__(foreign_fact_graph, "_cohort", first_graph._cohort)
    object.__setattr__(foreign_fact_graph, "_stage_auth", first_graph._stage_auth)

    foreign_relation_graph = object.__new__(type(first_graph))
    object.__setattr__(foreign_relation_graph, "facts", first_graph.facts)
    object.__setattr__(
        foreign_relation_graph,
        "relations",
        (second_graph.relations[0],),
    )
    object.__setattr__(
        foreign_relation_graph,
        "_facts_by_id",
        first_graph._facts_by_id,
    )
    object.__setattr__(foreign_relation_graph, "_cohort", first_graph._cohort)
    object.__setattr__(
        foreign_relation_graph,
        "_stage_auth",
        first_graph._stage_auth,
    )

    for corrupt in (foreign_fact_graph, foreign_relation_graph):
        with pytest.raises(ReconciliationInvariantError, match="graph build|stage"):
            validate_relation_graph(corrupt)


@pytest.mark.parametrize("member_kind", ("fact", "relation"))
def test_result_replay_binds_member_parent_to_actual_source_decision(
    member_kind: str,
) -> None:
    graph, outcome = _explicit_replacement_graph()
    member = graph.facts[0] if member_kind == "fact" else graph.relations[0]
    old_auth = member._stage_auth
    copied_parent_fingerprint = next(
        decision._stage_auth.fingerprint
        for decision in outcome.decisions
        if member_kind == "relation"
        or decision.fact_id != member.fact_id
    )
    forged_auth = object.__new__(type(old_auth))
    for field in fields(old_auth):
        object.__setattr__(
            forged_auth,
            field.name,
            (
                copied_parent_fingerprint
                if field.name == "parent_fingerprint"
                else object.__getattribute__(old_auth, field.name)
            ),
        )
    object.__setattr__(member, "_stage_auth", forged_auth)

    with pytest.raises(ReconciliationInvariantError, match="parent|source decision"):
        _validate_with_source(graph, outcome.decisions)


@pytest.mark.parametrize(
    "forged_status",
    (ReconciliationStatus.ACTIVE, ReconciliationStatus.PENDING),
)
def test_conflict_edge_requires_exact_conflicted_review_endpoints(
    forged_status: ReconciliationStatus,
) -> None:
    policy = default_policy()
    groups = _two_value_groups()
    outcome = resolve_predicate(
        tuple(classify_group(group, NOW, policy) for group in groups),
        policy,
    )
    graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )
    first = graph.facts[0]
    forged = _test_only_copy(
        first,
        status=forged_status,
        requires_human_review=False,
        activation_witness_candidate_ids=(
            (first.candidate_ids[0],)
            if forged_status is ReconciliationStatus.ACTIVE
            else ()
        ),
    )
    corrupt = _test_only_graph((forged, *graph.facts[1:]), graph.relations)

    with pytest.raises(
        ReconciliationInvariantError,
        match="CONFLICTS.*CONFLICTED.*human review",
    ):
        validate_relation_graph(corrupt)


def test_conflict_edge_requires_exact_symmetric_payload() -> None:
    policy = default_policy()
    groups = _two_value_groups()
    outcome = resolve_predicate(
        tuple(classify_group(group, NOW, policy) for group in groups),
        policy,
    )
    graph = materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )
    forged_reverse = _test_only_copy(
        graph.relations[1],
        relation_reason="forged reverse reason",
    )
    corrupt = _test_only_graph(
        graph.facts,
        (graph.relations[0], forged_reverse),
    )

    with pytest.raises(
        ReconciliationInvariantError,
        match="symmetric CONFLICTS.*same reason and provenance",
    ):
        validate_relation_graph(corrupt)


def _three_value_conflict_graph():
    policy = default_policy()
    groups = group_candidates(
        "synthetic-project",
        tuple(
            make_current_candidate(candidate_id=value, value=value)
            for value in ("left", "middle", "right")
        ),
    )
    outcome = resolve_predicate(
        tuple(classify_group(group, NOW, policy) for group in groups),
        policy,
    )
    return materialize_relation_graph(
        outcome.decisions,
        outcome.relation_requests,
    )


def _graph_with_rebuilt_indexes(graph, relations, *, reason=None):
    rebuilt_facts = []
    for fact in graph.facts:
        incident = tuple(
            relation
            for relation in relations
            if fact.fact_id in (relation.from_fact_id, relation.to_fact_id)
        )
        conflicts = tuple(
            sorted(
                relation.to_fact_id
                for relation in relations
                if relation.relation_type is RelationType.CONFLICTS
                and relation.from_fact_id == fact.fact_id
            )
        )
        conflict_candidates = tuple(
            sorted(
                {
                    candidate_id
                    for relation in incident
                    if relation.relation_type is RelationType.CONFLICTS
                    for candidate_id in relation.candidate_ids
                    if candidate_id not in fact.candidate_ids
                }
            )
        )
        rebuilt_facts.append(
            _test_only_copy(
                fact,
                reason=fact.reason if reason is None else reason,
                relation_ids=tuple(sorted(relation.relation_id for relation in incident)),
                supersedes_fact_ids=tuple(
                    sorted(
                        relation.to_fact_id
                        for relation in relations
                        if relation.relation_type is RelationType.SUPERSEDES
                        and relation.from_fact_id == fact.fact_id
                    )
                ),
                superseded_by_fact_ids=tuple(
                    sorted(
                        relation.from_fact_id
                        for relation in relations
                        if relation.relation_type is RelationType.SUPERSEDES
                        and relation.to_fact_id == fact.fact_id
                    )
                ),
                conflict_fact_ids=conflicts,
                conflict_candidate_ids=conflict_candidates,
            )
        )
    return _test_only_graph(tuple(rebuilt_facts), tuple(relations))


def test_conflict_edges_reject_ghost_provenance_and_arbitrary_reason() -> None:
    graph = _three_value_conflict_graph()
    for corruption in ("ghost-provenance", "arbitrary-reason"):
        relations = []
        reason = None
        for relation in graph.relations:
            if corruption == "ghost-provenance":
                candidate_ids = tuple(sorted((*relation.candidate_ids, "ghost")))
                evidence_refs = tuple(sorted((*relation.evidence_refs, "ghost:ref")))
                relation_reason = relation.relation_reason
            else:
                candidate_ids = relation.candidate_ids
                evidence_refs = relation.evidence_refs
                relation_reason = "arbitrary conflict reason"
                reason = relation_reason
            relations.append(
                RelationRecord.create(
                    RelationType.CONFLICTS,
                    relation.from_fact_id,
                    relation.to_fact_id,
                    relation_reason,
                    candidate_ids,
                    evidence_refs,
                )
            )
        corrupt = _graph_with_rebuilt_indexes(
            graph,
            tuple(relations),
            reason=reason,
        )

        with pytest.raises(
            ReconciliationInvariantError,
            match="CONFLICTS.*provenance|canonical conflict reason",
        ):
            validate_relation_graph(corrupt)


def test_conflicted_facts_reject_supersedes_and_require_complete_clique() -> None:
    graph = _three_value_conflict_graph()
    left, middle, right = graph.facts
    supersedes = RelationRecord.create(
        RelationType.SUPERSEDES,
        left.fact_id,
        middle.fact_id,
        "explicit candidate supersedes relation",
        tuple(sorted((*left.candidate_ids, *middle.candidate_ids))),
        tuple(sorted((*left.evidence_refs, *middle.evidence_refs))),
    )
    conflict_and_supersedes = _graph_with_rebuilt_indexes(
        graph,
        tuple((*graph.relations, supersedes)),
    )
    with pytest.raises(ReconciliationInvariantError, match="CONFLICTED.*SUPERSEDES"):
        validate_relation_graph(conflict_and_supersedes)

    omitted_pair = {left.fact_id, right.fact_id}
    incomplete_relations = tuple(
        relation
        for relation in graph.relations
        if {relation.from_fact_id, relation.to_fact_id} != omitted_pair
    )
    incomplete_clique = _graph_with_rebuilt_indexes(
        graph,
        incomplete_relations,
    )

    with pytest.raises(ReconciliationInvariantError, match="complete.*clique"):
        validate_relation_graph(incomplete_clique)
