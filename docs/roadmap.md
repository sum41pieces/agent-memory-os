# Roadmap

## Phase 3 — Current State Model

Build an immutable, queryable snapshot over reconciled facts without adding
write authority or external-service dependencies.

## Phase 4 — Context Compiler

Compile bounded task context from current state, provenance, open conflicts,
and pending work.

## Phase 5 — Authority Gate

Evaluate proposed actions against explicit permissions. Context awareness must
remain separate from authorization.

## Phase 6 — Checkpoint Writer

Write controlled, reviewable checkpoints only after policy approval, with
provenance and rollback-friendly boundaries.

## Phase 7 — Eval Harness

Measure continuity, stale-state detection, conflict handling, authority
compliance, determinism, and privacy behavior using synthetic scenarios.

## Phase 8 — Dashboard

Visualize current facts, provenance, conflicts, pending work, and evaluation
results without becoming an alternate source of truth.
