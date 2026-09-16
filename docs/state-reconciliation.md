# State reconciliation

The State Reconciler turns competing, time-sensitive candidates into explicit
fact states. Fixed inputs and a fixed clock produce the same result.

| Status | Meaning | Synthetic example |
| --- | --- | --- |
| `ACTIVE` | Best supported current fact | The finals project path is explicitly designated as the source of truth. |
| `SUPERSEDED` | Replaced by a newer or explicitly authoritative fact | The version-one project path is replaced by the finals path. |
| `CONFLICTED` | Competing evidence cannot be resolved safely | Two project documents claim different runtime ports. |
| `PENDING` | A valid future plan is not yet current | A dashboard task has a future activation time. |
| `DEPRECATED` | Retired without becoming current | A legacy architecture mode is explicitly deprecated. |

## Synthetic path replacement

The example uses only fictional paths:

```text
old: C:\Users\demo\projects\interview-agent-v1      -> SUPERSEDED
new: C:\Users\demo\projects\interview-agent-finals -> ACTIVE
reason: explicit current source-of-truth designation
```

The relation is directed from the active fact to the superseded fact and keeps
the candidate IDs and evidence references that justify the decision.

## Conflict handling

A conflict is retained instead of guessed away when candidates have comparable
authority and no rule can select one safely. Conflict relations are symmetric,
and the result records that human review is required.

## Provenance and invariants

Facts retain their original candidates, activation witnesses, and relation
indexes. Result validation checks status partitions, relation endpoints,
conflict symmetry, supersession direction, identity hashes, and provenance
consistency before a result can be serialized.
