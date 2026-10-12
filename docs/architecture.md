# Architecture

CLI → resolved config → checksum-verified dataset → protocol readiness gate → runner.
Runner provides catalog, training event index, immutable case objects and traced provider.
Model-specific official code is imported from pinned clean local checkouts. Models do
not receive the test answer in upstream task objects or LLM prompts. Evaluation alone
uses positive labels after complete ranking validation.

Parquet separates interactions/catalog/cases. SQLite commits one final result per case,
response cache and conservative call reservations. Raw responses are content-addressed;
atomic fsync/replace snapshots precede prepared-state manifest commits. The offline
protocol does not mutate model state during evaluation. MemRec warmup is completed and
snapshotted before evaluation; interrupted preparation rebuilds from cached responses.

iAgent/i²Agent sync calls are bridged from worker threads into async provider calls.
MemRec uses official rules+LLM path without importing unused torch MLP/encoders; absolute
imports are relocated under a private namespace, algorithm source is unchanged.
MemRec evaluation is serialized because its clients are mutable; requested/effective
concurrency are both recorded. Provider calls are serialized in v0.1 for a safe shared
cache/budget bound, while model task scheduling can be concurrent.

Each run_id has an OS-released SQLite exclusive lock. The same run can be resumed only
under the same config/data/source fingerprint. Cache bytes are preserved before parsing.
An unresolved request from a hard crash must not silently be sent again; see limitations.
