# Reproduction evidence

See model_mapping for exact commits, dependency versions, data fields and original
commands. P0 audit and P1 mock infrastructure have been implemented. P2 converter and
P3/P4/P5 official-code wrappers are available and checked on synthetic fixtures.
Raw InstructRec conversion parity, original user ID equivalence, and standalone
numeric differential runs on real data have not been demonstrated.

The iAgent client is injected; its provider parameters are resolved explicitly.
Runtime metric hook returns neutral values to suppress label-dependent self-reflection;
framework validation applies afterward. Retry sleeps are delegated to the provider;
the bridge latches exhausted errors so upstream catch loops cannot dispatch again.
This is a controlled-protocol change, not strict original repair behavior.

i²Agent follows the published source's per-case simulated recommendation/profile,
including `[-2]` indexing after the test is removed. It does not invent a persistent
history-learning procedure absent in that source.

MemRec delegates graph, rules, packing, retrieval, reranking and training Stage-W to
official components. Unused MLP imports are omitted. Candidate sampling/order and
training-only visibility differ from the original trainer and are manifested.
Bridge exceptions are propagated despite upstream fallback catches. No failed paid
call is silently converted into a successful default ranking.

MARS is intentionally gated. P6 real results and P7 lifecycle implementation remain
separate milestones requiring data/authorization and verified reference behavior.
