# Delivered scope and remaining evidence

| Stage | Available and verified locally | Remaining evidence |
|---|---|---|
| P0 | Pinned upstream audit, commands, mappings, source/license risks | Exact original data license and MARS official code unknown |
| P1 | Schemas, validators, providers, traces, cache, SQLite resume, CLI, fixtures, metrics | Live provider connectivity only after opt-in |
| P2 | Four-domain original-schema converter and ordered candidate freezing tested on synthetic pickle | Actual InstructRec ID/candidate/split parity report |
| P3 | Official static class wrapper; valid-response differential mock check | Real data and independent paid reference run |
| P4 | Official per-case dynamic/profile warmup; valid-response differential mock check | Original data/LLM parity; official source does not persist cross-case profiles |
| P5 | Official rules+graph+Stage-R/ReRank/Stage-W, one-round training warmup, snapshot replay | Numeric parity with full original trainer and original asynchronous schedule |
| P6 | 10/100-case frozen mock paired report, token/call/cost/failure summaries | Real 5→20→100→1000→full runs after data and spending authorization |
| P7 | Explicit MARS not_ready gate, paper/source mapping | Verified lifecycle implementation and reference outputs |
| P8 | Custom retrieval injection and opt-in lexical conflict analyzer | Scientific validation of diagnostic labels |

No real raw data, API secrets, or copied upstream source are committed. Current delivery
is a runnable, tested offline framework with official-code adapters; it does not meet
the separate scientific milestone of 100 real InstructRec cases with paid-model parity.
