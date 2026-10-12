# Protocols

v0.1 executes `offline_static`: prepare training memory → freeze state → predict
each case → validate whole candidate permutation → evaluate. No test labels are
revealed to memory updates. Future reveal/observe/commit guard is a contract only,
not an implemented online environment.

Controlled comparison freezes original candidates in original order, cases, label
IDs and train snapshot. It preserves algorithm-specific prompts and warmup, but
discloses changes from the official trainers. Missing/duplicate/unknown predictions
fail by default. `append_missing` repairs omissions only; duplicate/unknown IDs fail.
`one_retry` is an explicit bounded whole-rank repair, with serialized injection.

Paper reproduction remains not_ready until original dataset identity, candidate/split
protocol, model deployment and standalone differential results are verified. The
independent `paper_original` commands are in model_mapping; results must never be
mixed with controlled comparison metrics.

Original iAgent candidates are precomputed. MemRec trainer resamples them and can
consume test feedback; this framework deliberately freezes candidates and disables
test writes. Its graph uses training history excluding each user's penultimate
interaction, as audited leave-two-out. One-round warmup uses the final historical
interaction and a frozen seeded training candidate list. At each training feedback
timestamp, the graph excludes events at or after that boundary. Other-user memory
updates execute in stable chronological order; unknown calendar timestamps imply
only sequence-level fidelity and are disclosed.
