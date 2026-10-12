# Reproduction risks and milestone gates

P0 complete: pinned upstream checkouts and source audit in `model_mapping/`.
Original data are absent; user ID alignment, candidate checksums, download license,
and standalone numeric parity remain unknown. No strict paper reproduction claim.

P1 onward must use synthetic mock fixtures before any paid experiment. Real-provider
access is implemented but paid runs require an explicit command-line opt-in; no paid
API use is authorized by a generic implementation request.

Controlled comparison preserves test cases/candidate order and model-specific
upstream algorithms. It changes sample limits, repair criteria, graph visibility and
test-feedback behavior; all changes belong in run manifests. Synthetic metrics are
infrastructure validation, never evidence of recommendation quality.

P2 raw-data parity and P6 100→1000→full real experiments await legitimately obtained
InstructRec files and explicit budget authorization. P7 MARS remains gated.
