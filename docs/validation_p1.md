# P1 validation — 2026-10-12

Environment: Windows, Python 3.12.14, source checkout; upstream commits in lock.json.
Commands executed:

```
python -m ruff check src tests scripts
python -m ruff format src tests scripts
python -m pytest -q
python -m agentreclab.cli doctor
python -m agentreclab.cli run --config configs/runs/smoke_books.yaml --model memrec --run-id p1_smoke_memrec
```

Result: lint passed; 31 tests passed in 12.45s. Three official-code adapters passed
10-case mock integration + exact resume. MemRec CLI: completed, 10 evaluated,
0 failed, 50 mock calls, 0 CNY. Golden metrics, malformed data, global/local future
leakage, permutation errors, checksum tampering, 429/5xx/timeout/400, Azure request,
missing usage, budget, seed mismatch, paired candidate mismatch and reveal guard tested.

P1 evidence does not establish real-data or paper-result parity. Later test additions
and the final 100-case validation are recorded separately.
