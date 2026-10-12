# Final local validation — 2026-10-12

Windows / Python 3.12.14. Dependency closure is pinned in `requirements.lock`.
Source SHA-256: `2e3be6e57c4a8180026e7c1a179da26f073496fcbe9e3250e2a6c910805f9f6d`.
Sanitized run summaries and artifact hashes: `validation/acceptance.json`.

Executed successfully:

```
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python scripts/acceptance.py --prefix final_v1
agentreclab data freeze --dataset books --source datasets/fixtures/books100/dataset_manifest.json --limit 100 --out outputs/reports/final_v1_frozen.json
agentreclab validate --manifest outputs/reports/final_v1_frozen.json
agentreclab compare --runs outputs/runs/final_v1_100_iagent --runs outputs/runs/final_v1_100_i2agent --runs outputs/runs/final_v1_100_memrec --paired-only
agentreclab report --run outputs/runs/final_v1_100_memrec --out outputs/reports/final_v1_100_memrec.md
agentreclab evaluate --run outputs/runs/final_v1_100_memrec
agentreclab run --config configs/runs/books_100_fixture.yaml --model memrec --dry-run
git diff --check
```

49 tests passed in 19.86s. Lint and formatter passed. Acceptance reached `ALL_DONE`.

| Cases | Adapter | Evaluated | Failed | Mock dispatches | Cache replays | Resume new dispatches |
|---:|---|---:|---:|---:|---:|---:|
| 10 | iAgent | 10 | 0 | 20 | 0 | 0 |
| 10 | i²Agent | 10 | 0 | 50 | 0 | 0 |
| 10 | MemRec | 10 | 0 | 50 | 0 | 0 |
| 100 | iAgent | 100 | 0 | 75 | 125 | 0 |
| 100 | i²Agent | 100 | 0 | 255 | 245 | 0 |
| 100 | MemRec | 100 | 0 | 500 | 0 | 0 |

Paired comparisons covered all 10/100 frozen cases with identical labels, candidate
order and training snapshot. Actual paid API calls: zero; cost: zero CNY.
Fixtures intentionally repeat some synthetic content, producing legitimate response
cache hits; these are not 100 unique real user preference examples.

Expanded checks include four-domain synthetic original-format conversions, dataframe
index preservation, test-review exclusion, bijective namespaced/leading-zero IDs,
official iAgent/i²Agent independent valid-response differential checks, label-change
invariance, partial-case response replay, unknown-outcome redispatch blocking, malformed
JSON/duplicate/unknown/empty outputs, dated pricing/zero budget, user-level bootstrap,
and evidence-only conflict diagnostics. HTTP tests use MockTransport, not live APIs.

GitHub Actions configuration covers Python 3.11/3.12; remote CI success is not claimed
by these local results. Local verification has only exercised Python 3.12.14.
