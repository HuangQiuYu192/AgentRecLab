> Historical v0.1 source audit. Current runtime uses native models; wrapper plans below are superseded by docs/reproduction.md.

# MemRec audit — 2026-10-12

Source: https://github.com/rutgerswiselab/memrec/tree/58d9031ed91c623b8034d2bf04f39aa937424c33
Python 3.10 per environment.yml. requirements: torch 2.9.0, numpy 1.26.4,
pandas 2.2.2, openai 2.8.1, transformers 4.57.1, scipy/scikit-learn plus others.
No LICENSE found; redistribution permission unknown. Keep pinned local source only.

Original entry: `python scripts/run_train.py --model memrec_agent --dataset instructrec-books --config configs/memrec_agent_instructrec-books.yaml`.
Conversion: `scripts/convert_iagent_to_memrec.py --pkl <file> --mapping <csv> --output <dir>`.
Converter assigns user IDs by row counter (not original dataframe index), generates
sequence-position timestamps, includes full sequence in .inter, and copies the original
pickle. CSV mapping expectations differ (item_id/asin/title versus iAgent index).
Dataset uses leave-two-out: train = all but last two; valid = penultimate; test = last.
Actual raw identifier equivalence cannot be confirmed until raw data is supplied.

`ranked_lists` is loaded but trainer evaluation at lines 306-318 and 539-552 samples
negatives and shuffles new candidates. Thus same dataset name does not establish
paired comparability. Books agent config: 10 candidates, 1000 sampled users, seed 42,
Azure gpt-4o-mini, API 2024-08-01-preview; exact deployment/snapshot unknown.
Common config has 100 candidates; record resolved overrides.
Memory and recommender providers can differ. Books config uses k=16, tau=1800,
N_f=7, mix users/items=4/6, llm_rules, LLM reranker, temperature=0, max_tokens=4000,
warmup one round, fanout=8, evaluation feedback=gt, parallel workers=16.

Stage-R: official graph → domain rule pruner → packer → manager.
Stage-ReRank: official LLMReranker, sort scores descending.
Stage-W: MemRecAgent.write performs user/item overwrites and neighbor propagation.
Trainer warmup uses `-(offset+1)` in complete history, reranks then CLICK feedback
only for target in top 10. Evaluation may write ground-truth feedback; v1 offline
comparison explicitly disables this and records the protocol deviation.
Manager/reranker swallow exceptions and supply fallback results: wrapper must detect
bridge errors rather than count these as successful paid inference.

Use official components loaded under a private module namespace. Avoid importing
unused torch MLP/encoder modules for the rules+LLM path; this does not replace rules
or prompts. Warmup sampling/order and train-only graph boundaries must be manifested.
