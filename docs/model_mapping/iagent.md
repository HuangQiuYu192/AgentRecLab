# iAgent / i²Agent audit — 2026-10-12

Source: https://github.com/agiresearch/iAgent/tree/e546d17848fadbcaf00e4022aa949289d7e7900b
Python 3.9 per README; pandas 2.2.2, numpy 1.26.4, openai 1.43.0 per requirements.
The requirements also list standard-library modules as packages and cannot be installed verbatim.
No LICENSE file found: redistribution permission unknown. Use local pinned checkout.

Commands: `python main_iagent_mp.py --dataset amazon --domain books --agent_type static`
and `--agent_type dynamic`; goodreads uses `--dataset good --domain reads`.
Input: `data/booksAll_recagent.pkl` and `data/combined_books_asin_mapping.csv`.
Pickle rows: instruction, title, description, asin, ranked_lists, reviewText.
CSV: index, title, description. Actual downloaded data and original user ID field: unknown.

`main_iagent_mp.py` consumes pre-existing `ranked_lists` without resampling them.
It slices the last interaction as test; all preceding interactions feed static history.
Candidate descriptions use first 20 cleaned characters, titles first 50;
history descriptions use last 200 characters. Preserve these in the wrapper.
`if i > 1: break` after submission limits each 1000-row chunk to three cases.
It samples a warmup negative from catalog minus the complete positive history.

Both classes in `model.py` actually call `gpt-4o-mini` with strict JSON schema;
temperature/max_tokens/snapshot are unspecified in those calls. This is code evidence,
not inference from a paper's data-generation footnote. External knowledge is generated
by the LLM; the inspected calls do not supply a search/tool API.
Static: knowledge → rerank. Dynamic: recommendation simulation → profile from
training feedback → knowledge → dynamic interest/profile → rerank. Dynamic warmup
uses `title[-2]` of the already test-trimmed history; do not silently change this.
Profiles are rebuilt per case, not a persistent cross-test learning loop.
Parsing retries and self-reflection retries are bounded at three each; sleep=5s.
Original repair triggers when the test positive is absent, rather than validating
the whole candidate permutation. A controlled wrapper must remove this label-dependent
criterion; any strict reproduction must disclose the change.

Adapter plan: import the pinned model unchanged, replace its client with a traced
bridge, capture the final ranking and delegate metric computation to the framework.
Runtime metric hook validates the entire permutation without giving the model the
test answer. `paper_original` is retained as the independent upstream command;
framework strict-paper claims are blocked until data and protocol equivalence exist.
