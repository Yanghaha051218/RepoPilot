# RepoPilot

RepoPilot studies adaptive, budget-aware repository exploration for coding agents.

Given an issue, repository snapshot, and context budget, it returns ranked files and line spans, an exploration trajectory, and a stop reason. It does not generate patches.

## Quick start

~~~sh
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m repopilot --help
~~~

Download the [SWE-Explore dataset](https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench) and its repository snapshots at the published base commits, then run:

The released benchmark rows omit <code>base_commit</code>; provide the official instance-to-commit JSON map with <code>--commit-map</code>. The [official builder](https://github.com/Qiushao-E/SWE-Explore-Bench/blob/main/build_commit_map.py) produces this map. RepoPilot requires clean Git checkouts so it can verify HEAD against that map. Archive-only snapshots without Git metadata are rejected.

~~~sh
PYTHONPATH=src python -m repopilot baseline \
  --bench data/bench.jsonl --repos data/repos \
  --commit-map data/commit_map.json \
  --issue-map data/issue_map.json \
  --output results/dev.jsonl --limit 32 --seed 17 \
  --line-budgets 50,100,200,400
PYTHONPATH=src python -m repopilot experiment \
  --bench data/bench.jsonl --repos data/repos \
  --commit-map data/commit_map.json \
  --issue-map data/issue_map.json \
  --output results/m9.jsonl --seed 17 \
  --include-agents --ablations
~~~

The published SWE-Explore rows omit both <code>base_commit</code> and issue text, so supply the official instance-to-commit JSON map and an <code>instance_id → issue text</code> JSON map as <code>--commit-map</code> and <code>--issue-map</code>. Rows from other sources may include <code>base_commit</code>, <code>problem_statement</code>, or <code>issue_text</code> directly. The adapter requires a clean local Git snapshot whose HEAD exactly matches the full 40-character commit SHA, normalizes records into RepoPilot tasks, and does not pass gold spans to retrieval methods. A fixed repository-level hash split reserves 20% for held-out evaluation; all issues from one SWE-style repository stay together. The baseline command uses a 32-task development sample; the M9 experiment command evaluates every held-out task and requires snapshots for those tasks.

## Methods

- Random, text search, BM25-style chunk retrieval, and deterministic hashing-vector retrieval rank identical 20-line chunks under equal line budgets.
- Hashing vectors are a dependency-free lexical dense-vector control, **not a pretrained semantic embedding model**. Do not describe it as semantic retrieval.
- Repository tools are read-only: <code>SEARCH_TEXT</code>, <code>SEARCH_SYMBOL</code>, <code>OPEN</code>, <code>FIND_REFERENCES</code>, <code>FIND_TESTS</code>, and <code>STOP</code>. The fixed-call agent forbids early stop; RepoPilot carries explicit state and can stop adaptively.
- Agent commands need an OpenAI-compatible endpoint. Set <code>OPENAI_API_KEY</code>, <code>REPOPILOT_MODEL</code>, and optionally <code>REPOPILOT_API_BASE</code>.

~~~sh
PYTHONPATH=src python -m repopilot agent --issue "Describe the bug" \
  --repo data/repos/project --output results/fixed.json --steps 10
PYTHONPATH=src python -m repopilot adaptive --issue "Describe the bug" \
  --repo data/repos/project --output results/repopilot.json --token-budget 8000
~~~

The adaptive budget counts a documented lexical approximation over returned code text, not the model tokenizer's exact count.

The experiment runner ranks static methods and both agent modes under a shared lexical-token budget, records every failed run, stores benchmark, issue-map, and code hashes, prompt hashes, task commits, and aggregates metrics. <code>--include-agents</code> runs the two LLM arms; <code>--ablations</code> adds the five planned ablations. Failure labels are automatic triage suggestions; inspect trajectories before reporting a failure analysis.

For M11, export ContextBench records as JSONL with its published <code>instance_id</code>, <code>repo</code>, <code>base_commit</code>, <code>language</code>, <code>problem_statement</code>, and <code>gold_context</code> columns. Place each local snapshot at <code>data/context-repos/&lt;owner&gt;/&lt;repo&gt;/&lt;40-character-commit&gt;</code>; the adapter verifies <code>git rev-parse HEAD</code>. It reads the issue and gold file/line spans; patch fields are ignored. Run M9 first with agents and every planned ablation, then use those frozen artifacts for the cross-benchmark run:

~~~sh
PYTHONPATH=src python -m repopilot contextbench \
  --bench data/contextbench.jsonl --repos data/context-repos \
  --frozen-bench data/bench.jsonl \
  --frozen-results results/m9.jsonl \
  --frozen-summary results/m9.jsonl.summary.json \
  --output results/contextbench.jsonl
~~~

The command evaluates every supplied ContextBench task; it does not take a development sample or tune settings. It refuses to run if M9 artifacts are incomplete, failed, edited, or use a different code version, model, prompt, split, budget, or repository commit. Output rows include the official [`traj_data` shape](https://github.com/EuniAI/ContextBench/blob/main/docs/agents.md) for independent trajectory evaluation.

After all arms finish, run the scientific consistency checks and render the paper-style report:

~~~sh
PYTHONPATH=src python -m repopilot release \
  --bench data/bench.jsonl --results results/m9.jsonl \
  --summary results/m9.jsonl.summary.json --report results/report.md
~~~

The release audit requires every task/budget arm, pinned base commits, held-out separation, matching model and prompts, and all six main methods plus the five planned ablations. A final-paper audit also requires ContextBench results, downstream patch results, and a manual failure review. Those later-stage inputs are currently absent, so the command will render a clearly marked draft and exit non-zero.

## Stage status

M0–M2 are complete. The official SWE-Explore JSONL is present at `data/bench.jsonl` (848 rows; SHA-256 `dc4f114ececd0bfb987361c26ae5e2440456e2cccb36adfccb09ea5385aec202`). The accepted seed-17 development sample has 20 clean snapshots at their mapped base commits across 10 repositories; all tasks pass the adapter and repository-held-out check. Its SHA-256 is recorded in the report table.

The four deterministic retrieval baselines use line budgets 50, 100, 200, and 400. The accepted run is saved as `results/dev20.jsonl` (320 rows: 20 tasks × 4 methods × 4 budgets), with aggregate JSON at `results/dev20.jsonl.summary.json` and a report table at `results/dev20.table.md`. The audit verified unique task/method/budget rows, pinned commits, development-only task IDs, finite metrics, and line-budget compliance. A second run with the same inputs produced byte-identical rows and aggregates. The `embedding` control is deterministic lexical hashing, not a pretrained semantic model.

M3–M6 are implemented and self-checked. The read-only repository layer exposes `SEARCH_TEXT`, `SEARCH_SYMBOL`, `OPEN`, `FIND_REFERENCES`, `FIND_TESTS`, and `STOP`. Python files are indexed for imports and AST symbol ranges; other languages use declaration-line matching. `OPEN` returns bounded line spans, and tool logs include arguments, results, lexical-token and line counts, latency, and errors. M4's fixed-action Agent rejects `STOP`, executes exactly its requested number of calls, and its trajectory replays. M5 stores explicit state snapshots. M6 prevents repeated reads of inspected spans and enforces the context budget, including for `OPEN`.

M7's adaptive stop controller and M8's retrieval/trajectory metrics are implemented. Self-checks cover budget exhaustion, no-information-gain stopping, span deduplication, retrieval quality, tool cost, duplicate reads, and stop reasons. The 14 standard-library tests pass without an LLM. The planned fixed-versus-adaptive multi-budget comparison belongs to M9 and has not run.

M9 has a development-only static pilot at `results/m9-static-dev20.jsonl` (320 rows over 20 tasks, four static methods, and four token budgets; zero failed runs). It is not the formal experiment: no fixed-Agent or RepoPilot model arms ran, and no held-out results were used. The formal M9 command needs an OpenAI-compatible model/key and held-out task mappings and snapshots; this environment has no model credentials, while local commit/issue maps cover only the 32-task development candidate set. M10 failure review and M11–M14 therefore remain unstarted; there are no ContextBench measurements, downstream patch results, or passing release audit.
