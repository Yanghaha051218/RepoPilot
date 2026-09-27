"""Development-split retrieval and agent experiments."""

import json
import hashlib
from collections import defaultdict
from pathlib import Path

from .agent import FixedBudgetAgent, RepoPilot, REPOPILOT_PROMPT, SYSTEM_PROMPT
from .benchmark import (
    DEVELOPMENT_SPLIT_VERSION, HELDOUT_SPLIT_VERSION, load_split_tasks, mapping_sha256,
)
from .evaluation import evaluate, evaluate_trajectory
from .retrieval import chunks, retrieve, tokenize

STATIC_METHODS = ("random", "grep", "bm25", "embedding")


def _source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src" / "repopilot").rglob("*.py")) + [root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def classify_failure(metrics, trajectory, gold):
    """Suggest one failure label for triage; inspect the trajectory before conclusions."""
    recall = metrics.get("span_recall", 0.0)
    if recall == 0:
        return "bad_initial_search"
    if metrics.get("duplicate_reads", 0) > 0:
        return "cyclic_search"
    if metrics.get("stop_reason") == "sufficient_context" and recall < 0.5:
        return "premature_stop"
    if any("test" in region.path.lower() for region in gold):
        found_test = any(
            "test" in row.get("path", "").lower()
            for event in trajectory
            for row in ((event.get("result") or {}).get("lines") or [])
            if isinstance(row, dict)
        )
        if not found_test:
            return "missed_test"
    if metrics.get("waste_ratio", 0.0) >= 0.8:
        return "over_exploration"
    if metrics.get("stop_reason") == "search_saturated":
        return "cyclic_search"
    return None


def _select_static(task, corpus, token_budget, ranked):
    by_key = {(chunk.path, chunk.start): chunk for chunk in corpus}
    selected, used_tokens = [], 0
    for region in ranked:
        chunk = by_key[(region.path, region.start)]
        cost = len(tokenize(chunk.text))
        if used_tokens + cost <= token_budget:
            selected.append(region)
            used_tokens += cost
    return selected, used_tokens


def _compact_spans(spans):
    spans_by_file = defaultdict(list)
    for span in spans:
        if isinstance(span, dict) and isinstance(span.get("path"), str):
            start, end = span.get("start"), span.get("end")
            if type(start) is int and type(end) is int and start > 0 and end >= start:
                spans_by_file[span["path"]].append((start, end))
    compact = {}
    for path, values in sorted(spans_by_file.items()):
        intervals = []
        for start, end in sorted(values):
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        compact[path] = [{"start": start, "end": end} for start, end in intervals]
    return compact


def _add_contextbench_trajectory(record, task):
    steps, all_spans = [], []
    trajectory = record.get("trajectory")
    if trajectory:
        for event in trajectory:
            rows = (event.get("result") or {}).get("lines") or []
            spans = _compact_spans([{
                "path": row.get("path"), "start": row.get("line"), "end": row.get("line"),
            } for row in rows if isinstance(row, dict)])
            if spans:
                steps.append({"files": sorted(spans), "spans": spans})
                all_spans.extend({"path": path, "start": region["start"], "end": region["end"]}
                                 for path, regions in spans.items() for region in regions)
    else:
        regions = record.get("predicted_regions", [])
        spans = _compact_spans(regions)
        if spans:
            steps.append({"files": sorted(spans), "spans": spans})
            all_spans.extend({"path": path, "start": region["start"], "end": region["end"]}
                             for path, items in spans.items() for region in items)
    final_spans = _compact_spans(all_spans)
    record.update({
        "repo_url": task.repo_dir,
        "commit": task.base_commit,
        "model_patch": "",
        "traj_data": {
            "pred_steps": steps,
            "pred_files": sorted(final_spans),
            "pred_spans": final_spans,
        },
    })


def run_experiments(
    bench: Path,
    repos: Path,
    output: Path,
    limit: int = 32,
    seed: int = 17,
    token_budgets=(2048, 4096, 8192, 16384),
    client=None,
    fixed_steps: int = 10,
    max_steps: int = 30,
    ablations: bool = False,
    evaluation_tasks=None,
    split: str = "development",
    commit_map=None,
    issue_map=None,
):
    if evaluation_tasks is None:
        tasks = load_split_tasks(bench, repos, split, seed,
                                 limit if split == "development" else None, commit_map, issue_map)
        selected_split = split
    else:
        tasks = list(evaluation_tasks)
        selected_split = "contextbench"
    if not tasks:
        raise ValueError("no tasks available for experiment")
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregates = defaultdict(list)
    failed_runs = 0
    digest = hashlib.sha256()
    with bench.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    with output.open("w", encoding="utf-8") as sink:
        for task in tasks:
            corpus = chunks(Path(task.repo_dir))
            max_lines = sum(chunk.end - chunk.start + 1 for chunk in corpus)
            ranked_static = {}
            for budget in token_budgets:
                for method in STATIC_METHODS:
                    try:
                        if method not in ranked_static:
                            ranked_static[method] = retrieve(
                                method, task.issue_text, corpus, task.task_id, max_lines, seed=seed,
                            )
                        regions, token_cost = _select_static(task, corpus, budget, ranked_static[method])
                        metrics = evaluate(regions, task.gold_spans, sum(r.end - r.start + 1 for r in regions))
                        metrics["context_tokens_read"] = float(token_cost)
                        record = {
                            "task_id": task.task_id, "base_commit": task.base_commit, "method": method,
                            "token_budget": budget, "seed": seed, "status": "ok", "metrics": metrics,
                            "predicted_regions": [region.__dict__ for region in regions],
                            "failure_type": classify_failure(metrics, [], task.gold_spans),
                        }
                    except Exception as exc:
                        record = {"task_id": task.task_id, "method": method, "token_budget": budget,
                                  "seed": seed, "status": "failed", "error": str(exc)}
                    if selected_split == "contextbench" and record["status"] == "ok":
                        record["instance_id"] = task.task_id
                        _add_contextbench_trajectory(record, task)
                    sink.write(json.dumps(record, sort_keys=True) + "\n")
                    if record["status"] == "ok":
                        aggregates[(method, budget)].append(record["metrics"])
                    else:
                        failed_runs += 1

                if client is not None:
                    agent_runs = [("fixed_agent", None), ("repopilot", None)]
                    if ablations:
                        agent_runs.extend(("repopilot_" + name, name) for name in (
                            "no_explicit_state", "no_find_references", "no_find_tests",
                            "no_adaptive_stop", "no_information_gap",
                        ))
                    for method, ablation in agent_runs:
                        try:
                            if method == "fixed_agent":
                                result = FixedBudgetAgent(client).run(task.issue_text, Path(task.repo_dir),
                                                                      fixed_steps, token_budget=budget)
                            else:
                                result = RepoPilot(client).run(task.issue_text, Path(task.repo_dir),
                                                               budget, max_steps=max_steps, ablation=ablation)
                            metrics = evaluate_trajectory(result["trajectory"], task.gold_spans,
                                                          result.get("tool_usage"), result.get("stop_reason"))
                            record = {
                                "task_id": task.task_id, "base_commit": task.base_commit, "method": method,
                                "token_budget": budget, "seed": seed, "status": "ok", "metrics": metrics,
                                "stop_reason": result.get("stop_reason"), "trajectory": result["trajectory"],
                                "failure_type": classify_failure(metrics, result["trajectory"], task.gold_spans),
                                "model": getattr(client, "model", None),
                            }
                        except Exception as exc:
                            record = {"task_id": task.task_id, "method": method, "token_budget": budget,
                                      "seed": seed, "status": "failed", "error": str(exc)}
                        if selected_split == "contextbench" and record["status"] == "ok":
                            record["instance_id"] = task.task_id
                            _add_contextbench_trajectory(record, task)
                        sink.write(json.dumps(record, sort_keys=True) + "\n")
                        if record["status"] == "ok":
                            aggregates[(method, budget)].append(record["metrics"])
                        else:
                            failed_runs += 1
                    sink.flush()

    numeric_keys = ("file_recall", "file_precision", "span_recall", "region_recall", "precision", "f1",
                    "mrr", "ndcg", "lines_read", "context_tokens_read", "waste_ratio",
                    "context_efficiency", "tool_calls", "exploration_depth", "files_opened",
                    "search_calls", "duplicate_reads", "information_gain_per_step")
    summary = {
        "benchmark": str(bench), "benchmark_sha256": digest.hexdigest(),
        "issue_map_sha256": mapping_sha256(issue_map),
        "source_sha256": _source_hash(),
        "seed": seed,
        "split_version": {"development": DEVELOPMENT_SPLIT_VERSION,
                          "heldout": HELDOUT_SPLIT_VERSION,
                          "contextbench": "contextbench-all-v1"}[selected_split],
        "token_budget_definition": "lexical token approximation over returned code text",
        "config": {"fixed_steps": fixed_steps, "max_steps": max_steps, "token_budgets": list(token_budgets),
                   "ablations": ablations,
                   "model": getattr(client, "model", None),
                   "prompts_sha256": {
                       "fixed_agent": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                       "repopilot": hashlib.sha256(REPOPILOT_PROMPT.encode()).hexdigest(),
                   }},
        "results": {},
        "failed_runs": failed_runs,
    }
    summary["development_task_ids" if selected_split == "development" else "evaluation_task_ids"] = [
        task.task_id for task in tasks
    ]
    for (method, budget), records in sorted(aggregates.items()):
        summary["results"]["{}@{}".format(method, budget)] = {
            key: sum(record[key] for record in records if isinstance(record.get(key), (int, float)))
            / max(sum(isinstance(record.get(key), (int, float)) for record in records), 1)
            for key in numeric_keys if any(isinstance(record.get(key), (int, float)) for record in records)
        }
    summary["results_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
