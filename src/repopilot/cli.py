"""Command-line entry point for reproducible static baselines."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .benchmark import load_commit_map, load_issue_map, load_split_tasks, mapping_sha256
from .agent import FixedBudgetAgent, RepoPilot
from .audit import audit_experiment, render_report
from .contextbench import run_contextbench, verify_frozen_m9
from .evaluation import evaluate
from .experiments import run_experiments
from .llm import ChatClient
from .retrieval import chunks, retrieve

METHODS = ("random", "grep", "bm25", "embedding")


def run_baselines(bench: Path, repos: Path, output: Path, limit: int, seed: int,
                 budgets: List[int], commit_map=None, issue_map=None) -> None:
    tasks = load_split_tasks(bench, repos, "development", seed, limit, commit_map, issue_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
    with output.open("w", encoding="utf-8") as sink:
        for task in tasks:
            corpus = chunks(Path(task.repo_dir))
            for method in METHODS:
                ranked = retrieve(method, task.issue_text, corpus, task.task_id, max(budgets), seed)
                for budget in budgets:
                    selected, used = [], 0
                    for region in ranked:
                        cost = region.end - region.start + 1
                        if used + cost <= budget:
                            selected.append(region)
                            used += cost
                    metrics = evaluate(selected, task.gold_spans, budget)
                    sink.write(json.dumps({
                        "task_id": task.task_id,
                        "base_commit": task.base_commit,
                        "method": method,
                        "budget_lines": budget,
                        "seed": seed,
                        "development_limit": limit,
                        "metrics": metrics,
                        "predicted_regions": [region.__dict__ for region in selected],
                    }, sort_keys=True) + "\n")
                    summaries[(method, budget)].append(metrics)
    summary = {
        "benchmark": str(bench),
        "issue_map_sha256": mapping_sha256(issue_map),
        "seed": seed,
        "development_task_ids": [task.task_id for task in tasks],
        "methods": list(METHODS),
        "budgets_lines": budgets,
        "results": {
            "{}@{}".format(method, budget): {
                key: sum(row[key] for row in rows) / len(rows)
                for key in rows[0]
            }
            for (method, budget), rows in sorted(summaries.items())
        },
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="repopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="run the four static retrieval controls")
    baseline.add_argument("--bench", type=Path, required=True)
    baseline.add_argument("--repos", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--commit-map", type=Path, default=None)
    baseline.add_argument("--issue-map", type=Path, default=None)
    baseline.add_argument("--limit", type=int, default=32)
    baseline.add_argument("--seed", type=int, default=17)
    baseline.add_argument("--line-budgets", default="50,100,200,400")
    agent = subparsers.add_parser("agent", help="run a fixed-action-count agent baseline")
    agent.add_argument("--issue", required=True)
    agent.add_argument("--repo", type=Path, required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--steps", type=int, default=10)
    agent.add_argument("--model", default=None)
    agent.add_argument("--api-base", default=None)
    agent.add_argument("--api-key", default=None)
    adaptive = subparsers.add_parser("adaptive", help="run the stateful budget-aware RepoPilot controller")
    adaptive.add_argument("--issue", required=True)
    adaptive.add_argument("--repo", type=Path, required=True)
    adaptive.add_argument("--output", type=Path, required=True)
    adaptive.add_argument("--token-budget", type=int, default=8000)
    adaptive.add_argument("--max-steps", type=int, default=30)
    adaptive.add_argument("--model", default=None)
    adaptive.add_argument("--api-base", default=None)
    adaptive.add_argument("--api-key", default=None)
    experiment = subparsers.add_parser("experiment", help="run budget-matched retrieval and agent experiments")
    experiment.add_argument("--bench", type=Path, required=True)
    experiment.add_argument("--repos", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--commit-map", type=Path, default=None)
    experiment.add_argument("--issue-map", type=Path, default=None)
    experiment.add_argument("--seed", type=int, default=17)
    experiment.add_argument("--token-budgets", default="2048,4096,8192,16384")
    experiment.add_argument("--include-agents", action="store_true")
    experiment.add_argument("--ablations", action="store_true")
    experiment.add_argument("--steps", type=int, default=10)
    experiment.add_argument("--max-steps", type=int, default=30)
    experiment.add_argument("--model", default=None)
    experiment.add_argument("--api-base", default=None)
    experiment.add_argument("--api-key", default=None)
    release = subparsers.add_parser("release", help="audit experiment records and render a paper-style report")
    release.add_argument("--bench", type=Path, required=True)
    release.add_argument("--results", type=Path, required=True)
    release.add_argument("--summary", type=Path, required=True)
    release.add_argument("--report", type=Path, required=True)
    release.add_argument("--audit", type=Path, default=None)
    contextbench = subparsers.add_parser("contextbench", help="evaluate a frozen M9 method on ContextBench")
    contextbench.add_argument("--bench", type=Path, required=True)
    contextbench.add_argument("--repos", type=Path, required=True)
    contextbench.add_argument("--output", type=Path, required=True)
    contextbench.add_argument("--frozen-bench", type=Path, required=True)
    contextbench.add_argument("--frozen-summary", type=Path, required=True)
    contextbench.add_argument("--frozen-results", type=Path, required=True)
    contextbench.add_argument("--api-base", default=None)
    contextbench.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.command == "release":
        import sys
        audit = audit_experiment(args.bench, args.results, args.summary)
        rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(summary, rows, audit), encoding="utf-8")
        audit_path = args.audit or args.results.with_suffix(args.results.suffix + ".audit.json")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("{}: {}".format(audit["status"], audit_path))
        if audit["status"] != "PASS":
            sys.exit(1)
        print("Report: {}".format(args.report))
        return
    if args.command == "contextbench":
        frozen = verify_frozen_m9(args.frozen_summary, args.frozen_results, args.frozen_bench)
        client = ChatClient(frozen["config"]["model"], args.api_base, args.api_key)
        summary = run_contextbench(args.bench, args.repos, args.output,
                                   args.frozen_summary, args.frozen_results, args.frozen_bench,
                                   client, frozen=frozen)
        print("ContextBench runs: {}; failed: {}; summary: {}".format(
            len(summary["evaluation_task_ids"]), summary["failed_runs"],
            args.output.with_suffix(args.output.suffix + ".summary.json")))
        return
    if args.command == "experiment":
        if args.ablations and not args.include_agents:
            parser.error("--ablations requires --include-agents")
        try:
            budgets = sorted({int(value) for value in args.token_budgets.split(",")})
        except ValueError:
            parser.error("--token-budgets must be comma-separated integers")
        if not budgets or budgets[0] < 1:
            parser.error("token budgets must be positive integers")
        client = None
        if args.include_agents:
            model = args.model
            if not model:
                import os
                model = os.environ.get("REPOPILOT_MODEL")
            client = ChatClient(model, args.api_base, args.api_key)
        commit_map = load_commit_map(args.commit_map)
        issue_map = load_issue_map(args.issue_map)
        run_experiments(args.bench, args.repos, args.output, seed=args.seed, token_budgets=budgets,
                        client=client, fixed_steps=args.steps, max_steps=args.max_steps,
                        ablations=args.ablations, split="heldout", commit_map=commit_map,
                        issue_map=issue_map)
        return
    if args.command in ("agent", "adaptive"):
        model = args.model
        if not model:
            import os
            model = os.environ.get("REPOPILOT_MODEL")
        client = ChatClient(model, args.api_base, args.api_key)
        if args.command == "agent":
            result = FixedBudgetAgent(client).run(args.issue, args.repo, args.steps)
        else:
            result = RepoPilot(client).run(args.issue, args.repo, args.token_budget, args.max_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    try:
        budgets = sorted({int(value) for value in args.line_budgets.split(",")})
    except ValueError:
        parser.error("--line-budgets must be comma-separated integers")
    if not budgets or budgets[0] < 1:
        parser.error("line budgets must be positive integers")
    run_baselines(args.bench, args.repos, args.output, args.limit, args.seed, budgets,
                  load_commit_map(args.commit_map), load_issue_map(args.issue_map))


if __name__ == "__main__":
    main()
