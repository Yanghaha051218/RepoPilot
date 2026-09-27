"""Reproducibility checks and a data-driven paper-style report."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .agent import REPOPILOT_PROMPT, SYSTEM_PROMPT
from .benchmark import DEVELOPMENT_SPLIT_VERSION, HELDOUT_SPLIT_VERSION, _held_out
from .experiments import _source_hash


def audit_experiment(bench: Path, results: Path, summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(bench.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {}
    checks["benchmark_hash_matches"] = digest == summary.get("benchmark_sha256")
    issue_map_hash = summary.get("issue_map_sha256")
    checks["issue_map_hash_recorded"] = isinstance(issue_map_hash, str) and bool(
        re.fullmatch(r"[0-9a-f]{64}", issue_map_hash)
    )
    checks["results_hash_matches"] = hashlib.sha256(results.read_bytes()).hexdigest() == summary.get("results_sha256")
    checks["source_hash_matches"] = _source_hash() == summary.get("source_sha256")
    checks["failed_runs_retained"] = sum(row.get("status") == "failed" for row in rows) == summary.get("failed_runs")
    split_version = summary.get("split_version")
    if split_version == HELDOUT_SPLIT_VERSION:
        task_ids = set(summary.get("evaluation_task_ids", []))
    else:
        task_ids = set(summary.get("development_task_ids", []))
    row_task_ids = {row.get("task_id") for row in rows}
    checks["manifest_task_ids_match"] = task_ids == row_task_ids
    checks["task_manifest_unique"] = len(task_ids) == len(
        summary.get("development_task_ids", summary.get("evaluation_task_ids", []))
    )
    checks["split_version_known"] = split_version in (DEVELOPMENT_SPLIT_VERSION, HELDOUT_SPLIT_VERSION)
    checks["formal_run_uses_heldout_repositories"] = split_version == HELDOUT_SPLIT_VERSION
    if split_version == DEVELOPMENT_SPLIT_VERSION:
        checks["split_respects_holdout"] = not any(_held_out(task_id) for task_id in task_ids)
    elif split_version == HELDOUT_SPLIT_VERSION:
        checks["split_respects_holdout"] = bool(task_ids) and all(_held_out(task_id) for task_id in task_ids)
    else:
        checks["split_respects_holdout"] = False
    keys = [(row.get("task_id"), row.get("method"), row.get("token_budget")) for row in rows]
    checks["no_duplicate_runs"] = len(keys) == len(set(keys))
    arm_sets = {
        frozenset((row.get("method"), row.get("token_budget")) for row in rows if row.get("task_id") == task_id)
        for task_id in task_ids
    }
    checks["all_tasks_share_methods_and_budgets"] = bool(rows) and len(arm_sets) == 1
    methods = {row.get("method") for row in rows}
    required_methods = {
        "random", "grep", "bm25", "embedding", "fixed_agent", "repopilot",
        "repopilot_no_explicit_state", "repopilot_no_find_references",
        "repopilot_no_find_tests", "repopilot_no_adaptive_stop",
        "repopilot_no_information_gap",
    }
    checks["required_experiment_arms_present"] = required_methods.issubset(methods)
    checks["planned_ablations_included"] = summary.get("config", {}).get("ablations") is True
    expected_arms = {(method, budget) for method in required_methods
                     for budget in summary.get("config", {}).get("token_budgets", [])}
    checks["planned_arms_match_every_task"] = bool(expected_arms) and arm_sets == {
        frozenset(expected_arms) for _ in task_ids
    }
    commits = defaultdict(set)
    for row in rows:
        if row.get("base_commit"):
            commits[row.get("task_id")].add(row["base_commit"])
    checks["one_pinned_commit_per_task"] = (
        all(len(commits[task_id]) == 1 for task_id in task_ids)
        and all(re.fullmatch(r"[0-9a-fA-F]{40}", next(iter(commits[task_id]), "")) for task_id in task_ids)
    )
    seeds = {row.get("seed") for row in rows}
    checks["seed_matches_manifest"] = seeds == {summary.get("seed")}
    checks["retrieved_context_within_budget"] = all(
        row.get("status") == "failed"
        or (isinstance(row.get("metrics", {}).get("context_tokens_read"), (int, float))
            and row["metrics"]["context_tokens_read"] <= row.get("token_budget", -1))
        for row in rows
    )
    agent_models = {row.get("model") for row in rows
                    if (row.get("method") == "fixed_agent" or str(row.get("method", "")).startswith("repopilot"))
                    and row.get("status") == "ok"}
    manifest_model = summary.get("config", {}).get("model")
    checks["same_model_for_agent_arms"] = len(agent_models) <= 1 and (
        not agent_models or agent_models == {manifest_model}
    )
    checks["agent_model_recorded"] = isinstance(manifest_model, str) and bool(manifest_model)
    checks["prompts_match_manifest"] = summary.get("config", {}).get("prompts_sha256") == {
        "fixed_agent": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "repopilot": hashlib.sha256(REPOPILOT_PROMPT.encode()).hexdigest(),
    }
    checks["no_missing_base_commit_fields"] = len(commits) == len(task_ids)
    checks["contextbench_generalization_included"] = bool(summary.get("contextbench_results"))
    checks["downstream_patch_results_included"] = bool(summary.get("downstream_coding_results"))
    checks["manual_failure_review_included"] = bool(summary.get("manual_failure_review"))
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tasks": len(task_ids),
        "runs": len(rows),
        "failed_runs": sum(row.get("status") == "failed" for row in rows),
        "benchmark_sha256": digest,
    }


def render_report(summary: dict, rows, audit: dict) -> str:
    results = summary.get("results", {})
    lines = [
        "# RepoPilot: Budget-Aware Repository Exploration",
        "",
        "## 1. Introduction",
        "",
        "This report studies whether adaptive repository exploration can retrieve issue-relevant context under a fixed context budget. Results are descriptive and should not be interpreted as downstream patch success.",
        "",
        "## 2. Related Work",
        "",
        "SWE-Explore provides trajectory-grounded line-level context labels ([paper](https://arxiv.org/abs/2606.07297), [dataset](https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench)). ContextBench provides an independent human-annotated context-retrieval benchmark ([paper](https://arxiv.org/abs/2602.05892), [dataset](https://huggingface.co/datasets/Contextbench/ContextBench)).",
        "",
        "## 3. Problem Definition",
        "",
        "Given an issue, a frozen repository snapshot, and a context budget, return ranked repository regions and a stopping decision. The primary outcome is line-level context coverage under cost constraints.",
        "",
        "## 4. RepoPilot",
        "",
        "RepoPilot maintains explicit exploration state, uses read-only repository tools, tracks returned context cost, prevents duplicate reads, and stops when evidence is sufficient or exploration saturates.",
        "",
        "## 5. Experimental Setup",
        "",
        "- Benchmark SHA-256: {}".format(summary.get("benchmark_sha256", "missing")),
        "- Code SHA-256: {}".format(summary.get("source_sha256", "missing")),
        "- Evaluation tasks: {}".format(len(summary.get("development_task_ids", summary.get("evaluation_task_ids", [])))),
        "- Random seed: {}".format(summary.get("seed", "missing")),
        "- Context budget: {}".format(summary.get("token_budget_definition", "unspecified")),
        "- Model: {}".format(summary.get("config", {}).get("model") or "static methods only"),
        "- Scientific audit: **{}**".format(audit.get("status", "not run")),
        "- Failed runs retained: {}".format(summary.get("failed_runs", "unknown")),
        "",
        "## 6. Results",
        "",
        "| Method and budget | Span recall | Precision | F1 | MRR | NDCG | Context tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metrics in sorted(results.items()):
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} |".format(
            key, metrics.get("span_recall", 0), metrics.get("precision", 0), metrics.get("f1", 0),
            metrics.get("mrr", 0), metrics.get("ndcg", 0), metrics.get("context_tokens_read", 0),
        ))
    if not results:
        lines.append("| No benchmark runs | — | — | — | — | — | — |")
    has_ablations = any(key.startswith("repopilot_no_") for key in results)
    lines.extend(["", "## 7. Ablations", "",
                  "Ablation methods are included in the results table above." if has_ablations
                  else "Ablation runs are not included in this result file.", "",
                  "## 8. Failure Analysis", ""])
    failures = Counter(row.get("failure_type") for row in rows if row.get("failure_type"))
    lines.append("Automatic triage labels (review trajectories before drawing conclusions):")
    if failures:
        lines.extend("- {}: {}".format(name, count) for name, count in sorted(failures.items()))
    else:
        lines.append("- No labeled failures in the supplied runs.")
    lines.extend(["", "## 9. Downstream Coding", "",
                  "Patch generation and test-passing results are not included.", "",
                  "## 10. Limitations", "",
                  "The dependency-free embedding control uses lexical hashing, not pretrained semantic embeddings. Retrieved-context tokens use a lexical approximation. This report does not establish that better retrieval improves patch success.",
                  "",
                  "## Audit Details", "",
                  "~~~json",
                  json.dumps(audit, indent=2, sort_keys=True),
                  "~~~", ""])
    return "\n".join(lines)
