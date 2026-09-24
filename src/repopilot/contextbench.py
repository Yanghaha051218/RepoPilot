"""ContextBench's JSONL adapter and frozen cross-benchmark evaluation gate."""

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from .agent import REPOPILOT_PROMPT, SYSTEM_PROMPT
from .benchmark import ExplorationTask, Region, _held_out
from .experiments import _source_hash, run_experiments

REQUIRED_ARMS = {
    "random", "grep", "bm25", "embedding", "fixed_agent", "repopilot",
    "repopilot_no_explicit_state", "repopilot_no_find_references",
    "repopilot_no_find_tests", "repopilot_no_adaptive_stop",
    "repopilot_no_information_gap",
}
SHA256 = re.compile(r"^[0-9a-fA-F]{40}$")


def _snapshot(root: Path, repo: str, commit: str) -> Path:
    if not isinstance(repo, str) or not repo:
        raise ValueError("repo must be an owner/name repository slug")
    parts = PurePosixPath(repo).parts
    if (len(parts) != 2 or any(part in ("", ".", "..") for part in parts)
            or "\\" in repo or ":" in parts[0]):
        raise ValueError("repo must be an owner/name repository slug")
    if not isinstance(commit, str) or not SHA256.fullmatch(commit):
        raise ValueError("base_commit must be a full 40-character Git commit SHA")
    root = root.resolve()
    path = (root.joinpath(*parts, commit.lower())).resolve()
    if root not in path.parents:
        raise ValueError("repository snapshot must stay under --repos")
    if not path.is_dir():
        raise FileNotFoundError("ContextBench snapshot not found: {}".format(path))
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("could not verify Git snapshot {}: {}".format(path, exc))
    if head != commit.lower():
        raise ValueError("snapshot HEAD does not match ContextBench base_commit: {}".format(path))
    if status:
        raise ValueError("ContextBench snapshot has uncommitted or untracked files: {}".format(path))
    return path


def adapt_contextbench_record(row, repos_root: Path) -> ExplorationTask:
    """Normalize issue and human gold spans; patch and test_patch are never read."""
    issue = row.get("problem_statement")
    task_id = row.get("instance_id")
    if not isinstance(issue, str) or not issue.strip():
        raise ValueError("task has no problem_statement")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task has no instance_id")
    snapshot = _snapshot(repos_root, row.get("repo"), row.get("base_commit"))
    raw_gold = row.get("gold_context")
    try:
        gold_rows = json.loads(raw_gold) if isinstance(raw_gold, str) else raw_gold
    except json.JSONDecodeError as exc:
        raise ValueError("gold_context is not valid JSON: {}".format(exc))
    if not isinstance(gold_rows, list):
        raise ValueError("gold_context must be a list")
    gold = []
    for item in gold_rows:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("file") or item.get("path")
        if not isinstance(raw_path, str):
            continue
        path = PurePosixPath(raw_path.replace("\\", "/"))
        start, end = item.get("start_line"), item.get("end_line")
        if (not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]
                or type(start) is not int or type(end) is not int or start < 1 or end < start):
            continue
        gold.append(Region(path.as_posix(), start, end))
    if not gold:
        raise ValueError("task has no valid gold_context spans")
    language = row.get("language")
    return ExplorationTask(
        task_id=task_id.strip(), issue_text=issue.strip(), repo_dir=str(snapshot),
        base_commit=row["base_commit"].lower(), gold_spans=gold,
        language=language if isinstance(language, str) else None,
    )


def load_contextbench_tasks(path: Path, repos_root: Path):
    tasks = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                tasks.append(adapt_contextbench_record(json.loads(line), repos_root))
            except Exception as exc:
                raise ValueError("{}:{}: {}".format(path, line_no, exc))
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids in ContextBench")
    return tasks


def verify_frozen_m9(summary_path: Path, results_path: Path, bench_path: Path):
    """Require complete, successful M9 artifacts before opening ContextBench."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if hashlib.sha256(results_path.read_bytes()).hexdigest() != summary.get("results_sha256"):
        raise ValueError("M9 results hash does not match its frozen summary")
    if hashlib.sha256(bench_path.read_bytes()).hexdigest() != summary.get("benchmark_sha256"):
        raise ValueError("M9 benchmark hash does not match its frozen summary")
    issue_map_hash = summary.get("issue_map_sha256")
    if not isinstance(issue_map_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", issue_map_hash):
        raise ValueError("M9 issue map hash is missing from its frozen summary")
    if summary.get("source_sha256") != _source_hash():
        raise ValueError("code changed since M9; rerun and freeze the primary experiment first")
    config = summary.get("config", {})
    prompts = {
        "fixed_agent": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "repopilot": hashlib.sha256(REPOPILOT_PROMPT.encode()).hexdigest(),
    }
    if summary.get("split_version") != "repopilot-heldout-v2":
        raise ValueError("frozen artifact is not an M9 SWE-Explore held-out run")
    if config.get("prompts_sha256") != prompts:
        raise ValueError("M9 prompt hashes do not match the current prompts")
    if not config.get("model") or config.get("ablations") is not True:
        raise ValueError("M9 must include one fixed model and all planned ablations")
    task_ids = summary.get("evaluation_task_ids", [])
    budgets = config.get("token_budgets", [])
    if (not task_ids or any(not isinstance(task_id, str) for task_id in task_ids)
            or len(task_ids) != len(set(task_ids)) or not budgets
            or not all(_held_out(task_id) for task_id in task_ids)
            or any(type(budget) is not int or budget < 1 for budget in budgets)):
        raise ValueError("M9 task split or budget manifest is invalid")
    if any(type(config.get(name)) is not int or config[name] < 1 for name in ("fixed_steps", "max_steps")):
        raise ValueError("M9 action budgets are invalid")
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if summary.get("failed_runs") != 0:
        raise ValueError("M9 contains failed runs; resolve or rerun them before freezing")
    expected = {(task_id, method, budget) for task_id in task_ids
                for method in REQUIRED_ARMS for budget in budgets}
    seen = set()
    commits = {}
    for row in rows:
        key = row.get("task_id"), row.get("method"), row.get("token_budget")
        if key in seen or key not in expected or row.get("status") != "ok":
            raise ValueError("M9 has a duplicate, failed, or unexpected experiment arm")
        seen.add(key)
        if row.get("seed") != summary.get("seed"):
            raise ValueError("M9 run seeds do not match the frozen manifest")
        commit = row.get("base_commit")
        if not isinstance(commit, str) or not SHA256.fullmatch(commit):
            raise ValueError("M9 contains a missing or unpinned repository commit")
        commits.setdefault(row["task_id"], set()).add(commit.lower())
        if row["method"] in REQUIRED_ARMS - {"random", "grep", "bm25", "embedding"}:
            if row.get("model") != config["model"]:
                raise ValueError("M9 agent arms do not use the frozen model")
        cost = (row.get("metrics") or {}).get("context_tokens_read")
        if not isinstance(cost, (int, float)) or cost > row["token_budget"]:
            raise ValueError("M9 retrieved context exceeds its declared token budget")
    if seen != expected or set(commits) != set(task_ids) or any(len(value) != 1 for value in commits.values()):
        raise ValueError("M9 does not contain every successful arm on one pinned commit per task")
    return summary


def run_contextbench(bench: Path, repos: Path, output: Path, frozen_summary: Path,
                     frozen_results: Path, frozen_bench: Path, client, frozen=None):
    frozen = frozen or verify_frozen_m9(frozen_summary, frozen_results, frozen_bench)
    config = frozen["config"]
    if client.model != config["model"]:
        raise ValueError("ContextBench must use the exact M9 model")
    tasks = load_contextbench_tasks(bench, repos)
    return run_experiments(
        bench, repos, output, seed=frozen["seed"], token_budgets=config["token_budgets"],
        client=client, fixed_steps=config["fixed_steps"], max_steps=config["max_steps"],
        ablations=True, evaluation_tasks=tasks,
    )
