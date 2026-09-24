"""SWE-Explore adapter and development subset selection."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Region:
    path: str
    start: int
    end: int


@dataclass(frozen=True)
class ExplorationTask:
    task_id: str
    issue_text: str
    repo_dir: str
    base_commit: Optional[str]
    gold_spans: List[Region]
    language: Optional[str] = None


def _safe_repo_path(root: Path, repo_dir: str) -> str:
    root = root.resolve()
    path = Path(repo_dir)
    candidate = (path if path.is_absolute() else root / path).resolve()
    # SWE-Explore's repo_dir sometimes includes the configured root's basename.
    if not candidate.is_dir() and not path.is_absolute() and path.parts and path.parts[0] == root.name:
        candidate = root.joinpath(*path.parts[1:]).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("repo_dir must stay under --repos")
    if not candidate.is_dir():
        raise FileNotFoundError("repository snapshot not found: {}".format(candidate))
    return str(candidate)


def adapt_record(row: Dict[str, Any], repos_root: Path, commit_map=None, issue_map=None) -> ExplorationTask:
    """Normalize a SWE-Explore record. Gold spans are kept out of retrieval inputs."""
    task_id, repo_dir = row.get("instance_id") or row.get("task_id"), row.get("repo_dir")
    issue = row.get("problem_statement") or row.get("issue_text") or (issue_map or {}).get(task_id)
    if not isinstance(issue, str) or not issue.strip():
        raise ValueError("task has no problem_statement or issue_text")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task has no instance_id")
    if not isinstance(repo_dir, str) or not repo_dir:
        raise ValueError("task has no repo_dir; fetch its base-commit snapshot first")
    truth = row.get("ground_truth") or {}
    regions = truth.get("read_core_regions") or truth.get("gold_spans") or []
    gold = []
    for region in regions:
        path = region.get("path") or region.get("file")
        start, end = region.get("start"), region.get("end")
        if isinstance(path, str) and type(start) is int and type(end) is int and start > 0 and end >= start:
            gold.append(Region(path.replace("\\", "/"), start, end))
    if not gold:
        raise ValueError("task has no valid core gold regions")
    meta = row.get("meta") or {}
    base_commit = row.get("base_commit") or meta.get("base_commit") or (commit_map or {}).get(task_id)
    snapshot = Path(_safe_repo_path(repos_root, repo_dir))
    if not isinstance(base_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", base_commit):
        raise ValueError("task must provide a full base_commit SHA")
    try:
        head = subprocess.run(
            ["git", "-C", str(snapshot), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", str(snapshot), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("could not verify Git repository snapshot: {}".format(exc))
    if head != base_commit.lower():
        raise ValueError("repository HEAD does not match base_commit")
    if status:
        raise ValueError("repository snapshot has uncommitted or untracked files")
    return ExplorationTask(
        task_id=task_id,
        issue_text=issue.strip(),
        repo_dir=str(snapshot),
        base_commit=base_commit.lower(),
        gold_spans=gold,
        language=meta.get("language"),
    )


def load_commit_map(path: Optional[Path]):
    if path is None:
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or any(
        not isinstance(task_id, str) or not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-fA-F]{40}", commit)
        for task_id, commit in mapping.items()
    ):
        raise ValueError("commit map must map task IDs to full Git commit SHAs")
    return mapping


def load_issue_map(path: Optional[Path]):
    if path is None:
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or any(
        not isinstance(task_id, str) or not isinstance(issue, str) or not issue.strip()
        for task_id, issue in mapping.items()
    ):
        raise ValueError("issue map must map task IDs to non-empty issue text")
    return mapping


def mapping_sha256(mapping):
    payload = json.dumps(mapping or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_tasks(path: Path, repos_root: Path, commit_map=None, issue_map=None) -> List[ExplorationTask]:
    tasks = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                tasks.append(adapt_record(json.loads(line), repos_root, commit_map, issue_map))
            except Exception as exc:
                raise ValueError("{}:{}: {}".format(path, line_no, exc))
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids in benchmark")
    return tasks


def load_split_tasks(path: Path, repos_root: Path, split: str, seed: int,
                     limit: Optional[int] = None, commit_map=None, issue_map=None):
    """Select task records before requiring snapshots, so only the chosen split needs checkouts."""
    if split not in ("development", "heldout"):
        raise ValueError("split must be development or heldout")
    if split == "development" and (type(limit) is not int or limit < 1):
        raise ValueError("development split requires a positive limit")
    prefix = str(seed).encode("utf-8")
    records, ids = [], set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row.get("instance_id") or row.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("task has no instance_id")
                if task_id in ids:
                    raise ValueError("duplicate task id in benchmark")
                ids.add(task_id)
                if _held_out(task_id) == (split == "heldout"):
                    records.append((task_id, line_no, row))
            except Exception as exc:
                raise ValueError("{}:{}: {}".format(path, line_no, exc))
    records.sort(key=lambda item: hashlib.sha256(prefix + item[0].encode("utf-8")).digest())
    if split == "development":
        records = records[:limit]
    tasks = []
    for _, line_no, row in records:
        try:
            tasks.append(adapt_record(row, repos_root, commit_map, issue_map))
        except Exception as exc:
            raise ValueError("{}:{}: {}".format(path, line_no, exc))
    return tasks


def development_subset(tasks: Iterable[ExplorationTask], limit: int, seed: int) -> List[ExplorationTask]:
    """Choose a repeatable development sample after reserving repository-held-out tasks."""
    if limit < 1:
        raise ValueError("limit must be positive")
    prefix = str(seed).encode("utf-8")
    development = [task for task in tasks if not _held_out(task.task_id)]
    return sorted(development, key=lambda task: hashlib.sha256(prefix + task.task_id.encode("utf-8")).digest())[:limit]


def _held_out(task_id: str) -> bool:
    # SWE-style IDs are owner__repo-issue; all issues from a repository stay together.
    if "__" not in task_id:
        return False
    owner, repository = task_id.split("__", 1)
    repository = owner + "/" + repository.rsplit("-", 1)[0]
    key = hashlib.sha256(("repopilot-heldout-v2|" + repository).encode("utf-8")).digest()
    return int.from_bytes(key[:4], "big") % 5 == 0
