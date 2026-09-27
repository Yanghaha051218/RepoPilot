import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from repopilot.agent import REPOPILOT_PROMPT, SYSTEM_PROMPT
from repopilot.benchmark import HELDOUT_SPLIT_VERSION, _held_out, mapping_sha256
from repopilot.contextbench import REQUIRED_ARMS, load_contextbench_tasks, run_contextbench, verify_frozen_m9
from repopilot.experiments import _source_hash


class FakeClient:
    model = "frozen-model"

    def complete(self, messages):
        if messages[0]["content"] == SYSTEM_PROMPT:
            return '{"action":"SEARCH_TEXT","arguments":{"query":"Cache"}}'
        if "STOP is disabled" in messages[-1]["content"]:
            return '{"decision":"SEARCH","action":"SEARCH_TEXT","arguments":{"query":"Cache"}}'
        return '{"decision":"STOP","stop_reason":"sufficient_context"}'


class ContextBenchTest(unittest.TestCase):
    def _snapshot(self, root):
        work = root / "work"
        work.mkdir()
        (work / "cache.py").write_text("class Cache:\n    def expire(self):\n        return True\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        subprocess.run(["git", "-C", str(work), "add", "cache.py"], check=True)
        subprocess.run([
            "git", "-C", str(work), "-c", "user.name=RepoPilot", "-c",
            "user.email=repopilot@example.invalid", "commit", "-qm", "snapshot",
        ], check=True)
        commit = subprocess.run(
            ["git", "-C", str(work), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        destination = root / "repos" / "org" / "project" / commit
        destination.parent.mkdir(parents=True)
        work.rename(destination)
        return destination, commit

    def _frozen_m9(self, root, commit):
        repo_number = 1
        task_id = "org__project{}-1".format(repo_number)
        while not _held_out(task_id):
            repo_number += 1
            task_id = "org__project{}-1".format(repo_number)
        rows = []
        for method in sorted(REQUIRED_ARMS):
            row = {"task_id": task_id, "method": method, "token_budget": 100, "seed": 17,
                   "status": "ok", "base_commit": commit,
                   "metrics": {"context_tokens_read": 0}}
            if method not in {"random", "grep", "bm25", "embedding"}:
                row["model"] = "frozen-model"
            rows.append(row)
        result_path = root / "m9.jsonl"
        result_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        bench_path = root / "swe.jsonl"
        bench_path.write_text('{"frozen":"primary benchmark"}\n', encoding="utf-8")
        summary = {
            "source_sha256": _source_hash(),
            "benchmark_sha256": hashlib.sha256(bench_path.read_bytes()).hexdigest(),
            "issue_map_sha256": mapping_sha256({}),
            "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "seed": 17, "split_version": HELDOUT_SPLIT_VERSION,
            "evaluation_task_ids": [task_id],
            "failed_runs": 0,
            "config": {
                "model": "frozen-model", "fixed_steps": 1, "max_steps": 1,
                "token_budgets": [100], "ablations": True,
                "prompts_sha256": {
                    "fixed_agent": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                    "repopilot": hashlib.sha256(REPOPILOT_PROMPT.encode()).hexdigest(),
                },
            },
        }
        summary_path = root / "m9.summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path, result_path, bench_path

    def test_frozen_cross_benchmark_run_uses_all_rows_and_same_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, commit = self._snapshot(root)
            bench = root / "contextbench.jsonl"
            bench.write_text(json.dumps({
                "instance_id": "ctx-1", "repo": "org/project", "base_commit": commit,
                "language": "python", "problem_statement": "Find Cache expiration.",
                "gold_context": json.dumps([{
                    "file": "cache.py", "start_line": 1, "end_line": 3,
                    "content": "gold text is evaluation-only",
                }]),
                "patch": "must never enter the model", "test_patch": "must never enter the model",
            }) + "\n", encoding="utf-8")
            tasks = load_contextbench_tasks(bench, root / "repos")
            self.assertEqual(Path(tasks[0].repo_dir), repo.resolve())
            self.assertEqual(tasks[0].gold_spans[0].path, "cache.py")
            self.assertEqual(tasks[0].issue_text, "Find Cache expiration.")

            frozen_summary, frozen_results, frozen_bench = self._frozen_m9(root, commit)
            self.assertEqual(verify_frozen_m9(frozen_summary, frozen_results, frozen_bench)["config"]["model"], "frozen-model")
            output = root / "context-results.jsonl"
            summary = run_contextbench(
                bench, root / "repos", output, frozen_summary, frozen_results,
                frozen_bench, FakeClient(),
            )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), len(REQUIRED_ARMS))
            self.assertEqual(summary["split_version"], "contextbench-all-v1")
            self.assertEqual(summary["evaluation_task_ids"], ["ctx-1"])
            self.assertEqual(summary["failed_runs"], 0, records)
            self.assertEqual(summary["results_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            static_prediction = records[0]
            self.assertEqual(static_prediction["instance_id"], "ctx-1")
            self.assertEqual(static_prediction["commit"], commit)
            self.assertEqual(static_prediction["traj_data"]["pred_files"], ["cache.py"])
            self.assertEqual(static_prediction["traj_data"]["pred_spans"]["cache.py"],
                             [{"start": 1, "end": 3}])

    def test_frozen_gate_rejects_modified_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, commit = self._snapshot(root)
            summary, results, bench = self._frozen_m9(root, commit)
            results.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "results hash"):
                verify_frozen_m9(summary, results, bench)


if __name__ == "__main__":
    unittest.main()
