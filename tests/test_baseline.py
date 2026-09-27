import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from repopilot.benchmark import ExplorationTask, Region, _held_out, adapt_record, development_subset, load_commit_map, load_issue_map, load_split_tasks
from repopilot.agent import FixedBudgetAgent, RepoPilot, replay_trajectory
from repopilot.audit import audit_experiment, render_report
from repopilot.cli import run_baselines
from repopilot.evaluation import evaluate, evaluate_trajectory
from repopilot.experiments import classify_failure, run_experiments
from repopilot.retrieval import chunks, retrieve
from repopilot.tools import RepositoryTools


def commit_snapshot(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=RepoPilot", "-c",
        "user.email=repopilot@example.invalid", "commit", "-qm", "snapshot",
    ], check=True)
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


class BaselineTest(unittest.TestCase):
    def test_adapter_subset_and_all_retrievers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repos" / "sample"
            repo.mkdir(parents=True)
            (repo / "cache.py").write_text("class Cache:\n    def expire(self):\n        return True\n", encoding="utf-8")
            commit = commit_snapshot(repo)
            record = {
                "instance_id": "sample-1",
                "repo_dir": "repos/sample",
                "problem_statement": "Fix Cache expire behavior",
                "base_commit": commit,
                "ground_truth": {"read_core_regions": [{"path": "cache.py", "start": 1, "end": 3}]},
            }
            task = adapt_record(record, root / "repos")
            self.assertEqual(task.base_commit, commit)
            wrong_commit = dict(record, base_commit="0" * 40)
            with self.assertRaisesRegex(ValueError, "does not match"):
                adapt_record(wrong_commit, root / "repos")
            no_commit_field = dict(record)
            del no_commit_field["base_commit"]
            mapped = adapt_record(no_commit_field, root / "repos", {task.task_id: commit})
            self.assertEqual(mapped.base_commit, commit)
            commit_map_path = root / "commits.json"
            commit_map_path.write_text(json.dumps({task.task_id: commit}), encoding="utf-8")
            self.assertEqual(load_commit_map(commit_map_path), {task.task_id: commit})
            no_issue_field = dict(record)
            del no_issue_field["problem_statement"]
            issue_map = {task.task_id: "Issue text supplied by official ID mapping"}
            mapped_issue = adapt_record(no_issue_field, root / "repos", {task.task_id: commit}, issue_map)
            self.assertEqual(mapped_issue.issue_text, issue_map[task.task_id])
            issue_map_path = root / "issues.json"
            issue_map_path.write_text(json.dumps(issue_map), encoding="utf-8")
            self.assertEqual(load_issue_map(issue_map_path), issue_map)
            self.assertEqual(development_subset([task], 1, 17), [task])
            corpus = chunks(repo)
            for method in ("random", "grep", "bm25", "embedding"):
                first = retrieve(method, task.issue_text, corpus, task.task_id, 20, seed=17)
                second = retrieve(method, task.issue_text, corpus, task.task_id, 20, seed=17)
                self.assertEqual(first, second)
                self.assertLessEqual(sum(region.end - region.start + 1 for region in first), 20)
            metrics = evaluate([Region("cache.py", 1, 3)], task.gold_spans, 3)
            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["span_recall"], 1.0)
            self.assertEqual(metrics["ndcg"], 1.0)
            self.assertEqual(evaluate([], [Region("cache.py", 1, 3)], 0)["region_recall"], 0.0)
            trajectory_metrics = evaluate_trajectory(
                [{"action": "OPEN", "arguments": {"file": "cache.py"},
                  "result": {"lines": [{"path": "cache.py", "line": 1, "text": "class Cache:"}]}}],
                [Region("cache.py", 1, 1)],
            )
            self.assertEqual(trajectory_metrics["information_gain_per_step"], 1.0)
            bench = root / "bench.jsonl"
            bench.write_text(json.dumps(no_issue_field) + "\n", encoding="utf-8")
            output = root / "results" / "baseline.jsonl"
            run_baselines(bench, root / "repos", output, 1, 17, [20], {task.task_id: commit}, issue_map)
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 4)
            baseline_summary = json.loads(output.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(baseline_summary["issue_map_sha256"]), 64)

    def test_rejects_repo_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                "instance_id": "escape",
                "repo_dir": "../outside",
                "issue_text": "issue",
                "ground_truth": {"read_core_regions": [{"path": "x.py", "start": 1, "end": 1}]},
            }
            with self.assertRaises(ValueError):
                adapt_record(record, root / "repos")

    def test_heldout_split_keeps_repository_issues_together(self):
        def task(task_id):
            return ExplorationTask(task_id, "issue", ".", None, [Region("x.py", 1, 1)])

        issues = [task("org__repo-1"), task("org__repo-2")]
        selected = development_subset(issues, 10, 17)
        self.assertIn(len(selected), (0, 2))
        rebench = [
            "org__repo-" + "a" * 40 + "-v" + "b" * 40,
            "org__repo-" + "c" * 40 + "-v" + "d" * 40,
            "org__repo-" + "e" * 40,
        ]
        self.assertEqual(len({_held_out(task_id) for task_id in rebench}), 1)
        self.assertEqual(_held_out("org__scikit-learn-12"), _held_out("org__scikit-learn-99"))
        with self.assertRaisesRegex(ValueError, "cannot prevent repository leakage"):
            _held_out("org__unrecognized-task")

    def test_development_load_does_not_require_heldout_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repos" / "sample"
            repo.mkdir(parents=True)
            (repo / "cache.py").write_text("class Cache: pass\n", encoding="utf-8")
            commit = commit_snapshot(repo)
            repo_number = 1
            heldout_id = "org__project{}-1".format(repo_number)
            while not _held_out(heldout_id):
                repo_number += 1
                heldout_id = "org__project{}-1".format(repo_number)
            rows = [
                {"instance_id": "dev-1", "repo_dir": "sample", "base_commit": commit,
                 "issue_text": "Find Cache", "ground_truth": {"gold_spans": [
                     {"path": "cache.py", "start": 1, "end": 1}]}},
                {"instance_id": heldout_id, "repo_dir": "not-downloaded", "base_commit": "a" * 40,
                 "issue_text": "Issue", "ground_truth": {"gold_spans": [
                     {"path": "x.py", "start": 1, "end": 1}]}},
            ]
            bench = root / "bench.jsonl"
            bench.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            tasks = load_split_tasks(bench, root / "repos", "development", 17, 1)
            self.assertEqual([task.task_id for task in tasks], ["dev-1"])
            with self.assertRaisesRegex(ValueError, "snapshot not found"):
                load_split_tasks(bench, root / "repos", "heldout", 17)

    def test_budget_matched_experiment_writes_every_arm_and_keeps_failures_classifiable(self):
        class FakeClient:
            model = "test-model"

            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return '{"action":"SEARCH_TEXT","arguments":{"query":"Cache"}}'
                return json.dumps({
                    "decision": "STOP", "stop_reason": "sufficient_context", "confidence": 0.5,
                    "hypothesis": "", "known_facts": [], "relevant_spans": [],
                    "unresolved_questions": [], "expected_gain": "", "action": "", "arguments": {},
                })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repos" / "sample"
            repo.mkdir(parents=True)
            (repo / "cache.py").write_text("class Cache:\n    pass\n", encoding="utf-8")
            commit = commit_snapshot(repo)
            bench = root / "bench.jsonl"
            bench.write_text(json.dumps({
                "instance_id": "sample-1", "repo_dir": "sample",
                "base_commit": commit, "problem_statement": "Find Cache", "ground_truth": {
                    "read_core_regions": [{"path": "cache.py", "start": 1, "end": 1}],
                },
            }) + "\n", encoding="utf-8")
            output = root / "experiment.jsonl"
            summary = run_experiments(bench, root / "repos", output, limit=1, token_budgets=[20],
                                      client=FakeClient(), fixed_steps=1, max_steps=2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 6)
            self.assertEqual(summary["failed_runs"], 0)
            self.assertEqual(len(summary["issue_map_sha256"]), 64)
            self.assertIn("repopilot@20", summary["results"])
            self.assertEqual(classify_failure({"span_recall": 0}, [], []), "bad_initial_search")
            report = audit_experiment(bench, output, output.with_suffix(".jsonl.summary.json"))
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(report["checks"]["issue_map_hash_recorded"])
            self.assertTrue(report["checks"]["benchmark_hash_matches"])
            self.assertTrue(report["checks"]["results_hash_matches"])
            self.assertTrue(report["checks"]["split_version_known"])
            self.assertTrue(report["checks"]["split_respects_holdout"])
            self.assertFalse(report["checks"]["formal_run_uses_heldout_repositories"])
            self.assertFalse(report["checks"]["contextbench_generalization_included"])
            paper = render_report(summary, [], report)
            self.assertIn("## 10. Limitations", paper)

    def test_repository_tools_and_usage_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text(
                "class Cache:\n    def expire(self):\n        return True\n",
                encoding="utf-8",
            )
            (root / "test_cache.py").write_text("from cache import Cache\n", encoding="utf-8")
            tools = RepositoryTools(root)
            self.assertEqual(tools.index["cache.py"].language, "python")
            self.assertEqual(tools.index["cache.py"].symbols["Cache"], [(1, 3)])
            self.assertEqual(tools.index["cache.py"].symbols["expire"], [(2, 3)])
            search = tools.call("SEARCH_TEXT", {"query": "expire"})
            self.assertTrue(search["lines"])
            self.assertNotIn("matches", search)
            symbol = tools.call("SEARCH_SYMBOL", {"symbol": "Cache"})["lines"][0]
            self.assertEqual(symbol["path"], "cache.py")
            self.assertEqual(symbol["end_line"], 3)
            self.assertTrue(symbol["text"])
            self.assertTrue(tools.call("OPEN", {"file": "cache.py", "start_line": 1, "end_line": 2})["lines"])
            self.assertEqual(len(tools.call("FIND_REFERENCES", {"symbol": "Cache"})["lines"]), 2)
            self.assertTrue(tools.call("FIND_TESTS", {"target": "Cache"})["lines"])
            self.assertTrue(tools.logs[-1]["latency_ms"] >= 0)
            self.assertEqual(tools.logs[-1]["lines_returned"], 1)
            tools.call("STOP")
            with self.assertRaises(RuntimeError):
                tools.call("SEARCH_TEXT", {"query": "Cache"})
            self.assertEqual(tools.logs[-1]["error"]["type"], "RuntimeError")

    def test_trajectory_metrics_cover_quality_cost_and_behavior(self):
        trajectory = [
            {"decision": "SEARCH", "action": "SEARCH_TEXT", "arguments": {},
             "context_tokens_returned": 2, "result": {"lines": [
                 {"path": "cache.py", "line": 1, "text": "class Cache:"},
                 {"path": "cache.py", "line": 2, "text": "    def expire(self):"},
             ]}},
            {"decision": "SEARCH", "action": "OPEN", "arguments": {"file": "cache.py"},
             "context_tokens_returned": 2, "result": {"lines": [
                 {"path": "cache.py", "line": 2, "text": "    def expire(self):"},
                 {"path": "cache.py", "line": 3, "text": "        return True"},
             ]}},
        ]
        metrics = evaluate_trajectory(
            trajectory, [Region("cache.py", 1, 3)],
            tool_usage=[{}, {}, {}], stop_reason="sufficient_context",
        )
        self.assertEqual(metrics["span_recall"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["tool_calls"], 3.0)
        self.assertEqual(metrics["search_calls"], 1.0)
        self.assertEqual(metrics["files_opened"], 1.0)
        self.assertEqual(metrics["duplicate_reads"], 1.0)
        self.assertEqual(metrics["context_tokens_read"], 4.0)
        self.assertEqual(metrics["information_gain_per_step"], 1.5)
        self.assertEqual(metrics["stop_reason"], "sufficient_context")

    def test_fixed_budget_agent_trajectory_replays(self):
        class FakeClient:
            def __init__(self):
                self.outputs = iter([
                    '{"action":"SEARCH_TEXT","arguments":{"query":"Cache"}}',
                    '{"action":"SEARCH_SYMBOL","arguments":{"symbol":"Cache"}}',
                ])
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text("class Cache:\n    pass\n", encoding="utf-8")
            client = FakeClient()
            result = FixedBudgetAgent(client).run("Cache problem", root, 2)
            self.assertEqual(client.calls, 2)
            self.assertEqual(result["actions_attempted"], 2)
            replayed = replay_trajectory(root, result["trajectory"])
            self.assertEqual(replayed, [event["result"] for event in result["trajectory"]])

            class StopClient:
                def __init__(self):
                    self.calls = 0

                def complete(self, messages):
                    self.calls += 1
                    return '{"action":"STOP","arguments":{}}'

            stop_client = StopClient()
            with self.assertRaisesRegex(ValueError, "invalid tool decision"):
                FixedBudgetAgent(stop_client).run("Cache problem", root, 3)
            self.assertEqual(stop_client.calls, 1)

    def test_repopilot_tracks_state_and_stops_adaptively(self):
        class FakeClient:
            def __init__(self):
                self.outputs = iter([
                    json.dumps({
                        "decision": "SEARCH", "hypothesis": "Cache expiration path",
                        "known_facts": [], "relevant_spans": [], "unresolved_questions": ["Where is expiration handled?"],
                        "confidence": 0.4, "expected_gain": "Inspect Cache", "action": "OPEN",
                        "arguments": {"file": "cache.py", "start_line": 1, "end_line": 2},
                    }),
                    json.dumps({
                        "decision": "STOP", "hypothesis": "Found candidate", "known_facts": ["Cache is defined here."],
                        "relevant_spans": [{"path": "cache.py", "start": 1, "end": 1}],
                        "unresolved_questions": [], "confidence": 0.9, "expected_gain": "",
                        "action": "", "arguments": {}, "stop_reason": "sufficient_context",
                    }),
                ])
                self.messages = []

            def complete(self, messages):
                self.messages.append(messages)
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text("class Cache:\n    pass\n", encoding="utf-8")
            client = FakeClient()
            result = RepoPilot(client).run("Cache expiration", root, token_budget=20, max_steps=5)
            self.assertEqual(result["stop_reason"], "sufficient_context")
            self.assertEqual(result["budget_used"], 3)
            self.assertEqual(result["trajectory"][0]["context_tokens_returned"], 3)
            self.assertEqual(result["trajectory"][0]["state"]["candidate_files"], ["cache.py"])
            self.assertEqual(result["trajectory"][0]["state"]["inspected_spans"], [
                {"path": "cache.py", "start": 1, "end": 2},
            ])
            self.assertEqual(result["trajectory"][0]["state"]["budget_remaining"], 17)
            self.assertEqual(result["trajectory"][1]["state"]["known_facts"], ["Cache is defined here."])
            self.assertEqual(client.messages[1][-1]["role"], "user")

    def test_adaptive_stop_ablation_forces_a_search_action(self):
        class FakeClient:
            def __init__(self):
                self.outputs = iter([
                    '{"decision":"STOP","stop_reason":"sufficient_context"}',
                    '{"decision":"SEARCH","action":"SEARCH_TEXT","arguments":{"query":"Cache"}}',
                ])

            def complete(self, messages):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text("class Cache:\n    pass\n", encoding="utf-8")
            result = RepoPilot(FakeClient()).run("Cache", root, 10, max_steps=1, ablation="no_adaptive_stop")
            self.assertEqual(result["actions_attempted"], 1)
            self.assertEqual(result["trajectory"][0]["action"], "SEARCH_TEXT")

    def test_repopilot_deduplicates_open_spans_and_caps_context_budget(self):
        class FakeClient:
            def __init__(self):
                self.outputs = iter([
                    json.dumps({"decision": "SEARCH", "action": "OPEN",
                                "arguments": {"file": "cache.py", "start_line": 1, "end_line": 2}}),
                    json.dumps({"decision": "SEARCH", "action": "OPEN",
                                "arguments": {"file": "cache.py", "start_line": 1, "end_line": 3}}),
                    json.dumps({"decision": "STOP", "stop_reason": "sufficient_context"}),
                ])

            def complete(self, messages):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text(
                "class Cache:\n    def expire(self):\n        return True\n", encoding="utf-8",
            )
            result = RepoPilot(FakeClient()).run("Cache expiration", root, token_budget=6, max_steps=5)
            self.assertEqual(result["stop_reason"], "sufficient_context")
            self.assertEqual(result["budget_used"], 5)
            self.assertEqual(result["trajectory"][1]["context_tokens_returned"], 0)
            self.assertEqual(result["trajectory"][1]["state"]["inspected_spans"], [
                {"path": "cache.py", "start": 1, "end": 2},
            ])
            self.assertEqual(result["tool_usage"][1]["arguments"], {
                "file": "cache.py", "start_line": 3, "end_line": 3,
            })

    def test_repopilot_stops_on_budget_exhaustion_and_no_information_gain(self):
        class BudgetClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                return json.dumps({"decision": "SEARCH", "action": "SEARCH_TEXT",
                                   "arguments": {"query": "Cache"}})

        class NoGainClient:
            def __init__(self):
                self.outputs = iter([
                    json.dumps({"decision": "SEARCH", "action": "SEARCH_TEXT",
                                "arguments": {"query": "missing text"}}),
                    json.dumps({"decision": "SEARCH", "action": "SEARCH_SYMBOL",
                                "arguments": {"symbol": "MissingSymbol"}}),
                    json.dumps({"decision": "SEARCH", "action": "FIND_TESTS",
                                "arguments": {"target": "MissingTarget"}}),
                ])

            def complete(self, messages):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache.py").write_text("Cache\n", encoding="utf-8")
            client = BudgetClient()
            exhausted = RepoPilot(client).run("Cache", root, token_budget=1, max_steps=5)
            self.assertEqual(exhausted["stop_reason"], "budget_exhausted")
            self.assertEqual(exhausted["budget_used"], 1)
            self.assertEqual(client.calls, 1)

        with tempfile.TemporaryDirectory() as tmp:
            empty_repo = Path(tmp)
            no_gain = RepoPilot(NoGainClient()).run("Find missing code", empty_repo, 20, max_steps=5)
            self.assertEqual(no_gain["stop_reason"], "no_information_gain")
            self.assertEqual(no_gain["actions_attempted"], 3)
            self.assertTrue(all(not event["result"]["lines"] for event in no_gain["trajectory"]))


if __name__ == "__main__":
    unittest.main()
