"""Explicit, serializable exploration state."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RepoPilotState:
    issue: str
    token_budget: int
    hypothesis: str = ""
    candidate_files: List[str] = field(default_factory=list)
    inspected_spans: List[dict] = field(default_factory=list)
    relevant_spans: List[dict] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    previous_actions: List[dict] = field(default_factory=list)
    budget_used: int = 0
    confidence: float = 0.0

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.budget_used)

    def snapshot(self) -> dict:
        return {
            "issue": self.issue,
            "hypothesis": self.hypothesis,
            "candidate_files": list(self.candidate_files),
            "inspected_spans": list(self.inspected_spans),
            "relevant_spans": list(self.relevant_spans),
            "known_facts": list(self.known_facts),
            "unresolved_questions": list(self.unresolved_questions),
            "previous_actions": list(self.previous_actions),
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "confidence": self.confidence,
        }

    def update(self, decision: Dict, event: Dict, tokens_returned: int) -> None:
        self.hypothesis = str(decision.get("hypothesis", self.hypothesis))[:2000]
        self.known_facts = _strings(decision.get("known_facts", self.known_facts))
        self.relevant_spans = _regions(decision.get("relevant_spans", self.relevant_spans))
        self.unresolved_questions = _strings(decision.get("unresolved_questions", self.unresolved_questions))
        try:
            self.confidence = min(1.0, max(0.0, float(decision.get("confidence", self.confidence))))
        except (TypeError, ValueError):
            self.confidence = 0.0
        result = event.get("result") or {}
        rows = result.get("lines") or []
        for row in rows:
            path = row.get("path") if isinstance(row, dict) else None
            if path and path not in self.candidate_files:
                self.candidate_files.append(path)
        if event.get("action") == "OPEN" and "error" not in event:
            opened = result.get("lines") or []
            by_path = {}
            for line in opened:
                if isinstance(line, dict) and isinstance(line.get("path"), str) and type(line.get("line")) is int:
                    by_path.setdefault(line["path"], []).append(line["line"])
            for path, lines in by_path.items():
                for start, end in _ranges(lines):
                    self.inspected_spans.append({"path": path, "start": start, "end": end})
        self.budget_used += tokens_returned
        self.previous_actions.append(dict(event))


def _strings(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:1000] for item in value[:30] if isinstance(item, str)]


def _regions(value) -> List[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:100]:
        if (isinstance(item, dict) and isinstance(item.get("path"), str)
                and type(item.get("start")) is int and type(item.get("end")) is int
                and item["start"] > 0 and item["end"] >= item["start"]):
            result.append({"path": item["path"][:1000], "start": item["start"], "end": item["end"]})
    return result


def _ranges(lines):
    ordered = sorted(set(lines))
    if not ordered:
        return []
    ranges, start, end = [], ordered[0], ordered[0]
    for line in ordered[1:]:
        if line == end + 1:
            end = line
        else:
            ranges.append((start, end))
            start = end = line
    ranges.append((start, end))
    return ranges
