"""Line-level retrieval metrics for ranked regions."""

import math
from typing import Dict, Sequence, Set, Tuple

from .benchmark import Region


def evaluate(predicted: Sequence[Region], gold: Sequence[Region], line_budget: int) -> Dict[str, float]:
    pred_lines: Set[Tuple[str, int]] = set()
    gold_lines: Set[Tuple[str, int]] = set()
    for region in predicted:
        pred_lines.update((region.path, line) for line in range(region.start, region.end + 1))
    for region in gold:
        gold_lines.update((region.path, line) for line in range(region.start, region.end + 1))
    overlap = pred_lines & gold_lines
    precision = len(overlap) / float(len(pred_lines) or 1)
    recall = len(overlap) / float(len(gold_lines) or 1)
    f1 = 2 * precision * recall / (precision + recall or 1)
    gold_files = {path for path, _ in gold_lines}
    pred_files = {path for path, _ in pred_lines}
    hit_rank = next((rank for rank, region in enumerate(predicted, 1) if any((region.path, line) in gold_lines for line in range(region.start, region.end + 1))), 0)
    gains = [sum((region.path, line) in gold_lines for line in range(region.start, region.end + 1)) for region in predicted]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    region_hits = sum(
        any(pred.path == target.path and pred.start <= target.end and target.start <= pred.end
            for pred in predicted)
        for target in gold
    )
    return {
        "file_recall": len(pred_files & gold_files) / float(len(gold_files) or 1),
        "file_precision": len(pred_files & gold_files) / float(len(pred_files) or 1),
        "span_recall": recall,
        "region_recall": region_hits / float(len(gold) or 1),
        "precision": precision,
        "f1": f1,
        "mrr": 1.0 / hit_rank if hit_rank else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "lines_read": float(len(pred_lines)),
        "line_budget": float(line_budget),
        "waste_ratio": 1.0 - precision,
        "context_efficiency": recall / float(len(pred_lines) or 1),
        "noise_file_rate": len(pred_files - gold_files) / float(len(pred_files) or 1),
    }


def trajectory_regions(trajectory):
    """Convert unique returned source lines into ordered, merged regions."""
    seen, order = set(), []
    for event in trajectory:
        for row in ((event.get("result") or {}).get("lines") or []):
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("line"), int):
                continue
            key = row["path"], row["line"]
            if key not in seen:
                seen.add(key)
                order.append(key)
    # Preserve first-hit order while joining only adjacent lines from the same file.
    regions = []
    for path, line in order:
        if regions and regions[-1].path == path and line == regions[-1].end + 1:
            prev = regions[-1]
            regions[-1] = Region(path, prev.start, line)
        else:
            regions.append(Region(path, line, line))
    return regions


def evaluate_trajectory(trajectory, gold, tool_usage=None, stop_reason=None) -> Dict[str, object]:
    predicted = trajectory_regions(trajectory)
    metrics = evaluate(predicted, gold, sum(region.end - region.start + 1 for region in predicted))
    retrieved, unique = [], set()
    for event in trajectory:
        for row in ((event.get("result") or {}).get("lines") or []):
            key = (row.get("path"), row.get("line")) if isinstance(row, dict) else None
            if key:
                retrieved.append(key)
                unique.add(key)
    actions = [event for event in trajectory if event.get("decision") == "SEARCH" or event.get("action")]
    overlap = unique & {(gold_region.path, line) for gold_region in gold
                        for line in range(gold_region.start, gold_region.end + 1)}
    metrics.update({
        "tool_calls": float(len(tool_usage or [])),
        "exploration_depth": float(len(actions)),
        "context_tokens_read": float(sum(event.get("context_tokens_returned", 0) for event in trajectory)),
        "files_opened": float(len({event.get("arguments", {}).get("file") for event in actions
                                   if event.get("action") == "OPEN"})),
        "search_calls": float(sum(event.get("action") in ("SEARCH_TEXT", "SEARCH_SYMBOL", "FIND_REFERENCES", "FIND_TESTS")
                                  for event in actions)),
        "duplicate_reads": float(len(retrieved) - len(unique)),
        "information_gain_per_step": len(overlap) / float(len(actions) or 1),
        "relevant_context_per_step": len(overlap) / float(len(actions) or 1),
        "stop_reason": stop_reason or (trajectory[-1].get("reason") if trajectory else None),
    })
    return metrics
