"""Deterministic static retrieval controls over fixed line chunks."""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .benchmark import Region

TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv"}
MAX_FILE_BYTES = 1_000_000
CHUNK_LINES = 20


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int
    end: int
    text: str

    def region(self) -> Region:
        return Region(self.path, self.start, self.end)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN.findall(text)]


def chunks(repo: Path, chunk_lines: int = CHUNK_LINES) -> List[Chunk]:
    found = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.is_symlink() or SKIP_DIRS.intersection(path.relative_to(repo).parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        lines = text.splitlines()
        rel = path.relative_to(repo).as_posix()
        for offset in range(0, len(lines), chunk_lines):
            found.append(Chunk(rel, offset + 1, min(offset + chunk_lines, len(lines)), "\n".join(lines[offset:offset + chunk_lines])))
    return found


def _truncate(ranked: Sequence[Tuple[float, Chunk]], line_budget: int) -> List[Region]:
    selected, used = [], 0
    for _, chunk in ranked:
        cost = chunk.end - chunk.start + 1
        if used + cost <= line_budget:
            selected.append(chunk.region())
            used += cost
    return selected


def retrieve(method: str, issue: str, corpus: Sequence[Chunk], task_id: str, line_budget: int, seed: int = 17) -> List[Region]:
    query = tokenize(issue)
    docs = [tokenize(chunk.text) for chunk in corpus]
    if method == "random":
        # Stable per task and independent of Python's randomized hash seed.
        ranked = sorted(corpus, key=lambda chunk: hashlib.sha256((str(seed) + "|" + task_id + "|" + chunk.path + ":" + str(chunk.start)).encode()).digest())
        return _truncate([(0.0, chunk) for chunk in ranked], line_budget)
    if method == "grep":
        q = set(query)
        scored = [(sum(1 for token in set(doc) if token in q), chunk) for chunk, doc in zip(corpus, docs)]
    elif method == "bm25":
        df = Counter(token for doc in docs for token in set(doc))
        avgdl = sum(map(len, docs)) / float(max(len(docs), 1))
        qfreq = Counter(query)
        scored = []
        for chunk, doc in zip(corpus, docs):
            tf, length = Counter(doc), len(doc)
            score = 0.0
            for term, qn in qfreq.items():
                freq = tf[term]
                if freq:
                    idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                    score += qn * idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / avgdl))
            scored.append((score, chunk))
    elif method == "embedding":
        scored = _hash_embedding_scores(query, corpus, docs)
    else:
        raise ValueError("unknown retrieval method: {}".format(method))
    ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1].path, pair[1].start))
    return _truncate(ranked, line_budget)


def _hash_embedding_scores(query: Sequence[str], corpus: Sequence[Chunk], docs: Sequence[Sequence[str]]) -> List[Tuple[float, Chunk]]:
    """Deterministic lexical hashing vectors; not a semantic language model."""
    def vector(tokens: Iterable[str]) -> Dict[int, float]:
        values: Dict[int, float] = {}
        for token in tokens:
            features = ["w:" + token]
            padded = "^" + token + "$"
            features.extend("c:" + padded[i:i + 3] for i in range(max(1, len(padded) - 2)))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index, sign = value % 4096, (1.0 if value & (1 << 63) else -1.0)
                values[index] = values.get(index, 0.0) + sign
        norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
        return {index: value / norm for index, value in values.items()}

    q = vector(query)
    result = []
    for chunk, doc in zip(corpus, docs):
        d = vector(doc)
        result.append((sum(value * d.get(index, 0.0) for index, value in q.items()), chunk))
    return result
