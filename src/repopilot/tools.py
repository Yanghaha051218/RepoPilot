"""Small, shell-free repository exploration tools with usage logs."""

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .benchmark import Region

TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
# ponytail: non-Python symbols get declaration-line spans; add parsers when full ranges matter.
SYMBOL = re.compile(r"\b(?:class|def|function|func|struct|interface|enum|type)\s+([A-Za-z_]\w*)")
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv"}
# ponytail: cap indexed files at 1 MB to bound memory; stream larger files if a task needs them.
MAX_FILE_BYTES = 1_000_000
MAX_RESULTS = 50
MAX_OPEN_LINES = 200


@dataclass(frozen=True)
class FileIndex:
    path: str
    language: str
    line_count: int
    symbols: Dict[str, List[Tuple[int, int]]]
    imports: List[str]


class RepositoryTools:
    """Read-only SEARCH_TEXT, SEARCH_SYMBOL, OPEN, FIND_REFERENCES, FIND_TESTS, STOP."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(str(root))
        self.files: Dict[str, str] = {}
        self.index: Dict[str, FileIndex] = {}
        self.logs: List[dict] = []
        self.stopped = False
        self._index()

    def _index(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink() or SKIP_DIRS.intersection(path.relative_to(self.root).parts):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            rel = path.relative_to(self.root).as_posix()
            lines = text.splitlines()
            symbols: Dict[str, List[Tuple[int, int]]] = {}
            imports = []
            python_file = path.suffix.lower() in (".py", ".pyi")
            if not python_file:
                for number, line in enumerate(lines, 1):
                    for match in SYMBOL.finditer(line):
                        symbols.setdefault(match.group(1), []).append((number, number))
            if python_file:
                try:
                    tree = ast.parse(text)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            symbols.setdefault(node.name, []).append(
                                (node.lineno, getattr(node, "end_lineno", node.lineno))
                            )
                        elif isinstance(node, ast.Import):
                            imports.extend(alias.name for alias in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imports.append(node.module)
                except SyntaxError:
                    for number, line in enumerate(lines, 1):
                        for match in SYMBOL.finditer(line):
                            symbols.setdefault(match.group(1), []).append((number, number))
            self.files[rel] = text
            suffix = path.suffix.lower()
            language = "python" if suffix in (".py", ".pyi") else suffix.lstrip(".") or "unknown"
            self.index[rel] = FileIndex(
                rel, language, len(lines),
                {name: sorted(set(rows)) for name, rows in symbols.items()},
                sorted(set(imports)),
            )

    def _query(self, value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            raise ValueError("{} must be a non-empty string of at most 500 characters".format(name))
        return value.strip()

    def _record(self, action: str, arguments: dict, result: dict, started: float) -> dict:
        lines = result.get("lines", [])
        text = "\n".join(str(line.get("text", "")) for line in lines if isinstance(line, dict))
        entry = {
            "action": action,
            "arguments": arguments,
            "result": result,
            "lines_returned": len(lines),
            "tokens_returned": len(TOKEN.findall(text)),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self.logs.append(entry)
        return result

    def call(self, action: str, arguments: Optional[dict] = None) -> dict:
        started = time.perf_counter()
        log_count = len(self.logs)
        try:
            return self._execute(action, arguments, started)
        except Exception as exc:
            if len(self.logs) == log_count:
                self.logs.append({
                    "action": action,
                    "arguments": {} if arguments is None else arguments,
                    "result": {"error": str(exc), "lines": []},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "lines_returned": 0,
                    "tokens_returned": 0,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                })
            raise

    def _execute(self, action: str, arguments: Optional[dict], started: float) -> dict:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if self.stopped:
            raise RuntimeError("exploration has stopped")
        if action == "SEARCH_TEXT":
            query = self._query(arguments.get("query"), "query")
            matches = []
            for path, text in self.files.items():
                for number, line in enumerate(text.splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append({"path": path, "line": number, "text": line[:500]})
                        if len(matches) == MAX_RESULTS:
                            break
                if len(matches) == MAX_RESULTS:
                    break
            return self._record(action, {"query": query}, {"lines": matches}, started)
        if action == "SEARCH_SYMBOL":
            query = self._query(arguments.get("symbol"), "symbol")
            matches = []
            for path, entry in self.index.items():
                if query in entry.symbols:
                    lines = self.files[path].splitlines()
                    matches.extend({"path": path, "line": start, "end_line": end,
                                    "symbol": query, "text": lines[start - 1]}
                                   for start, end in entry.symbols[query])
            matches = matches[:MAX_RESULTS]
            return self._record(action, {"symbol": query}, {"lines": matches}, started)
        if action == "OPEN":
            rel = self._query(arguments.get("file"), "file").replace("\\", "/")
            start, end = arguments.get("start_line"), arguments.get("end_line")
            candidate = (self.root / rel).resolve()
            if self.root not in candidate.parents or candidate.is_dir():
                raise ValueError("file must be inside the repository")
            if type(start) is not int or type(end) is not int or start < 1 or end < start or end - start + 1 > MAX_OPEN_LINES:
                raise ValueError("line range must be valid and no larger than {} lines".format(MAX_OPEN_LINES))
            if rel not in self.files:
                raise FileNotFoundError(rel)
            lines = self.files[rel].splitlines()
            selected = [{"path": rel, "line": number, "text": lines[number - 1]} for number in range(start, min(end, len(lines)) + 1)]
            return self._record(action, {"file": rel, "start_line": start, "end_line": end}, {"lines": selected}, started)
        if action == "FIND_REFERENCES":
            symbol = self._query(arguments.get("symbol"), "symbol")
            pattern = re.compile(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)")
            matches = []
            for path, text in self.files.items():
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        matches.append({"path": path, "line": number, "text": line[:500]})
                        if len(matches) == MAX_RESULTS:
                            break
                if len(matches) == MAX_RESULTS:
                    break
            return self._record(action, {"symbol": symbol}, {"lines": matches}, started)
        if action == "FIND_TESTS":
            target = self._query(arguments.get("target"), "target")
            name = Path(target).name.lower()
            symbol = Path(target).stem if "/" in target or "." in target else target
            candidates = []
            for path, text in self.files.items():
                lower = path.lower()
                is_test = "test" in lower or lower.endswith("_spec.js") or lower.endswith("_spec.ts")
                if not is_test:
                    continue
                score = int(name in lower) + int(symbol.lower() in lower) + int(symbol.lower() in text.lower())
                candidates.append((score, path, text))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            matches = [{"path": path, "line": 1, "text": text.splitlines()[0][:500] if text.splitlines() else ""}
                       for score, path, text in candidates[:MAX_RESULTS] if score]
            return self._record(action, {"target": target}, {"lines": matches}, started)
        if action == "STOP":
            self.stopped = True
            return self._record(action, {}, {"stop_reason": "requested", "lines": []}, started)
        raise ValueError("unknown action: {}".format(action))
