"""Structure-aware Python parsing (PRD §8).

Source files are split along their *structural* boundaries — modules, classes,
functions, methods — rather than by a fixed token window.  Each emitted chunk
knows its qualified name, its parent, the imports visible to it and the symbols
it references, which is what makes symbol retrieval and change-impact reasoning
possible later on.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..config import IngestionConfig
from ..schema import Chunk

_TEST_MARKERS = ("tests/", "test/", "/test_", "conftest.py")


def is_test_path(rel_path: str) -> bool:
    """Classify a path as test code by pytest's own naming conventions.

    Args:
      rel_path: Repository-relative path.

    Returns:
      True for `test_*.py`, `*_test.py`, `conftest.py`, or any path under a
      `tests/` or `test/` directory.
    """
    name = Path(rel_path).name
    p = "/" + rel_path
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or any(m in p for m in ("/tests/", "/test/"))
    )


def module_name_for(rel_path: str) -> str:
    """Convert a file path to its dotted Python module name.

    Args:
      rel_path: Repository-relative path, e.g. "vllm/core/block.py".

    Returns:
      The dotted module name, e.g. "vllm.core.block". A trailing
      `__init__` is dropped so a package resolves to the package name.
    """
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _ReferenceCollector(ast.NodeVisitor):
    """Collects the names a node refers to.

    Call targets feed the caller edges of the symbol graph; bare names and
    attribute accesses are kept separately because they are far noisier.

    Attributes:
      calls: Names appearing in call position.
      names: Bare names and attribute names appearing anywhere.
    """

    def __init__(self) -> None:
        """Start with both name sets empty."""
        self.calls: set[str] = set()
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Record the call target, then descend into the arguments."""
        target = _callable_name(node.func)
        if target:
            self.calls.add(target)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        """Record a bare name reference."""
        self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Record the attribute name, then descend into the value."""
        self.names.add(node.attr)
        self.generic_visit(node)


def _callable_name(node: ast.AST) -> str | None:
    """Extract the bare name a call expression targets.

    Args:
      node: The `func` of an `ast.Call`.

    Returns:
      The name for `f()` and the attribute for `obj.f()`, or `None` for
      anything more involved, such as a call on a subscript.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _decorator_name(node: ast.AST) -> str:
    """Render a decorator expression as a dotted name.

    Args:
      node: A decorator node, possibly a call such as `@lru_cache(8)`.

    Returns:
      The dotted name with any call arguments stripped, falling back to the
      unparsed source for expressions that have no name form.
    """
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node) if hasattr(ast, "unparse") else "?"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a function definition's signature as source text.

    Args:
      node: The function or coroutine definition.

    Returns:
      A line such as `async def run(self, k: int) -> list[str]`. Arguments
      degrade to "..." and the return annotation is dropped if unparsing
      fails, so an exotic annotation cannot abort ingestion of the file.
    """
    try:
        args = ast.unparse(node.args)
    except Exception:  # pragma: no cover - defensive
        args = "..."
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:  # pragma: no cover
            returns = ""
    return f"{prefix} {node.name}({args}){returns}"


def _node_lines(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """Compute the inclusive line span a node occupies.

    Decorators are folded into the span, so a citation to a decorated
    function points at the `@` line rather than skipping it.

    Args:
      node: Node to measure.
      lines: The file's lines, used only to clamp the end.

    Returns:
      A 1-based, inclusive `(start, end)` pair.
    """
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start) or start
    # include immediately preceding decorators
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, getattr(dec, "lineno", start))
    return start, min(end, len(lines))


def _slice(lines: list[str], start: int, end: int) -> str:
    """Return lines `start`..`end` inclusive, using 1-based indexing."""
    return "\n".join(lines[start - 1 : end])


class PythonCodeParser:
    """Parses one Python file into structural chunks.

    Emits a module overview, one chunk per class (header, docstring,
    attributes and method signatures -- not the method bodies) and one chunk
    per function or method. Because a class chunk holds only its overview, a
    question about a class retrieves the summary rather than a thousand lines
    of implementation, while a question about one method retrieves that
    method.

    Attributes:
      config: Chunking limits: line caps, overlap and minimum chunk size.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Store the ingestion config supplying the chunking limits."""
        self.config = config

    # ------------------------------------------------------------------
    def parse(
        self, *, repository: str, commit: str, rel_path: str, source: str
    ) -> list[Chunk]:
        """Parse one Python file into chunks.

        Args:
          repository: Repository name recorded on every chunk.
          commit: Repository HEAD recorded on every chunk.
          rel_path: Repository-relative path of the file.
          source: File contents.

        Returns:
          Module, class and function chunks in source order. Chunks whose
          body is shorter than `min_chunk_chars` are dropped. A file that
          does not parse falls back to fixed windows rather than being lost.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._fallback(repository, commit, rel_path, source)

        lines = source.splitlines()
        artifact = "test" if is_test_path(rel_path) else "source"
        module = module_name_for(rel_path)
        imports = self._imports(tree)
        chunks: list[Chunk] = []

        def emit(chunk: Chunk) -> None:
            if len(chunk.content.strip()) >= self.config.min_chunk_chars:
                chunks.append(chunk)

        # --- module overview chunk -------------------------------------
        if self.config.index_module_chunks:
            overview = self._module_overview(tree, source, module, rel_path)
            if overview:
                emit(
                    Chunk(
                        repository=repository,
                        commit=commit,
                        path=rel_path,
                        artifact_type=artifact,
                        language="python",
                        start_line=1,
                        end_line=min(len(lines), 1 + overview.count("\n")),
                        content=overview,
                        symbol=module.rsplit(".", 1)[-1] if module else rel_path,
                        qualified_name=module or rel_path,
                        symbol_type="module",
                        imports=imports,
                        docstring=ast.get_docstring(tree),
                    )
                )

        # --- top-level definitions -------------------------------------
        for node in tree.body:
            chunks.extend(
                self._from_node(
                    node,
                    lines=lines,
                    repository=repository,
                    commit=commit,
                    rel_path=rel_path,
                    artifact=artifact,
                    module=module,
                    imports=imports,
                    parent=None,
                )
            )
        return [c for c in chunks if len(c.content.strip()) >= self.config.min_chunk_chars]

    # ------------------------------------------------------------------
    def _from_node(
        self,
        node: ast.AST,
        *,
        lines: list[str],
        repository: str,
        commit: str,
        rel_path: str,
        artifact: str,
        module: str,
        imports: list[str],
        parent: str | None,
    ) -> list[Chunk]:
        """Emit chunks for one top-level or nested definition.

        Classes recurse into their own bodies so that methods become chunks
        in their own right, with the class recorded as their parent.

        Args:
          node: Candidate node; anything that is not a function, coroutine or
            class yields nothing.
          lines: The file's lines, for slicing bodies out.
          repository: Repository name recorded on every chunk.
          commit: Repository HEAD recorded on every chunk.
          rel_path: Repository-relative path of the file.
          artifact: "source" or "test", decided once per file.
          module: Dotted module name, prefixed onto qualified names.
          imports: Module-level imports, copied onto every chunk from the
            file so that a retrieved method still shows what is in scope.
          parent: Enclosing class name, or `None` at module level.

        Returns:
          Chunks for this node and, for a class, all of its children.
        """
        out: list[Chunk] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_lines(node, lines)
            qualified = ".".join(x for x in (module, parent, node.name) if x)
            refs = _ReferenceCollector()
            refs.visit(node)
            body = _slice(lines, start, end)
            symbol_type = "method" if parent else "function"
            for piece_start, piece_end, piece in self._split_long(body, start, end):
                out.append(
                    Chunk(
                        repository=repository,
                        commit=commit,
                        path=rel_path,
                        artifact_type=artifact,
                        language="python",
                        start_line=piece_start,
                        end_line=piece_end,
                        content=piece,
                        symbol=node.name,
                        qualified_name=qualified,
                        symbol_type=symbol_type,
                        parent_symbol=".".join(x for x in (module, parent) if x) or None,
                        imports=imports,
                        references=sorted(refs.calls),
                        decorators=[_decorator_name(d) for d in node.decorator_list],
                        docstring=ast.get_docstring(node),
                        signature=_signature(node),
                    )
                )
        elif isinstance(node, ast.ClassDef):
            start, end = _node_lines(node, lines)
            qualified = ".".join(x for x in (module, parent, node.name) if x)
            overview = self._class_overview(node, lines)
            refs = _ReferenceCollector()
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    refs.visit(child)
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:  # pragma: no cover
                    pass
            out.append(
                Chunk(
                    repository=repository,
                    commit=commit,
                    path=rel_path,
                    artifact_type=artifact,
                    language="python",
                    start_line=start,
                    end_line=min(end, start + overview.count("\n")),
                    content=overview,
                    symbol=node.name,
                    qualified_name=qualified,
                    symbol_type="class",
                    parent_symbol=".".join(x for x in (module, parent) if x) or None,
                    imports=imports,
                    references=sorted(refs.calls | set(bases)),
                    decorators=[_decorator_name(d) for d in node.decorator_list],
                    docstring=ast.get_docstring(node),
                    signature=f"class {node.name}({', '.join(bases)})",
                    extra={"bases": bases, "full_end_line": end},
                )
            )
            child_parent = f"{parent}.{node.name}" if parent else node.name
            for child in node.body:
                out.extend(
                    self._from_node(
                        child,
                        lines=lines,
                        repository=repository,
                        commit=commit,
                        rel_path=rel_path,
                        artifact=artifact,
                        module=module,
                        imports=imports,
                        parent=child_parent,
                    )
                )
        return out

    # ------------------------------------------------------------------
    def _split_long(
        self, body: str, start: int, end: int
    ) -> list[tuple[int, int, str]]:
        """Split an oversized symbol into overlapping line windows.

        Args:
          body: Source text of the symbol.
          start: First line of the symbol, 1-based inclusive.
          end: Last line of the symbol, inclusive.

        Returns:
          A list of `(start_line, end_line, text)` windows of at most
          `max_chunk_lines` lines, overlapping by `chunk_overlap_lines` so
          that a construct straddling a cut survives whole in one window.
          Symbols within the limit are returned unsplit.
        """
        n_lines = end - start + 1
        limit = self.config.max_chunk_lines
        if n_lines <= limit:
            return [(start, end, body)]
        body_lines = body.splitlines()
        overlap = self.config.chunk_overlap_lines
        step = max(limit - overlap, 1)
        pieces: list[tuple[int, int, str]] = []
        for offset in range(0, n_lines, step):
            piece_lines = body_lines[offset : offset + limit]
            if not piece_lines:
                break
            piece_start = start + offset
            piece_end = piece_start + len(piece_lines) - 1
            pieces.append((piece_start, piece_end, "\n".join(piece_lines)))
            if offset + limit >= n_lines:
                break
        return pieces

    def _imports(self, tree: ast.Module) -> list[str]:
        """Collect every module imported anywhere in a file.

        Args:
          tree: Parsed module.

        Returns:
          Sorted, deduplicated dotted names. Relative imports keep their
          leading dots, so "from ..config import X" yields "..config.X".
        """
        out: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                prefix = "." * (node.level or 0) + mod
                out.extend(f"{prefix}.{alias.name}" if prefix else alias.name
                           for alias in node.names)
        return sorted(set(out))

    def _module_overview(
        self, tree: ast.Module, source: str, module: str, rel_path: str
    ) -> str:
        """Build the compact "what is in this file" chunk.

        This is docstring plus imports plus a listing of the public API, and
        it is what lets a question about a file retrieve anything at all --
        no individual symbol chunk describes the file as a whole.

        Args:
          tree: Parsed module.
          source: File contents, unused but kept for signature symmetry with
            the other builders.
          module: Dotted module name.
          rel_path: Repository-relative path, used when there is no module
            name.

        Returns:
          The overview text, or the empty string when it would be too short
          to be worth indexing.
        """
        parts: list[str] = [f"# module {module or rel_path} ({rel_path})"]
        doc = ast.get_docstring(tree)
        if doc:
            parts.append(doc.strip())
        imports = self._imports(tree)
        if imports:
            parts.append("imports: " + ", ".join(imports[:60]))
        api: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [
                    c.name
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                api.append(f"class {node.name}: " + ", ".join(methods[:25]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                api.append(_signature(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        api.append(f"constant {target.id}")
        if api:
            parts.append("defines:\n" + "\n".join(api[:80]))
        text = "\n".join(parts)
        return text if len(text) > self.config.min_chunk_chars else ""

    def _class_overview(self, node: ast.ClassDef, lines: list[str]) -> str:
        """Build a class chunk: header, docstring, attributes, signatures.

        Method bodies are deliberately excluded; they are chunked separately.

        Args:
          node: The class definition.
          lines: The file's lines, for slicing the header and attributes out.

        Returns:
          The overview text, with blank parts dropped.
        """
        start = node.lineno
        first_body = node.body[0]
        header_end = getattr(first_body, "lineno", start) - 1
        header = _slice(lines, start, max(start, header_end))
        parts = [header]
        doc = ast.get_docstring(node)
        if doc:
            parts.append('    """' + doc.strip() + '"""')
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                c_start, c_end = _node_lines(child, lines)
                parts.append(_slice(lines, c_start, c_end))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append("    " + _signature(child) + ": ...")
        return "\n".join(p for p in parts if p.strip())

    def _fallback(
        self, repository: str, commit: str, rel_path: str, source: str
    ) -> list[Chunk]:
        """Chunk an unparseable file into fixed line windows.

        A syntax error is usually a Python 2 file or a template, and losing
        it entirely would be worse than indexing it without structure.

        Args:
          repository: Repository name recorded on every chunk.
          commit: Repository HEAD recorded on every chunk.
          rel_path: Repository-relative path of the file.
          source: File contents.

        Returns:
          Windowed chunks of `symbol_type` "file", carrying no symbol
          structure and therefore no graph edges.
        """
        lines = source.splitlines()
        limit = self.config.max_chunk_lines
        out: list[Chunk] = []
        for offset in range(0, len(lines), limit):
            piece = lines[offset : offset + limit]
            if not piece:
                break
            out.append(
                Chunk(
                    repository=repository,
                    commit=commit,
                    path=rel_path,
                    artifact_type="test" if is_test_path(rel_path) else "source",
                    language="python",
                    start_line=offset + 1,
                    end_line=offset + len(piece),
                    content="\n".join(piece),
                    symbol=Path(rel_path).stem,
                    qualified_name=module_name_for(rel_path),
                    symbol_type="file",
                )
            )
        return out
