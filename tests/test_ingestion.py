"""Ingestion: structure-aware parsing, documents, git history."""

from __future__ import annotations

from eka.config import IngestionConfig
from eka.ingestion.code_parser import PythonCodeParser, is_test_path, module_name_for
from eka.ingestion.document_parser import DocumentParser


def test_symbols_are_chunked_structurally(chunks):
    """Methods become their own chunks, with parent, line range and docstring."""
    by_symbol = {c.qualified_name: c for c in chunks if c.symbol_type in ("class", "method")}
    assert "minirepo.cache_engine.CacheEngine" in by_symbol
    allocate = by_symbol["minirepo.cache_engine.CacheEngine.allocate_gpu_cache"]
    assert allocate.symbol_type == "method"
    assert allocate.parent_symbol == "minirepo.cache_engine.CacheEngine"
    assert allocate.start_line < allocate.end_line
    assert "def allocate_gpu_cache" in allocate.content
    assert allocate.docstring and "Allocate" in allocate.docstring


def test_line_ranges_match_the_file(fixture_repo, chunks):
    """A chunk's recorded line range indexes the same text as the source file."""
    source = (fixture_repo / "minirepo/worker.py").read_text().splitlines()
    for chunk in chunks:
        if chunk.path != "minirepo/worker.py" or chunk.symbol_type != "method":
            continue
        first = source[chunk.start_line - 1]
        assert chunk.content.splitlines()[0] == first
        assert chunk.end_line <= len(source)


def test_imports_and_references_are_recorded(chunks):
    """Call targets and module imports are captured for the symbol graph."""
    worker = [c for c in chunks if c.qualified_name == "minirepo.worker.Worker.initialize_cache"]
    assert worker, "worker method chunk missing"
    assert "allocate_gpu_cache" in worker[0].references
    assert any("minirepo.cache_engine" in imp for imp in worker[0].imports)


def test_tests_are_marked_as_tests(chunks):
    """Test files are classified by pytest naming, and lookalikes are not."""
    test_chunks = [c for c in chunks if c.artifact_type == "test"]
    assert test_chunks
    assert all(c.path.startswith("tests/") for c in test_chunks)
    assert is_test_path("tests/v1/test_x.py")
    assert is_test_path("pkg/test_thing.py")
    assert not is_test_path("pkg/latest_thing.py")


def test_documents_keep_heading_hierarchy(chunks):
    """Document chunks carry the heading path that led to them."""
    docs = [c for c in chunks if c.artifact_type == "doc"]
    assert docs
    assert any(c.heading_path and c.heading_path[-1] == "KV cache" for c in docs)


def test_document_parser_nested_headings():
    """Nested markdown headings produce nested heading paths."""
    parser = DocumentParser(IngestionConfig(doc_min_chars=1))
    source = "# Top\n\nintro text here\n\n## Child\n\nchild body text\n\n### Grand\n\ndeep body\n"
    parsed = parser.parse(
        repository="r", commit="c", rel_path="docs/x.md", source=source
    )
    paths = [tuple(c.heading_path) for c in parsed]
    assert ("Top",) in paths
    assert ("Top", "Child") in paths
    assert ("Top", "Child", "Grand") in paths


def test_commits_are_ingested(chunks):
    """Commits become chunks carrying SHA, touched files and message."""
    commits = [c for c in chunks if c.artifact_type == "commit"]
    assert commits
    commit = commits[0]
    assert commit.commit_sha and len(commit.commit_sha) == 40
    assert commit.files_changed
    assert "paged KV cache" in commit.content


def test_oversized_symbols_are_split():
    """A symbol over the line cap splits into windows that cover it without gaps."""
    config = IngestionConfig(max_chunk_lines=10, chunk_overlap_lines=2, min_chunk_chars=1)
    body = "\n".join(f"    x = {i}" for i in range(40))
    source = f"def big():\n{body}\n"
    parsed = PythonCodeParser(config).parse(
        repository="r", commit="c", rel_path="m.py", source=source
    )
    pieces = [c for c in parsed if c.symbol == "big"]
    assert len(pieces) > 1
    assert pieces[0].end_line - pieces[0].start_line + 1 <= 10
    # the windows must cover the symbol without gaps
    assert pieces[1].start_line <= pieces[0].end_line + 1


def test_unparseable_file_still_indexed():
    """A syntax error falls back to windowed chunks rather than losing the file."""
    config = IngestionConfig(min_chunk_chars=1)
    parsed = PythonCodeParser(config).parse(
        repository="r", commit="c", rel_path="broken.py",
        source="def f(:\n  this is not python\n" * 5,
    )
    assert parsed and parsed[0].symbol_type == "file"


def test_module_name_for():
    """Paths convert to dotted module names, dropping a trailing __init__."""
    assert module_name_for("vllm/v1/core/sched/scheduler.py") == "vllm.v1.core.sched.scheduler"
    assert module_name_for("vllm/v1/__init__.py") == "vllm.v1"


def test_chunk_ids_are_stable(config, chunks):
    """Re-ingesting an unchanged checkout reproduces the same chunk ids."""
    from eka.ingestion.scanner import RepositoryScanner

    again, _ = RepositoryScanner(config).scan(config.repo_dir, include_git=False)
    ids = {c.chunk_id for c in chunks}
    assert {c.chunk_id for c in again} <= ids
