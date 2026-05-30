"""
Directory scanner.

For each non-binary source file produces a record containing:
  path         – relative path from the scanned root
  size         – file size in bytes
  language     – file extension without leading dot (e.g. "py", "js")
  hashes       – {md5, sha1, sha256, sha512}
  identifiers  – function/class/import names found in the file
  winnow_fp    – list[int] winnowing fingerprint of the file's source text
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from .fingerprinter import compute_fingerprint

# ── Ignore rules ──────────────────────────────────────────────────────────────

_IGNORE_DIRS: Set[str] = {
    '.git', '.hg', '.svn',
    '__pycache__', '.mypy_cache', '.pytest_cache', '.tox',
    'node_modules', 'bower_components',
    '.venv', 'venv', 'env', '.env',
    'dist', 'build', 'target', 'out', 'bin', 'obj',
    '.idea', '.vscode',
    'vendor',
    '.eggs',
}

_IGNORE_EXTENSIONS: Set[str] = {
    # Compiled / binary
    '.pyc', '.pyo', '.pyd', '.class', '.o', '.obj', '.so', '.dll',
    '.exe', '.bin', '.wasm', '.a', '.lib',
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.jar', '.war',
    # Media
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac',
    # Documents
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Lock files
    '.lock',
    # Fonts
    '.ttf', '.woff', '.woff2', '.eot',
}

_IGNORE_FILENAMES: Set[str] = {
    'package-lock.json', 'yarn.lock', 'Cargo.lock', 'poetry.lock',
    'Pipfile.lock', 'composer.lock', 'Gemfile.lock',
    '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes',
}

# Extensions we attempt identifier / fingerprint extraction on
_CODE_EXTENSIONS: Set[str] = {
    '.py', '.pyw',
    '.js', '.mjs', '.cjs', '.jsx',
    '.ts', '.tsx',
    '.go',
    '.java', '.kt', '.scala', '.groovy',
    '.rs',
    '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hxx',
    '.cs',
    '.rb',
    '.php',
    '.swift',
    '.r', '.R',
    '.m',                   # Objective-C / MATLAB
    '.lua',
    '.ex', '.exs',          # Elixir
    '.erl',                 # Erlang
    '.hs',                  # Haskell
    '.ml', '.mli',          # OCaml
    '.sh', '.bash', '.zsh',
}

# ── Hashing ───────────────────────────────────────────────────────────────────

def _hash_file(path: str) -> Dict[str, str]:
    algos = {
        'md5':    hashlib.md5(),
        'sha1':   hashlib.sha1(),
        'sha256': hashlib.sha256(),
        'sha512': hashlib.sha512(),
    }
    try:
        with open(path, 'rb') as fh:
            while chunk := fh.read(65536):
                for h in algos.values():
                    h.update(chunk)
        return {k: v.hexdigest() for k, v in algos.items()}
    except OSError:
        return {}


def _is_binary(path: str, sample: int = 8192) -> bool:
    """Heuristic: treat files containing null bytes as binary."""
    try:
        with open(path, 'rb') as fh:
            return b'\x00' in fh.read(sample)
    except OSError:
        return True

# ── Identifier extraction ─────────────────────────────────────────────────────

def _extract_python_identifiers(source: str) -> Set[str]:
    """
    Use the Python AST for accurate extraction.
    Returns function names, class names, and top-level import names.
    Skips private/dunder names (start with _).
    """
    ids: Set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_generic_identifiers(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            if not name.startswith('_'):
                ids.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                ids.add((alias.asname or alias.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                ids.add(node.module.split('.')[0])
    return ids


# Regex for C-family, Go, Rust, Java, JS/TS, etc.
_DECL_RE   = re.compile(
    r'\b(?:function|func|def|class|struct|interface|enum|type|fn|sub|method'
    r'|procedure|record|module|namespace)\s+([A-Za-z_][A-Za-z0-9_]*)'
)
_IMPORT_RE = re.compile(
    r'\b(?:import|require|include|use|from|using|extern\s+crate)\s+'
    r'[\'"]?([A-Za-z_][A-Za-z0-9_./-]*)[\'"]?'
)


def _extract_generic_identifiers(source: str) -> Set[str]:
    ids: Set[str] = set()
    for m in _DECL_RE.finditer(source):
        name = m.group(1)
        if not name.startswith('_'):
            ids.add(name)
    for m in _IMPORT_RE.finditer(source):
        # Take only the root package/module name
        ids.add(m.group(1).split('.')[0].split('/')[0])
    return ids


def extract_identifiers(source: str, ext: str) -> Set[str]:
    if ext in {'.py', '.pyw'}:
        return _extract_python_identifiers(source)
    return _extract_generic_identifiers(source)

# ── Public API ────────────────────────────────────────────────────────────────

def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Walk *directory* recursively and return one record per processable file.

    Binary files, ignored directories, lock files, and compiled artifacts are
    skipped automatically.
    """
    base = Path(directory).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results: List[Dict[str, Any]] = []

    for root, dirs, files in os.walk(base):
        # Prune ignored directories in-place (prevents os.walk descending into them)
        dirs[:] = [
            d for d in dirs
            if d not in _IGNORE_DIRS and not d.endswith('.egg-info')
        ]

        for fname in files:
            if fname in _IGNORE_FILENAMES:
                continue
            ext = Path(fname).suffix.lower()
            if ext in _IGNORE_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            if _is_binary(fpath):
                continue

            hashes = _hash_file(fpath)
            if not hashes:
                continue

            rel       = os.path.relpath(fpath, base)
            size      = os.path.getsize(fpath)
            lang      = ext.lstrip('.') or 'unknown'
            identifiers: List[str] = []
            winnow_fp:   List[int]  = []

            if ext in _CODE_EXTENSIONS:
                try:
                    source = Path(fpath).read_text(errors='ignore')
                    identifiers = sorted(extract_identifiers(source, ext))
                    winnow_fp   = sorted(compute_fingerprint(source))
                except OSError:
                    pass

            results.append({
                'path':        rel,
                'size':        size,
                'language':    lang,
                'hashes':      hashes,
                'identifiers': identifiers,
                'winnow_fp':   winnow_fp,
            })

    return results
