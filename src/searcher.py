"""
GitHub repository search.

Strategy (most specific → least specific):

  1. Paired-identifier queries — search for two co-occurring function/class
     names.  These are highly specific and produce few false positives.

  2. Single-identifier queries — fall back to one distinctive name at a time.

  3. Filename queries — last resort; search for uncommon filenames.

For each candidate repository found, a lightweight profile is built from the
repo's top-level file listing (identifiers = stem names of files).  This is
then scored via the similarity engine.

Requires:  GITHUB_TOKEN environment variable
           PyGitHub  (`pip install PyGitHub`)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Set

from . import similarity as sim

try:
    from github import Github, GithubException, RateLimitExceededException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

# Identifiers too generic to be useful search terms
_COMMON_TOKENS: Set[str] = {
    'main', 'init', 'run', 'start', 'stop', 'setup', 'teardown',
    'get', 'set', 'add', 'remove', 'update', 'delete', 'create',
    'read', 'write', 'open', 'close', 'load', 'save', 'parse', 'format',
    'log', 'logger', 'error', 'result', 'data', 'value', 'item', 'items',
    'key', 'name', 'path', 'file', 'dir', 'config', 'options', 'args',
    'kwargs', 'self', 'cls', 'obj', 'test', 'helper', 'util', 'utils',
    'handler', 'manager', 'service', 'client', 'server', 'model',
    'view', 'controller', 'router', 'app', 'api',
    # Builtins / keywords that appear as identifiers
    'true', 'false', 'none', 'null', 'new', 'this', 'super',
    'string', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
}

# Filenames too common to narrow the search
_COMMON_FILENAMES: Set[str] = {
    '__init__.py', 'index.js', 'index.ts', 'main.py', 'main.go',
    'app.py', 'app.js', 'app.ts', 'server.py', 'server.js',
    'utils.py', 'helpers.py', 'config.py', 'settings.py',
    'README.md', 'LICENSE', 'Makefile',
}


# ── Query builders ────────────────────────────────────────────────────────────

def _distinctive_ids(profile: Dict[str, Any], top_n: int = 8) -> List[str]:
    """
    Return the most distinctive (longest, non-generic) identifiers.
    Skips private names, all-uppercase constants, and common vocab.
    """
    ids = [
        i for i in profile.get('identifiers', [])
        if i.lower() not in _COMMON_TOKENS
        and len(i) >= 6
        and not i.startswith('_')
        and not i.isupper()
    ]
    ids.sort(key=lambda x: -len(x))
    return ids[:top_n]


def _build_queries(profile: Dict[str, Any]) -> List[str]:
    """Build ordered list of GitHub code-search queries."""
    queries: List[str] = []
    ids = _distinctive_ids(profile)

    # Paired queries (AND semantics — most precise)
    for i in range(0, min(len(ids) - 1, 6), 2):
        queries.append(f'"{ids[i]}" "{ids[i + 1]}"')

    # Single-identifier queries (broader)
    for ident in ids[:4]:
        queries.append(f'"{ident}"')

    # Filename fallback
    fps = sorted(
        profile.get('fingerprints', []),
        key=lambda x: x.get('size', 0),
        reverse=True,
    )
    for fp in fps[:3]:
        basename = os.path.basename(fp['path'])
        if basename not in _COMMON_FILENAMES:
            queries.append(f'filename:{basename}')

    return queries


# ── Rate-limit guard ──────────────────────────────────────────────────────────

def _wait_for_rate_limit(g: Any) -> None:
    rl = g.get_rate_limit()
    if rl.search.remaining < 3:
        wait = max(0, rl.search.reset.timestamp() - time.time()) + 5
        print(f"  [rate limit] sleeping {wait:.0f}s …")
        time.sleep(wait)


# ── Candidate profile builder ─────────────────────────────────────────────────

def _fetch_candidate_profile(repo: Any) -> Dict[str, Any]:
    """
    Build a lightweight candidate dict from a GitHub repo object.

    We fetch the top-level directory listing and use file stem names as a
    proxy for identifiers — cheap (1 API call) and surprisingly effective.
    """
    candidate: Dict[str, Any] = {
        'sha256_hashes': [],
        'winnow_fp':     [],
        'identifiers':   [],
        'stars':         getattr(repo, 'stargazers_count', 0),
        'primary_language': repo.language or 'unknown',
    }
    try:
        contents = repo.get_contents('')
        candidate['identifiers'] = [
            os.path.splitext(c.name)[0]
            for c in contents
            if c.type == 'file'
        ]
    except Exception:
        pass
    return candidate


# ── Public API ────────────────────────────────────────────────────────────────

def search_repositories(
    profile: Dict[str, Any],
    threshold: float = 0.15,
    max_results_per_query: int = 8,
) -> List[Dict[str, Any]]:
    """
    Search GitHub for repositories that may be copies of the scanned codebase.

    Returns a list of match dicts sorted by composite similarity (descending).
    Only matches at or above *threshold* are included.
    """
    token = os.environ.get('GITHUB_TOKEN')

    if not GITHUB_AVAILABLE or not token:
        return _mock_response()

    g = Github(token)
    seen_repos: Set[str]                   = set()
    candidates: Dict[str, Dict[str, Any]]  = {}

    queries = _build_queries(profile)
    print(f"  Running {len(queries)} search queries …")

    for query in queries:
        _wait_for_rate_limit(g)
        try:
            results = g.search_code(query)
            for item in results[:max_results_per_query]:
                full_name = item.repository.full_name
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)
                if full_name not in candidates:
                    candidates[full_name] = _fetch_candidate_profile(item.repository)

        except RateLimitExceededException:
            _wait_for_rate_limit(g)
            continue
        except GithubException as exc:
            if exc.status == 422:   # invalid query — skip
                continue
            print(f"  GitHub API error for query {query!r}: {exc}")
            continue
        except Exception as exc:
            print(f"  Search error ({query!r}): {exc}")
            continue

    # Score every candidate and filter by threshold
    matches: List[Dict[str, Any]] = []
    for full_name, candidate in candidates.items():
        scores = sim.score(profile, candidate)
        if scores['composite'] >= threshold:
            matches.append({
                'source':   'github',
                'repo':     full_name,
                'repo_url': f'https://github.com/{full_name}',
                'stars':    candidate.get('stars', 0),
                'language': candidate.get('primary_language', 'unknown'),
                **scores,
            })

    matches.sort(key=lambda x: x['composite'], reverse=True)
    return matches


def _mock_response() -> List[Dict[str, Any]]:
    """Returned when GITHUB_TOKEN is not set (demo / offline mode)."""
    return [
        {
            'source':             'mock',
            'repo':               'example/clone-demo',
            'repo_url':           'https://github.com/example/clone-demo',
            'stars':              0,
            'language':           'Python',
            'hash_overlap':       0.0,
            'winnow_jaccard':     0.0,
            'identifier_overlap': 0.0,
            'composite':          0.0,
            'confidence':         'NEGLIGIBLE',
            '_note':              'Set the GITHUB_TOKEN env var to enable real search.',
        }
    ]
