"""
Winnowing-based code fingerprinting (MOSS-style).

Reference: "Winnowing: Local Algorithms for Document Fingerprinting"
           Schleimer, Wilkerson, Aiken — SIGMOD 2003

The algorithm:
  1. Normalize code (strip comments, strings, collapse whitespace)
  2. Compute hashes of every consecutive k-gram of the normalized text
  3. Slide a window of width w over the hash sequence, selecting the minimum
     hash in each window as a fingerprint

The resulting fingerprint set is robust to:
  - Renaming variables (whitespace normalization absorbs alignment shifts)
  - Reformatting and comment changes
  - String-literal changes
  - Minor insertions/deletions

Two codebases with high Jaccard similarity of their fingerprint sets are
very likely structural copies.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Set

# ── Tuning constants ──────────────────────────────────────────────────────────

K = 40   # k-gram length (characters, post-normalization)
W = 20   # window width for winnowing

# ── Normalization ─────────────────────────────────────────────────────────────

_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_DOCSTRING_RE     = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)
_LINE_COMMENT_RE  = re.compile(r'(//|#)[^\n]*')
_DQUOTE_STR_RE    = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')
_SQUOTE_STR_RE    = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'")
_WHITESPACE_RE    = re.compile(r'\s+')


def normalize_code(text: str) -> str:
    """
    Strip noise so fingerprints are invariant to cosmetic edits.
    Removes comments, replaces string literals with a placeholder,
    collapses whitespace, and lowercases everything.
    """
    text = _BLOCK_COMMENT_RE.sub(' ', text)
    text = _DOCSTRING_RE.sub(' ', text)
    text = _LINE_COMMENT_RE.sub(' ', text)
    text = _DQUOTE_STR_RE.sub('"S"', text)
    text = _SQUOTE_STR_RE.sub("'S'", text)
    text = _WHITESPACE_RE.sub(' ', text).strip().lower()
    return text


# ── Core algorithm ────────────────────────────────────────────────────────────

def kgram_hashes(text: str, k: int = K) -> List[int]:
    """
    Return hashes of every consecutive k-gram in *text*.
    For short texts (len < k) a single hash of the full text is returned.
    """
    if len(text) < k:
        return [int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)]
    hashes: List[int] = []
    for i in range(len(text) - k + 1):
        h = int(hashlib.sha256(text[i : i + k].encode()).hexdigest(), 16) % (2 ** 32)
        hashes.append(h)
    return hashes


def winnow(hashes: List[int], w: int = W) -> Set[int]:
    """
    Apply the winnowing algorithm: slide a window of width *w* over *hashes*
    and record the minimum hash in each window position as a fingerprint.
    """
    if not hashes:
        return set()
    fps: Set[int] = set()
    for i in range(max(1, len(hashes) - w + 1)):
        fps.add(min(hashes[i : i + w]))
    return fps


def compute_fingerprint(text: str, k: int = K, w: int = W) -> Set[int]:
    """Full pipeline: normalize → k-grams → winnow."""
    return winnow(kgram_hashes(normalize_code(text), k), w)


# ── Similarity metric ─────────────────────────────────────────────────────────

def jaccard(fp1: Set[int], fp2: Set[int]) -> float:
    """Jaccard similarity ∈ [0, 1] between two fingerprint sets."""
    if not fp1 and not fp2:
        return 1.0
    if not fp1 or not fp2:
        return 0.0
    return len(fp1 & fp2) / len(fp1 | fp2)
