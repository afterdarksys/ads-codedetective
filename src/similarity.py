"""
Multi-signal similarity scoring engine.

Three independent signals are combined into a weighted composite score:

  hash_overlap     (weight 0.40) — fraction of our SHA-256 file hashes that
                                   appear in the candidate.  An exact match
                                   proves byte-for-byte copying.

  winnow_jaccard   (weight 0.35) — Jaccard similarity of the project-level
                                   winnowing fingerprints.  Robust to variable
                                   renaming, reformatting, and minor edits.

  identifier_overlap (weight 0.25) — Jaccard of function/class/import names.
                                     Catches structural clones even when code
                                     has been lightly refactored.

Confidence labels:
  CRITICAL  ≥ 0.80   (extremely strong evidence of copying)
  HIGH      ≥ 0.60
  MEDIUM    ≥ 0.40
  LOW       ≥ 0.20
  NEGLIGIBLE < 0.20
"""
from __future__ import annotations

from typing import Any, Dict, Set

WEIGHT_HASH    = 0.40
WEIGHT_WINNOW  = 0.35
WEIGHT_IDS     = 0.25

_CONFIDENCE_THRESHOLDS = [
    (0.80, 'CRITICAL'),
    (0.60, 'HIGH'),
    (0.40, 'MEDIUM'),
    (0.20, 'LOW'),
]


def score(our: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare *our* .cd profile against a *candidate* dict.

    The candidate must contain:
      sha256_hashes  – list[str]   SHA-256 hex digests of its files
      winnow_fp      – list[int]   winnowing fingerprint integers
      identifiers    – list[str]   function/class/import names

    Returns a dict:
      hash_overlap        float
      winnow_jaccard      float
      identifier_overlap  float
      composite           float
      confidence          str
    """
    # 1. Hash overlap ──────────────────────────────────────────────────────────
    our_hashes: Set[str] = {
        fp['hashes']['sha256']
        for fp in our.get('fingerprints', [])
        if fp.get('hashes', {}).get('sha256')
    }
    cand_hashes: Set[str] = set(candidate.get('sha256_hashes', []))
    hash_score = (
        len(our_hashes & cand_hashes) / len(our_hashes)
        if our_hashes else 0.0
    )

    # 2. Winnowing Jaccard ─────────────────────────────────────────────────────
    our_fp: Set[int]  = set(our.get('winnow_fingerprint', []))
    cand_fp: Set[int] = set(candidate.get('winnow_fp', []))
    union_fp = our_fp | cand_fp
    winnow_score = len(our_fp & cand_fp) / len(union_fp) if union_fp else 0.0

    # 3. Identifier Jaccard ────────────────────────────────────────────────────
    our_ids: Set[str]  = set(our.get('identifiers', []))
    cand_ids: Set[str] = set(candidate.get('identifiers', []))
    union_ids = our_ids | cand_ids
    id_score = len(our_ids & cand_ids) / len(union_ids) if union_ids else 0.0

    # Composite ────────────────────────────────────────────────────────────────
    composite = (
        WEIGHT_HASH   * hash_score
        + WEIGHT_WINNOW * winnow_score
        + WEIGHT_IDS    * id_score
    )

    return {
        'hash_overlap':       round(hash_score,   4),
        'winnow_jaccard':     round(winnow_score,  4),
        'identifier_overlap': round(id_score,      4),
        'composite':          round(composite,     4),
        'confidence':         _label(composite),
    }


def _label(s: float) -> str:
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if s >= threshold:
            return label
    return 'NEGLIGIBLE'
