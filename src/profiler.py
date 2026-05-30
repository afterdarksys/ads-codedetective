"""
Profile generator.

Aggregates per-file scanner records into a project-level .cd profile:

  version             – schema version string
  total_files         – int
  total_size_bytes    – int
  languages           – {ext: file_count}
  identifiers         – sorted union of all per-file identifier names
  winnow_fingerprint  – sorted union of all per-file winnowing fingerprint ints
  ml_vector           – TF-IDF feature vector (requires scikit-learn)
  neural_fingerprint  – autoencoder embedding  (requires torch)
  fingerprints        – raw per-file records from scanner
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Set

PROFILE_VERSION = "2.0"
FIXED_INPUT_SIZE = 100      # must match model input dim

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from .model import CodeNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Extensions we try to read source text from for TF-IDF
_ML_EXTENSIONS = {
    'py', 'js', 'ts', 'jsx', 'tsx', 'go', 'java', 'rs',
    'c', 'cpp', 'h', 'hpp', 'cs', 'rb', 'php', 'swift',
}


def generate_profile(fingerprints: List[Dict[str, Any]], directory: str) -> Dict[str, Any]:
    """
    Build a rich project profile from *fingerprints* (the output of
    ``scanner.scan_directory``).

    *directory* is only needed to re-read source files for TF-IDF; all
    structural data (identifiers, winnow_fp) is taken directly from the
    fingerprint records.
    """
    profile: Dict[str, Any] = {
        'version':           PROFILE_VERSION,
        'total_files':       len(fingerprints),
        'total_size_bytes':  sum(f.get('size', 0) for f in fingerprints),
        'languages':         {},
        'identifiers':       [],
        'winnow_fingerprint': [],
        'fingerprints':      fingerprints,
    }

    all_identifiers: Set[str] = set()
    combined_winnow: Set[int] = set()
    code_texts: List[str]     = []

    for item in fingerprints:
        lang = item.get('language', 'unknown')
        profile['languages'][lang] = profile['languages'].get(lang, 0) + 1

        all_identifiers.update(item.get('identifiers', []))
        combined_winnow.update(item.get('winnow_fp', []))

        if lang in _ML_EXTENSIONS:
            full_path = os.path.join(directory, item['path'])
            try:
                with open(full_path, 'r', errors='ignore') as fh:
                    code_texts.append(fh.read())
            except OSError:
                pass

    profile['identifiers']       = sorted(all_identifiers)
    profile['winnow_fingerprint'] = sorted(combined_winnow)

    # ── TF-IDF feature vector ─────────────────────────────────────────────────
    project_vector: List[float] = [0.0] * FIXED_INPUT_SIZE

    if SKLEARN_AVAILABLE and code_texts:
        try:
            vec = TfidfVectorizer(
                max_features=FIXED_INPUT_SIZE,
                token_pattern=r'[A-Za-z_][A-Za-z0-9_]{2,}',   # code tokens ≥ 3 chars
                sublinear_tf=True,      # log(1+tf) — compresses very frequent tokens
                stop_words='english',
            )
            X = vec.fit_transform(code_texts)
            raw: List[float] = X.sum(axis=0).tolist()[0]
            if len(raw) < FIXED_INPUT_SIZE:
                raw += [0.0] * (FIXED_INPUT_SIZE - len(raw))
            project_vector = raw[:FIXED_INPUT_SIZE]
            profile['ml_vector'] = {
                'vocabulary': list(vec.get_feature_names_out()),
                'weights':    project_vector,
            }
        except Exception as exc:
            profile['ml_error'] = str(exc)

    # ── Neural embedding ──────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        try:
            model = CodeNet(input_size=FIXED_INPUT_SIZE)
            model.eval()
            inp = torch.tensor([project_vector], dtype=torch.float32)
            with torch.no_grad():
                embedding = model.encode(inp)
            profile['neural_fingerprint'] = embedding.numpy().tolist()[0]
        except Exception as exc:
            profile['neural_error'] = str(exc)

    return profile
