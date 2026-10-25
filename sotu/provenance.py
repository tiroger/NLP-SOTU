"""Content-addressed analysis provenance; no credentials or machine paths."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_manifest():
    from sotu.models import REVISIONS, SENTENCE_MAX_LENGTH, EMBEDDING_MAX_LENGTH

    packages = ["numpy", "pandas", "scikit-learn", "spacy", "en-core-web-sm", "textstat",
                "torch", "transformers", "sentence-transformers", "umap-learn"]
    return {
        "schema": 2, "python": platform.python_version(), "models": REVISIONS,
        "sentence_max_length": SENTENCE_MAX_LENGTH, "embedding_max_length": EMBEDDING_MAX_LENGTH,
        "pooling": "word-weighted complete chunks, embedding renormalized",
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "source_sha256": {f: file_hash(ROOT / "sotu" / f) for f in ["models.py", "text.py", "process.py"]},
        "uv_lock_sha256": file_hash(ROOT / "uv.lock"),
    }
