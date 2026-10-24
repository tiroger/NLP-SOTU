"""Run the expensive per-speech NLP (spaCy + transformers) with an on-disk cache.

Usage:
    uv run python -m sotu.process
"""

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sotu import models
from sotu.text import analyze, clean_paragraphs

ROOT = Path(__file__).resolve().parent.parent
SPEECHES_PATH = ROOT / "data" / "speeches.jsonl"
CACHE_DIR = ROOT / ".cache" / "processed"
# Bump to invalidate every cached result after changing cleaning or models.
PIPELINE_VERSION = "1"


def load_speeches() -> list[dict]:
    with SPEECHES_PATH.open() as f:
        return [json.loads(line) for line in f]


def _cache_path(speech: dict) -> Path:
    digest = hashlib.sha1((PIPELINE_VERSION + json.dumps(speech["paragraphs"])).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{speech['id']}-{digest}.pkl"


def bigrams(tokens: list[str]) -> Counter:
    from spacy.lang.en.stop_words import STOP_WORDS

    return Counter(
        f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a not in STOP_WORDS and b not in STOP_WORDS
    )


def process_speech(speech: dict) -> dict:
    paragraphs = clean_paragraphs(speech["paragraphs"])
    a = analyze(paragraphs)
    emotion_labels, emotion_probs = models.emotions(a["sentences"])
    return {
        "id": speech["id"],
        "paragraphs": paragraphs,
        "sentences": a["sentences"],
        "sentence_paragraph": np.array(a["sentence_paragraph"], dtype=np.int32),
        "stats": a["stats"],
        "entities": a["entities"],
        "unigrams": Counter(a["tokens"]),
        "bigrams": bigrams(a["tokens"]),
        "tone": models.tone(a["sentences"]),
        "emotion_labels": emotion_labels,
        "emotions": emotion_probs.astype(np.float16),
        "paragraph_embeddings": models.embed(paragraphs).astype(np.float16),
    }


def load_processed(speeches: list[dict] | None = None) -> list[dict]:
    """Return processed results for every speech, computing any that are missing."""
    speeches = speeches or load_speeches()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for speech in tqdm(speeches, desc="speeches"):
        path = _cache_path(speech)
        if path.exists():
            with path.open("rb") as f:
                results.append(pickle.load(f))
            continue
        for stale in CACHE_DIR.glob(f"{speech['id']}-*.pkl"):
            stale.unlink()
        result = process_speech(speech)
        with path.open("wb") as f:
            pickle.dump(result, f)
        results.append(result)
    return results


if __name__ == "__main__":
    print(f"Processed {len(load_processed())} speeches")
