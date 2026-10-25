"""Run the expensive per-speech NLP (spaCy + transformers) with an on-disk cache.

Usage:
    uv run python -m sotu.process
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sotu import models
from sotu.text import analyze, clean_paragraphs
from sotu.provenance import fingerprint, pipeline_manifest

ROOT = Path(__file__).resolve().parent.parent
SPEECHES_PATH = ROOT / "data" / "speeches.jsonl"
CACHE_DIR = ROOT / ".cache" / "processed"
# Pipeline code, model revisions and dependency versions are hashed automatically.


def load_speeches() -> list[dict]:
    with SPEECHES_PATH.open() as f:
        return [json.loads(line) for line in f]


def _cache_path(speech: dict) -> Path:
    digest = fingerprint({"pipeline": pipeline_manifest(), "paragraphs": speech["paragraphs"]})[:20]
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
    tone_labels, tone_probs = models.tone_probabilities(a["sentences"])
    return {
        "id": speech["id"],
        "paragraphs": paragraphs,
        "sentences": a["sentences"],
        "sentence_paragraph": np.array(a["sentence_paragraph"], dtype=np.int32),
        "stats": a["stats"],
        "entities": a["entities"],
        "unigrams": Counter(a["tokens"]),
        "bigrams": bigrams(a["tokens"]),
        "tone": tone_probs[:, tone_labels.index("positive")] - tone_probs[:, tone_labels.index("negative")],
        "tone_labels": tone_labels,
        "tone_probs": tone_probs,
        "provenance": pipeline_manifest(),
        "emotion_labels": emotion_labels,
        "emotions": emotion_probs.astype(np.float16),
        "paragraph_embeddings": models.embed(paragraphs).astype(np.float16),
    }


def load_processed(speeches: list[dict] | None = None) -> list[dict]:
    """Return processed results for every speech, computing any that are missing."""
    speeches = load_speeches() if speeches is None else speeches
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for speech in tqdm(speeches, desc="speeches"):
        path = _cache_path(speech)
        if path.exists():
            with path.open("rb") as f:
                results.append(pickle.load(f))
            continue
        result = process_speech(speech)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as f:
            pickle.dump(result, f)
        temporary.replace(path)
        results.append(result)
    return results


if __name__ == "__main__":
    print(f"Processed {len(load_processed())} speeches")
