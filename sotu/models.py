"""Pinned transformer models with complete, sentence-aware input coverage."""

import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TONE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISIONS = {
    TONE_MODEL: "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    EMOTION_MODEL: "0e1cd914e3d46199ed785853e12b57304e04178b",
    EMBEDDING_MODEL: "e8c3b32edf5434bc2275fc9bab85f82640a19130",
}
SENTENCE_MAX_LENGTH = 256
EMBEDDING_MAX_LENGTH = 384


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_text(text, tokenizer, max_length):
    """Partition text without loss; prefer sentence ends, then token boundaries.

    Offsets preserve the original text, including words unknown to the tokenizer.
    Even a sentence longer than the model limit is split rather than truncated.
    """
    if not text.strip():
        return [text]
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError("Model limit must leave room for text tokens")
    chunks, start = [], 0
    while start < len(text):
        rest = text[start:]
        offsets = tokenizer(rest, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)["offset_mapping"]
        if len(offsets) <= budget:
            chunks.append(rest)
            break
        end = offsets[budget - 1][1]
        boundaries = [m.end() for m in re.finditer(r"[.!?;][\"'’)]*\s+|\n+", rest[:end])]
        if boundaries:
            end = boundaries[-1]
        else:
            whitespace = [m.end() for m in re.finditer(r"\s+", rest[:end])]
            if whitespace:
                end = whitespace[-1]
        # Retokenization of a prefix can differ at its edge.
        while len(tokenizer(rest[:end], add_special_tokens=False)["input_ids"]) > budget:
            end -= 1
        if end <= 0:
            raise ValueError("Unable to split input within the model limit")
        chunks.append(rest[:end])
        start += end
    return chunks


def _expanded(texts, tokenizer, max_length):
    chunks, owners, weights = [], [], []
    for i, text in enumerate(texts):
        parts = split_text(text, tokenizer, max_length)
        chunks.extend(parts)
        owners.extend([i] * len(parts))
        weights.extend(max(len(p.split()), 1) for p in parts)
    return chunks, np.asarray(owners), np.asarray(weights, dtype=np.float32)


def _pool(values, owners, weights, n):
    out = np.zeros((n, values.shape[1]), dtype=np.float32)
    totals = np.zeros(n, dtype=np.float32)
    np.add.at(out, owners, values * weights[:, None])
    np.add.at(totals, owners, weights)
    return out / np.maximum(totals[:, None], 1)


class SentenceClassifier:
    def __init__(self, name: str, max_length: int = SENTENCE_MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=REVISIONS[name])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=REVISIONS[name]).to(device()).eval()
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, texts: list[str], batch_size: int = 64, desc: str = "") -> np.ndarray:
        chunks, owners, weights = _expanded(texts, self.tokenizer, self.max_length)
        order = np.argsort([len(t) for t in chunks])
        out = np.zeros((len(chunks), len(self.labels)), dtype=np.float32)
        for start in tqdm(range(0, len(chunks), batch_size), desc=desc, leave=False):
            idx = order[start:start + batch_size]
            batch = self.tokenizer([chunks[i] for i in idx], padding=True,
                                   truncation=False, return_tensors="pt").to(self.model.device)
            out[idx] = torch.softmax(self.model(**batch).logits, dim=-1).float().cpu().numpy()
        return _pool(out, owners, weights, len(texts))


_cache = {}


def _get(key, factory):
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def tone_probabilities(sentences):
    clf = _get("tone", lambda: SentenceClassifier(TONE_MODEL))
    return clf.labels, clf(sentences, desc="tone")


def tone(sentences: list[str]) -> np.ndarray:
    labels, probs = tone_probabilities(sentences)
    return probs[:, labels.index("positive")] - probs[:, labels.index("negative")]


def emotions(sentences: list[str]) -> tuple[list[str], np.ndarray]:
    clf = _get("emotion", lambda: SentenceClassifier(EMOTION_MODEL))
    return clf.labels, clf(sentences, desc="emotion")


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = _get("embed", lambda: SentenceTransformer(
        EMBEDDING_MODEL, revision=REVISIONS[EMBEDDING_MODEL], device=device()))
    chunks, owners, weights = _expanded(texts, model.tokenizer, EMBEDDING_MAX_LENGTH)
    vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    pooled = _pool(vectors, owners, weights, len(texts))
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
