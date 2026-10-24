"""Transformer models: sentence tone, sentence emotion, paragraph embeddings."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 3-class (negative / neutral / positive) RoBERTa. Having a neutral class matters
# here: most State of the Union sentences are neither praise nor complaint.
TONE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Ekman's six basic emotions plus neutral.
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceClassifier:
    def __init__(self, name: str, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device()).eval()
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, texts: list[str], batch_size: int = 64, desc: str = "") -> np.ndarray:
        """Return an (n_texts, n_labels) array of class probabilities."""
        # Sorting by length keeps padding (and wasted compute) to a minimum.
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
            idx = order[start:start + batch_size]
            batch = self.tokenizer(
                [texts[i] for i in idx], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.model.device)
            out[idx] = torch.softmax(self.model(**batch).logits, dim=-1).float().cpu().numpy()
        return out


_cache = {}


def _get(key, factory):
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def tone(sentences: list[str]) -> np.ndarray:
    """Per-sentence tone in [-1, 1]: P(positive) - P(negative)."""
    clf = _get("tone", lambda: SentenceClassifier(TONE_MODEL))
    probs = clf(sentences, desc="tone")
    return probs[:, clf.labels.index("positive")] - probs[:, clf.labels.index("negative")]


def emotions(sentences: list[str]) -> tuple[list[str], np.ndarray]:
    clf = _get("emotion", lambda: SentenceClassifier(EMOTION_MODEL))
    return clf.labels, clf(sentences, desc="emotion")


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = _get("embed", lambda: SentenceTransformer(EMBEDDING_MODEL, device=device()))
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
