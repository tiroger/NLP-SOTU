"""Transcript cleaning and spaCy-based linguistic features."""

import re
from collections import Counter

import spacy
import textstat

# Paragraphs that are not the president's words: crowd and heckler lines,
# editorial notes, and signature blocks at the end of written messages.
DROP_PARAGRAPH = re.compile(
    r"^(Audience members?\s*\.|(Rep|Sen|Representative|Senator)\.? [A-Z][\w'-]+ ?\.|NOTE:)"
)
SPEAKER_TAG = re.compile(r"^The President\s*\.\s*")
ANNOTATION = re.compile(
    r"\[\s*[^\]]{1,400}\]|\(\s*(Applause|Laughter|Inaudible)[^)]{0,30}\)", re.I
)

# Self-references dominate every speech's entity list; leave them out so the
# places a president talks *about* stand out.
SELF_REFERENCES = {"america", "united states", "the united states", "u.s.", "usa", "nation", "states", "union"}

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}


def clean_paragraphs(paragraphs: list[str]) -> list[str]:
    cleaned = []
    for p in paragraphs:
        if DROP_PARAGRAPH.match(p):
            continue
        # All-caps lines of a few words are signatures ("DWIGHT D. EISENHOWER").
        if len(p.split()) <= 6 and p.upper() == p and any(c.isalpha() for c in p):
            continue
        p = ANNOTATION.sub(" ", SPEAKER_TAG.sub("", p))
        p = " ".join(p.split())
        if p:
            cleaned.append(p)
    return cleaned


_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        _nlp.max_length = 2_000_000
    return _nlp


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010).

    Unlike a raw unique-word count, MTLD does not grow with text length, so long
    written messages and short modern speeches can be compared fairly.
    """

    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for tok in seq:
            types.add(tok)
            count += 1
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0
        return len(seq) / factors if factors else float(len(seq))

    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def analyze(paragraphs: list[str]) -> dict:
    """Sentence split, token counts, entities and style metrics for one speech."""
    sentences, para_index = [], []
    words, entities = [], Counter()
    for i, doc in enumerate(nlp().pipe(paragraphs, batch_size=64)):
        for sent in doc.sents:
            text = sent.text.strip()
            if any(c.isalpha() for c in text):
                sentences.append(text)
                para_index.append(i)
        words.extend(t.lower_ for t in doc if t.is_alpha)
        for ent in doc.ents:
            name = re.sub(r"^the\s+", "", ent.text.strip(), flags=re.I)
            if ent.label_ in ("GPE", "LOC") and name.lower() not in SELF_REFERENCES:
                entities[name] += 1

    full_text = "\n".join(paragraphs)
    n_words = len(words)
    pronouns = Counter(w for w in words if w in FIRST_PERSON_SINGULAR | FIRST_PERSON_PLURAL)
    return {
        "sentences": sentences,
        "sentence_paragraph": para_index,
        "tokens": words,
        "entities": entities,
        "stats": {
            "words": n_words,
            "sentences": len(sentences),
            "words_per_sentence": n_words / max(len(sentences), 1),
            "grade_level": textstat.flesch_kincaid_grade(full_text),
            "lexical_diversity": mtld(words),
            "i_per_1k": 1000 * sum(pronouns[w] for w in FIRST_PERSON_SINGULAR) / n_words,
            "we_per_1k": 1000 * sum(pronouns[w] for w in FIRST_PERSON_PLURAL) / n_words,
        },
    }
