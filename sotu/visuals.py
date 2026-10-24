"""Data preparation for the second set of visuals (map, anatomy, arcs, new words,
sentence lengths, president profiles). Each function returns tidy rows."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

from sotu.places import country_code, country_name

# Topic groups for the speech-anatomy strips: 19 topics are too many colors to
# tell apart, so related ones share a color. "Other" is ceremony and unclassified.
TOPIC_GROUPS = {
    "Foreign & defense": ["Foreign affairs", "Defense & war"],
    "Economy & money": ["Economy & jobs", "Taxes & budget", "Trade & tariffs", "Banking & currency"],
    "Social policy": ["Health care", "Education", "Social welfare", "Immigration", "Civil rights", "Crime & justice"],
    "Government": ["Government & administration"],
    "Land & development": ["Energy & environment", "Agriculture", "Infrastructure", "Science & technology"],
    "Expansion & tribes": ["Native American affairs", "Territories & expansion"],
    "Ceremony & other": ["Other"],
}
GROUP_OF = {t: g for g, ts in TOPIC_GROUPS.items() for t in ts}

# Generic words that pass the statistical filter in new_words() without marking
# anything historical. Editorial, kept short; "tonight" and "kids" stay because
# they do mark eras (prime-time TV addresses, a conversational register).
GENERIC_WORDS = {"enabled", "method", "methods", "tasks", "objectives", "suggestions", "authorizations", "luck",
                 "fashion", "big", "sector", "initiatives", "commitment", "warrants", "got", "tile", "estimated", "sum"}
COMMON_DOC_SHARE = 0.35  # unigrams found in more than this share of all messages are everyday vocabulary

ERAS = [(1790, 1860, "1790–1860"), (1861, 1912, "1861–1912"), (1913, 1964, "1913–1964"), (1965, 2030, "1965–today")]

PROFILE_METRICS = {
    "words": "Length",
    "grade_level": "Reading level",
    "lexical_diversity": "Vocabulary richness",
    "tone": "Net tone",
    "we_per_1k": "“We” rate",
    "i_per_1k": "“I” rate",
}


def places_by_decade(df: pd.DataFrame, processed: list[dict]) -> pd.DataFrame:
    """Mentions of each foreign country per decade, as a share of that decade's country mentions."""
    counts = defaultdict(Counter)
    for year, p in zip(df.year, processed):
        for name, n in p["entities"].items():
            code = country_code(name)
            if code:
                counts[year // 10 * 10][code] += n
    rows = []
    for decade, c in sorted(counts.items()):
        total = sum(c.values())
        rows += [{"decade": decade, "id": code, "country": country_name(code), "mentions": n, "share": n / total}
                 for code, n in c.items()]
    return pd.DataFrame(rows)


def anatomy(df: pd.DataFrame, paragraph_df: pd.DataFrame) -> pd.DataFrame:
    """Each speech as runs of consecutive paragraphs in the same topic group, positioned 0-1 by words."""
    rows = []
    row_of = {sid: i for i, sid in enumerate(df.id)}
    for sid, g in paragraph_df.groupby("speech_id", sort=False):
        words = g.text.str.split().str.len().to_numpy(float)
        edges = np.concatenate([[0], np.cumsum(words)]) / words.sum()
        groups = [GROUP_OF[t] for t in g.topic]
        start = 0
        for i in range(1, len(groups) + 1):
            if i == len(groups) or groups[i] != groups[start]:
                rows.append({"row": row_of[sid], "id": sid, "s": round(edges[start], 4), "e": round(edges[i], 4), "group": groups[start]})
                start = i
    return pd.DataFrame(rows)


def arcs_by_era(df: pd.DataFrame, arcs: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for lo, hi, label in ERAS:
        ids = df.id[df.year.between(lo, hi)]
        mean = np.mean([arcs[i] for i in ids], axis=0)
        rows += [{"era": label, "position": k / (len(mean) - 1), "tone": float(v)} for k, v in enumerate(mean)]
    return pd.DataFrame(rows)


def new_words(df: pd.DataFrame, processed: list[dict], per_decade: int = 5) -> pd.DataFrame:
    """Terms that first become common in each decade.

    A term "breaks out" in decade d when it is used at least 15 times per
    million words there, in at least two messages, and its rate in every
    earlier decade was under a tenth of that. Everyday words (in more than 35%
    of all messages) and GENERIC_WORDS are skipped.
    """
    docfreq = Counter()
    for p in processed:
        docfreq.update(p["unigrams"].keys())
    common = {w for w, n in docfreq.items() if n / len(processed) > COMMON_DOC_SHARE} | GENERIC_WORDS
    decades = df.year // 10 * 10
    words_by_decade = df.groupby(decades).words.sum()
    counts, speeches = defaultdict(Counter), defaultdict(Counter)
    for d, p in zip(decades, processed):
        for c in (p["unigrams"], p["bigrams"]):
            for w, n in c.items():
                counts[w][d] += n
                speeches[w][d] += 1
    order = sorted(words_by_decade.index)
    rows = []
    for d in order[1:]:
        cands = []
        for w, c in counts.items():
            parts = w.split()
            if any(len(t) < 3 or t in STOP_WORDS or not t.isalpha() for t in parts) or (len(parts) == 1 and w in common):
                continue
            n = c.get(d, 0)
            if n < 10 or speeches[w][d] < 2:
                continue
            rate = 1e6 * n / words_by_decade[d]
            prior = max((1e6 * c.get(e, 0) / words_by_decade[e] for e in order if e < d), default=0)
            if rate >= 15 and prior < 0.1 * rate:
                cands.append((rate, w))
        chosen = []
        for rate, w in sorted(cands, reverse=True):
            # Skip a bigram's parts once the bigram is chosen, and vice versa.
            if any(w in c.split() or c in w.split() for _, c in chosen):
                continue
            chosen.append((rate, w))
            if len(chosen) == per_decade:
                break
        rows += [{"decade": d, "rank": k + 1, "term": w, "rate": rate} for k, (rate, w) in enumerate(chosen)]
    return pd.DataFrame(rows)


def sentence_lengths(sentences: pd.DataFrame) -> pd.DataFrame:
    s = sentences.assign(decade=sentences.year // 10 * 10, n=sentences.text.str.split().str.len())
    q = s.groupby("decade").n.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = ["p10", "p25", "p50", "p75", "p90"]
    q["over_40"] = s.groupby("decade").n.apply(lambda x: (x > 40).mean())
    return q.reset_index()


def president_profiles(df: pd.DataFrame, sigma: float = 10) -> pd.DataFrame:
    """Each president's average on several measures, as z-scores against all
    presidents and against their own era (speeches within about a decade)."""
    when = df["when"].to_numpy()
    w = np.exp(-0.5 * ((when[:, None] - when[None, :]) / sigma) ** 2)
    np.fill_diagonal(w, 0)
    rows = []
    for metric, label in PROFILE_METRICS.items():
        x = df[metric].to_numpy(float)
        resid = x - (w @ x) / w.sum(axis=1)
        g = pd.DataFrame({"president": df.president, "x": x, "r": resid}).groupby("president", sort=False).mean()
        z_all = (g.x - g.x.mean()) / g.x.std()
        z_era = g.r / g.r.std()
        rows += [{"president": p, "metric": label, "value": g.x[p], "z_all": z_all[p], "z_era": z_era[p]} for p in g.index]
    return pd.DataFrame(rows)


def new_words_lines(rows: pd.DataFrame, width: int = 38) -> pd.DataFrame:
    """One row per decade with its terms joined and wrapped, for the phone layout."""
    out = []
    for decade, g in rows.groupby("decade"):
        terms = g.sort_values("rank").term.tolist()
        lines, cur = [], ""
        for t in terms:
            nxt = f"{cur} · {t}" if cur else t
            if len(nxt) > width and cur:
                lines.append(cur)
                nxt = t
            cur = nxt
        lines.append(cur)
        out.append({"decade": decade, "line": "\n".join(lines), "terms": ", ".join(terms)})
    return pd.DataFrame(out)
