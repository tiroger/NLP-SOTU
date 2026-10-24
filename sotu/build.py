"""Aggregate processed speeches into the website's data files and chart specs.

Usage:
    uv run python -m sotu.build
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

from sotu import charts, models, topics
from sotu.process import ROOT, load_processed, load_speeches

SITE_DATA = ROOT / "site" / "data"
SITE_CHARTS = ROOT / "site" / "charts"
DERIVED = ROOT / "data" / "derived"

MODERN_PARTY_ERA = 1965  # party comparison window, matching the original analysis
ARC_POINTS = 40
# Domestic self-references and generic places that crowd out the interesting ones.
PLACE_EXCLUDE = {"united states of america", "earth", "the world", "world", "washington", "washington, dc",
                 "washington, d.c.", "d.c.", "district of columbia", "country", "north", "south", "west", "east"}
EXTRA_STOP_WORDS = {"applause", "laughter", "america", "american", "americans", "united", "states", "nation"}


def party_group(party: str) -> str:
    return party if party in ("Democratic", "Republican") else "Earlier parties"


def round_floats(obj, digits=4):
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def write_json(path: Path, obj, digits=4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(obj, digits), ensure_ascii=False, separators=(",", ":")))


# --- Statistics -------------------------------------------------------------

def log_odds_z(target: Counter, rest: Counter, prior: Counter, prior_scale: float = 0.1) -> dict[str, float]:
    """Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008).

    Returns a z-score per word: positive means over-used in ``target``. The
    prior (background counts) shrinks rare words toward zero, which is what
    makes this far more reliable than raw frequency or TF-IDF for "which words
    define this speaker".
    """
    n_t, n_r = sum(target.values()), sum(rest.values())
    a0 = prior_scale * sum(prior.values())
    z = {}
    for w, p in prior.items():
        a = prior_scale * p
        yt, yr = target.get(w, 0), rest.get(w, 0)
        delta = math.log((yt + a) / (n_t + a0 - yt - a)) - math.log((yr + a) / (n_r + a0 - yr - a))
        z[w] = delta / math.sqrt(1 / (yt + a) + 1 / (yr + a))
    return z


def content_words(c: Counter) -> Counter:
    return Counter({w: n for w, n in c.items() if len(w) > 2 and w not in STOP_WORDS and w not in EXTRA_STOP_WORDS})


def tone_arc(tone: np.ndarray) -> list[float]:
    """Smoothed tone sampled at ARC_POINTS evenly spaced positions."""
    if len(tone) == 0:
        return [0.0] * ARC_POINTS
    window = max(5, len(tone) // 12)
    smooth = pd.Series(tone).rolling(window, center=True, min_periods=1).mean().to_numpy()
    idx = np.linspace(0, len(tone) - 1, ARC_POINTS).round().astype(int)
    return smooth[idx].tolist()


# --- Build steps -------------------------------------------------------------

def speech_table(speeches, processed, topic_names, centroids):
    rows, topic_rows, para_rows, embeddings = [], [], [], []
    for s, p in zip(speeches, processed):
        emb = p["paragraph_embeddings"].astype(np.float32)
        para_words = np.array([len(x.split()) for x in p["paragraphs"]], dtype=float)
        labels, scores = topics.assign(emb, topic_names, centroids)
        para_rows.append(pd.DataFrame({
            "speech_id": s["id"], "year": int(s["date"][:4]), "paragraph": np.arange(len(labels)),
            "topic": labels, "similarity": scores.astype(np.float32), "text": p["paragraphs"],
        }))
        shares = Counter()
        for label, n in zip(labels, para_words):
            shares[label] += n / para_words.sum()
        for t in topic_names:
            topic_rows.append({"id": s["id"], "year": int(s["date"][:4]), "topic": t, "words": float(shares.get(t, 0) * p["stats"]["words"])})

        speech_vec = (emb * para_words[:, None]).sum(axis=0)
        embeddings.append(speech_vec / np.linalg.norm(speech_vec))

        dominant = np.asarray(p["emotions"], dtype=np.float32).argmax(axis=1)
        emotion_share = Counter(p["emotion_labels"][i] for i in dominant)
        n_sent = max(len(dominant), 1)
        day = pd.Timestamp(s["date"]).dayofyear
        named = {t: v for t, v in shares.items() if t != "Other"}
        rows.append({
            "id": s["id"], "president": s["president"], "party": s["party"], "party_group": party_group(s["party"]),
            "date": s["date"], "year": int(s["date"][:4]), "when": int(s["date"][:4]) + day / 366,
            "delivery": s["delivery"], "formal_sotu": s["formal_sotu"], "title": s["title"], "url": s["url"],
            **p["stats"],
            "tone": float(np.mean(p["tone"])),
            "positive_share": float(np.mean(p["tone"] > 0.5)),
            "negative_share": float(np.mean(p["tone"] < -0.5)),
            **{f"emotion_{e}": emotion_share.get(e, 0) / n_sent for e in p["emotion_labels"]},
            "top_topic": max(named, key=named.get) if named else "Other",
            "topics": {t: v for t, v in sorted(shares.items(), key=lambda kv: -kv[1]) if v >= 0.01},
            "places": [n for n, _ in p["entities"].most_common(20) if len(n) > 2 and n.lower() not in PLACE_EXCLUDE][:8],
        })
    return pd.DataFrame(rows), pd.DataFrame(topic_rows), pd.concat(para_rows, ignore_index=True), np.stack(embeddings)


def add_map_and_neighbors(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    import umap

    xy = umap.UMAP(n_neighbors=12, min_dist=0.35, metric="cosine", random_state=42).fit_transform(embeddings)
    df["map_x"], df["map_y"] = xy[:, 0], xy[:, 1]
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1)
    df["similar"] = [[df.id.iloc[j] for j in np.argsort(-row)[:5]] for row in sims]


def decade_topics(topic_df: pd.DataFrame, speeches_df: pd.DataFrame, topic_names: list[str]) -> pd.DataFrame:
    topic_df = topic_df.assign(decade=topic_df.year // 10 * 10)
    totals = speeches_df.assign(decade=speeches_df.year // 10 * 10).groupby("decade").words.sum()
    out = topic_df.groupby(["decade", "topic"]).words.sum().reset_index()
    out["share"] = out.words / out.decade.map(totals)
    return out[["decade", "topic", "share"]]


def decade_emotions(df: pd.DataFrame) -> pd.DataFrame:
    emotions = ["joy", "fear", "anger", "sadness", "surprise", "disgust"]
    d = df.assign(decade=df.year // 10 * 10)
    rows = []
    for decade, g in d.groupby("decade"):
        w = g.sentences
        for e in emotions:
            rows.append({"decade": int(decade), "emotion": e, "share": float((g[f"emotion_{e}"] * w).sum() / w.sum())})
    return pd.DataFrame(rows)


def president_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("president")
    out = pd.DataFrame({
        "party": g.party.agg(lambda x: x.mode()[0]),
        "party_group": g.party_group.first(),
        "first_year": g.year.min(),
        "speeches": g.size(),
        "tone_mean": g.tone.mean(), "tone_min": g.tone.min(), "tone_max": g.tone.max(),
        "grade_mean": g.grade_level.mean(), "words_mean": g.words.mean(),
    }).reset_index()
    return out.sort_values("first_year")


def distinctive_words(df, processed, top_n=15, min_count=5) -> pd.DataFrame:
    by_pres = defaultdict(Counter)
    for pres, p in zip(df.president, processed):
        by_pres[pres].update(content_words(p["unigrams"]))
    corpus = sum(by_pres.values(), Counter())
    rows = []
    for pres, counts in by_pres.items():
        z = log_odds_z(counts, corpus - counts, corpus)
        best = sorted((w for w in z if counts.get(w, 0) >= min_count), key=lambda w: -z[w])[:top_n]
        rows += [{"president": pres, "word": w, "z": z[w], "count": counts[w]} for w in best]
    return pd.DataFrame(rows)


def party_words(df, processed, top_n=20) -> pd.DataFrame:
    dem, rep = Counter(), Counter()
    for (year, group), p in zip(zip(df.year, df.party_group), processed):
        if year >= MODERN_PARTY_ERA:
            words = content_words(p["unigrams"]) + content_words(p["bigrams"])
            (dem if group == "Democratic" else rep if group == "Republican" else Counter()).update(words)
    prior = dem + rep
    prior = Counter({w: n for w, n in prior.items() if n >= 15})
    z = log_odds_z(rep, dem, prior)
    ranked = sorted(z, key=z.get)
    rows = [{"word": w, "z": z[w], "side": "Democratic", "dem": dem[w], "rep": rep[w]} for w in ranked[:top_n]]
    rows += [{"word": w, "z": z[w], "side": "Republican", "dem": dem[w], "rep": rep[w]} for w in ranked[-top_n:]]
    return pd.DataFrame(rows)


def word_index(df, processed) -> dict:
    """Compact per-speech counts for the word-trend explorer."""
    uni_df, bi_df, uni_tot, bi_tot = Counter(), Counter(), Counter(), Counter()
    for p in processed:
        uni_df.update(p["unigrams"].keys()); uni_tot.update(p["unigrams"])
        bi_df.update(p["bigrams"].keys()); bi_tot.update(p["bigrams"])
    vocab = [w for w, n in uni_tot.items() if n >= 8 and uni_df[w] >= 3]
    vocab += [w for w, n in bi_tot.items() if n >= 12 and bi_df[w] >= 4]
    vocab_set = set(vocab)
    postings = defaultdict(list)
    for i, p in enumerate(processed):
        for counts in (p["unigrams"], p["bigrams"]):
            for w, n in counts.items():
                if w in vocab_set:
                    postings[w].append(f"{i}:{n}")
    return {
        "ids": df.id.tolist(), "when": df["when"].round(3).tolist(), "words": df.words.tolist(),
        "terms": {w: " ".join(v) for w, v in sorted(postings.items())},
    }


def sentence_table(df, processed) -> pd.DataFrame:
    frames = []
    for (sid, year, pres, party), p in zip(df[["id", "year", "president", "party"]].itertuples(index=False), processed):
        probs = np.asarray(p["emotions"], dtype=np.float32)
        frames.append(pd.DataFrame({
            "speech_id": sid, "year": year, "president": pres, "party": party,
            "sentence": np.arange(len(p["sentences"])), "paragraph": p["sentence_paragraph"],
            "text": p["sentences"], "tone": p["tone"].astype(np.float32),
            "emotion": [p["emotion_labels"][i] for i in probs.argmax(axis=1)],
            "emotion_score": probs.max(axis=1),
        }))
    return pd.concat(frames, ignore_index=True)


def export_chart(name: str, chart: alt.TopLevelMixin) -> None:
    SITE_CHARTS.mkdir(parents=True, exist_ok=True)
    (SITE_CHARTS / f"{name}.json").write_text(chart.to_json(indent=None))


def main() -> None:
    speeches = load_speeches()
    processed = load_processed(speeches)
    topic_names, cents = topics.centroids(models.embed)

    df, topic_df, paragraph_df, embeddings = speech_table(speeches, processed, topic_names, cents)
    add_map_and_neighbors(df, embeddings)

    by_decade_topics = decade_topics(topic_df, df, topic_names)
    topic_order = by_decade_topics.groupby("topic").share.mean().sort_values(ascending=False).index.tolist()
    emotions = decade_emotions(df)
    presidents = president_summary(df)
    vocab = defaultdict(set)
    for pres, p in zip(df.president, processed):
        vocab[pres].update(p["unigrams"])
    presidents["unique_words"] = presidents.president.map(lambda x: len(vocab[x]))
    presidents["total_words"] = presidents.president.map(df.groupby("president").words.sum())
    distinctive = distinctive_words(df, processed)
    party = party_words(df, processed)

    # Website data
    speech_cols = [c for c in df.columns if c not in ("formal_sotu",)]
    write_json(SITE_DATA / "speeches.json", df[speech_cols + ["formal_sotu"]].to_dict("records"))
    write_json(SITE_DATA / "topics_by_decade.json", by_decade_topics.to_dict("records"))
    write_json(SITE_DATA / "emotions_by_decade.json", emotions.to_dict("records"))
    write_json(SITE_DATA / "presidents.json", presidents.to_dict("records"))
    write_json(SITE_DATA / "distinctive_words.json", distinctive.to_dict("records"), digits=2)
    write_json(SITE_DATA / "party_words.json", party.to_dict("records"), digits=2)
    write_json(SITE_DATA / "word_index.json", word_index(df, processed))
    write_json(SITE_DATA / "arcs.json", {s: tone_arc(p["tone"]) for s, p in zip(df.id, processed)}, digits=3)
    for sid, p in zip(df.id, processed):
        paras = defaultdict(list)
        for text, para, t in zip(p["sentences"], p["sentence_paragraph"], p["tone"]):
            paras[int(para)].append([text, float(t)])
        write_json(SITE_DATA / "text" / f"{sid}.json", [paras[i] for i in sorted(paras)], digits=2)

    # Tidy tables for the notebooks
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["topics", "places", "similar"]).to_csv(DERIVED / "speech_features.csv", index=False)
    sentence_table(df, processed).to_parquet(DERIVED / "sentences.parquet", compression="zstd")
    topic_df.to_csv(DERIVED / "speech_topics.csv", index=False)
    paragraph_df.to_parquet(DERIVED / "paragraph_topics.parquet", compression="zstd")

    # Chart specs (data by URL, relative to site/)
    write_json(SITE_CHARTS / "palette.json", charts.DARK_SWAP)
    url = lambda name: alt.UrlData(f"data/{name}.json")
    export_chart("length", charts.length_chart(url("speeches")))
    export_chart("grade", charts.grade_chart(url("speeches")))
    export_chart("diversity", charts.diversity_chart(url("speeches")))
    export_chart("tone", charts.tone_chart(url("speeches")))
    export_chart("pronouns", charts.pronoun_chart(url("speeches")))
    export_chart("emotions", charts.emotion_chart(url("emotions_by_decade")))
    export_chart("topics", charts.topic_heatmap(url("topics_by_decade"), topic_order))
    export_chart("president_tone", charts.president_tone_chart(url("presidents")))
    pres_order = presidents.sort_values("first_year", ascending=False).president.tolist()
    export_chart("distinctive", charts.distinctive_chart(url("distinctive_words"), pres_order, pres_order[0]))
    export_chart("party_words", charts.party_words_chart(url("party_words")))
    export_chart("trend", charts.trend_chart())
    export_chart("map", charts.map_chart(url("speeches")))
    export_chart("arc", charts.arc_chart())
    export_chart("speech_topics", charts.speech_topics_chart())

    write_json(SITE_DATA / "summary.json", {
        "speeches": len(df), "presidents": df.president.nunique(), "words": int(df.words.sum()),
        "first": df.date.min(), "last": df.date.max(), "latest": df.iloc[-1][["id", "president", "date"]].to_dict(),
        "spoken": int((df.delivery == "spoken").sum()), "written": int((df.delivery == "written").sum()),
        "topics": topic_order,
    })
    print(f"Built site data for {len(df)} speeches -> {SITE_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
