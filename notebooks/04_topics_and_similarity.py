import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from sotu import charts
    from sotu.process import ROOT
    from sotu.topics import MIN_SIMILARITY, TOPICS

    DERIVED = ROOT / "data" / "derived"
    SITE_DATA = ROOT / "site" / "data"
    return DERIVED, MIN_SIMILARITY, SITE_DATA, TOPICS, alt, charts, mo, pd


@app.cell
def _(MIN_SIMILARITY, mo):
    mo.md(rf"""
    # 4 · Topics and speech similarity

    Each paragraph is embedded with `all-mpnet-base-v2`. Each topic in
    `sotu/topics.py` is described by a few sentences written in both modern
    and 19th-century phrasing, and the topic's centroid is the average of
    their embeddings. A paragraph gets the closest topic if its cosine
    similarity is at least **{MIN_SIMILARITY}**; otherwise it is “Other”
    (greetings, applause lines, ceremony).

    A fixed taxonomy gives stable, comparable labels from year to year,
    which matters on a site that gets a new speech every year. An
    unsupervised model (LDA, BERTopic) would rediscover different clusters on
    every re-run. To change the taxonomy, edit `TOPICS` and re-run
    `python -m sotu.build`. Embeddings are cached, so that takes about a minute.
    """)
    return


@app.cell
def _(DERIVED, SITE_DATA, pd):
    speeches = pd.read_csv(DERIVED / "speech_features.csv")
    paragraphs = pd.read_parquet(DERIVED / "paragraph_topics.parquet")
    by_decade = pd.read_json(SITE_DATA / "topics_by_decade.json")
    summary = pd.read_json(SITE_DATA / "summary.json", typ="series")
    return by_decade, paragraphs, speeches, summary


@app.cell
def _(by_decade, charts, summary):
    charts.topic_heatmap(by_decade, list(summary["topics"]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spot-check the classifier

    Pick a topic to see random paragraphs assigned to it, along with their
    similarity scores. Paragraphs near the threshold are where
    misclassifications live.
    """)
    return


@app.cell
def _(TOPICS, mo):
    topic = mo.ui.dropdown(options=list(TOPICS) + ["Other"], value="Health care", label="Topic")
    era = mo.ui.range_slider(start=1790, stop=2030, step=10, value=[1790, 2030], label="Years", show_value=True)
    mo.hstack([topic, era], justify="start")
    return era, topic


@app.cell
def _(era, mo, paragraphs, topic):
    _sel = paragraphs[(paragraphs.topic == topic.value) & paragraphs.year.between(*era.value)]
    _sample = _sel.sample(min(12, len(_sel)), random_state=0).sort_values("year")
    mo.vstack([
        mo.md(f"{len(_sel):,} paragraphs labelled **{topic.value}** in {era.value[0]}–{era.value[1]}."),
        mo.ui.table(_sample[["year", "speech_id", "similarity", "text"]].round(3), selection=None, page_size=12),
    ])
    return


@app.cell
def _(alt, charts, paragraphs):
    _counts = paragraphs.assign(other=paragraphs.topic == "Other", decade=paragraphs.year // 10 * 10)
    alt.Chart(_counts.groupby("decade").other.mean().reset_index()).mark_line(color=charts.SERIES[0]).encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("other:Q", title="Share of paragraphs labelled Other", axis=alt.Axis(format=".0%")),
    ).properties(width="container", height=200, title="Unclassified share by decade")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Speech similarity map

    Each speech's vector is the word-weighted average of its paragraph
    embeddings, projected to 2-D with UMAP (cosine metric). The main axis of
    variation is **time**: an 1830s message has more in common with an 1850s
    message from the other party than with anything from the 1990s.
    """)
    return


@app.cell
def _(charts, mo, speeches):
    chart = mo.ui.altair_chart(charts.map_chart(speeches))
    chart
    return (chart,)


@app.cell
def _(chart, mo):
    _picked = chart.value
    mo.ui.table(_picked[["president", "date", "party", "top_topic", "tone", "grade_level"]], selection=None) if len(_picked) else mo.md(
        "*Drag a box on the map to list the speeches inside it.*"
    )
    return


if __name__ == "__main__":
    app.run()
