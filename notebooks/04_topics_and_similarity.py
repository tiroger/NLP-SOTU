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
    from sotu.topics import TOPICS

    DERIVED = ROOT / "data" / "derived"
    SITE_DATA = ROOT / "site" / "data"
    return DERIVED, SITE_DATA, TOPICS, alt, charts, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 4 · Topics and speech similarity

    Each paragraph is embedded with `all-mpnet-base-v2` and classified into
    one of 19 policy topics (or “Other”: greetings, applause lines, ceremony)
    by a logistic regression trained on:

    - 26,000 State of the Union sentences from 1946–2025, hand-coded by the
      [Policy Agendas Project](https://comparativeagendas.net/us/datasets);
    - 750 paragraphs from every era, labeled by Claude under a written rubric
      (`sotu/validate_topics.py`), so the model also learns 19th-century
      topics such as Native American affairs, currency and territories,
      which the modern benchmark barely covers;
    - the topic descriptions in `sotu/topics.py`, as anchors.

    Without the Claude labels the build falls back to the original method:
    the nearest topic description by cosine similarity. A fixed taxonomy gives
    stable, comparable labels from year to year, which matters on a site that
    gets a new speech every year.
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
    from sotu.validate_topics import REPORT_PATH

    mo.md(
        REPORT_PATH.read_text().replace("# Topic classifier validation", "## How accurate is it?")
        if REPORT_PATH.exists()
        else "## How accurate is it?\n\nRun `uv run python -m sotu.validate_topics evaluate` to generate the validation report."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spot-check the classifier

    Pick a topic to see random paragraphs assigned to it, with the
    classifier's uncalibrated score. Low-score paragraphs are where
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
        mo.ui.table(_sample[["year", "speech_id", "score", "text"]].round(3), selection=None, page_size=12),
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

    Complete paragraph text is chunked and pooled before projection. The map is exploratory;
    era adjustment does not isolate party or personal style. Historical human topic validation
    remains pending; Claude-reference agreement is not human accuracy.

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


@app.cell
def _(mo):
    mo.md(r"""
    ## The anatomy of every message

    One strip per message, paragraphs colored by topic group and sized by words.
    """)
    return


@app.cell
def _(SITE_DATA, charts, pd, speeches):
    from sotu.visuals import TOPIC_GROUPS

    charts.anatomy_chart(pd.read_json(SITE_DATA / "anatomy.json"), list(TOPIC_GROUPS), speeches.year.tolist())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where presidents look

    Share of foreign-country mentions by decade; use the slider.
    """)
    return


@app.cell
def _(SITE_DATA, charts, pd):
    charts.world_map_chart(
        pd.read_json(SITE_DATA / "places_by_decade.json"),
        "https://cdn.jsdelivr.net/npm/vega-datasets@2.11.0/data/world-110m.json",
    )
    return


if __name__ == "__main__":
    app.run()
