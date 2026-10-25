import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from sotu import charts
    from sotu.process import ROOT

    DERIVED = ROOT / "data" / "derived"
    SITE_DATA = ROOT / "site" / "data"
    return DERIVED, SITE_DATA, alt, charts, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 3 · Tone and emotion

    Now every sentence is scored by a 3-class RoBERTa model
    (`cardiffnlp/twitter-roberta-base-sentiment-latest`):
    **net tone = P(positive) − P(negative)**, averaged over the sentences of
    a speech. A DistilRoBERTa emotion model
    (`j-hartmann/emotion-english-distilroberta-base`) adds Ekman's six basic
    emotions plus neutral. Probabilities are uncalibrated estimates; historical
    human validation is pending. Neutral class shares use all three model classes,
    rather than a net-score threshold band.
    """)
    return


@app.cell
def _(DERIVED, SITE_DATA, pd):
    speeches = pd.read_csv(DERIVED / "speech_features.csv")
    sentences = pd.read_parquet(DERIVED / "sentences.parquet")
    presidents = pd.read_json(SITE_DATA / "presidents.json")
    emotions_by_decade = pd.read_json(SITE_DATA / "emotions_by_decade.json")
    return emotions_by_decade, presidents, sentences, speeches


@app.cell
def _(charts, speeches):
    charts.tone_chart(speeches)
    return


@app.cell
def _(charts, presidents):
    charts.president_tone_chart(presidents)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Do Democrats and Republicans sound different? (1965–present)

    The original notebook compared party means after 1965 and stopped there.
    Treating each speech as independent overstates the evidence: four Reagan
    addresses are not four separate data points about Republicans. So the unit
    here is the **president** (his speeches averaged). With 11 presidents
    there are only C(11, 5) = 462 ways to split them into 5 and 6, so the
    permutation test enumerates all of them (an exact p-value), and the
    confidence interval comes from a bootstrap that resamples presidents.
    Spoken-only sensitivity and graph-label permutations use the same committed
    comparison results as the website. This is an observational comparison
    assuming exchangeable presidents, not a randomized experiment.
    """)
    return


@app.cell
def _(SITE_DATA, mo, pd):
    import json

    _results = json.loads((SITE_DATA / "comparisons.json").read_text())
    mo.vstack([
        mo.md("Exploratory president-level comparisons, including spoken-only sensitivity. Non-significance does not establish equivalence; party and era were not randomized."),
        mo.ui.table(pd.DataFrame(_results["party"]).round(3), selection=None),
        mo.ui.table(pd.DataFrame(_results["neighbor_party"]).round(3), selection=None),
    ])
    return

@app.cell
def _(alt, charts, speeches):
    _modern = speeches[(speeches.year >= 1965) & speeches.party_group.isin(["Democratic", "Republican"])]
    _scale = alt.Scale(domain=charts.PARTY_GROUPS[:2], range=charts.PARTY_COLORS[:2])
    _strip = alt.Chart(_modern).mark_point(filled=True, size=60, opacity=0.8).encode(
        y=alt.Y("party_group:N", title=None),
        x=alt.X("tone:Q", title="Net tone"),
        yOffset=alt.YOffset("jitter:Q"),
        color=alt.Color("party_group:N", scale=_scale, legend=None),
        tooltip=["president:N", "date:N", alt.Tooltip("tone:Q", format="+.2f")],
    ).transform_calculate(jitter="random()")
    _mean = alt.Chart(_modern.groupby(["president", "party_group"]).tone.mean().reset_index()).mark_tick(thickness=3, size=36, color=charts.INK).encode(
        y="party_group:N", x="mean(tone):Q"
    )
    (_strip + _mean).properties(width="container", height=160)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sanity check: transformer vs. lexicon

    VADER is a widely used rule-based sentiment lexicon. Agreement with the
    transformer is a method comparison, not independent accuracy validation. Where they diverge,
    the speeches are worth reading closely. Running VADER on a sample of
    sentences takes a few seconds.
    """)
    return


@app.cell
def _(alt, charts, mo, sentences):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _sample = sentences.sample(15_000, random_state=1).copy()
    _vader = SentimentIntensityAnalyzer()
    _sample["vader"] = [_vader.polarity_scores(t)["compound"] for t in _sample.text]
    _by_speech = _sample.groupby(["speech_id", "year"]).agg(transformer=("tone", "mean"), vader=("vader", "mean")).reset_index()
    _r = _by_speech.transformer.corr(_by_speech.vader)
    mo.vstack([
        mo.md(f"Speech-level correlation between the two methods: **r = {_r:.2f}**"),
        alt.Chart(_by_speech).mark_point(filled=True, size=50, color=charts.SERIES[0]).encode(
            x=alt.X("vader:Q", title="VADER compound (mean)"),
            y=alt.Y("transformer:Q", title="RoBERTa net tone (mean)"),
            tooltip=["speech_id:N"],
        ).properties(width="container", height=300),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Emotions by decade
    """)
    return


@app.cell
def _(charts, emotions_by_decade):
    charts.emotion_chart(emotions_by_decade)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Read the extremes

    Pick a speech to see the sentences the model scored most positive and
    most negative. This is the quickest way to judge whether the scores
    mean what you think they mean.
    """)
    return


@app.cell
def _(mo, speeches):
    speech = mo.ui.dropdown(options=speeches.id.tolist()[::-1], value=speeches.id.iloc[-1], label="Speech")
    speech
    return (speech,)


@app.cell
def _(mo, sentences, speech):
    _s = sentences[sentences.speech_id == speech.value].sort_values("tone")
    _cols = ["tone", "emotion", "text"]
    mo.hstack([
        mo.vstack([mo.md("**Most negative**"), mo.ui.table(_s.head(8)[_cols].round(2), selection=None)]),
        mo.vstack([mo.md("**Most positive**"), mo.ui.table(_s.tail(8)[::-1][_cols].round(2), selection=None)]),
    ], widths="equal")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How messages open and close

    Average tone from the first sentence to the last, by era.
    """)
    return


@app.cell
def _(SITE_DATA, charts, pd):
    from sotu.visuals import ERAS

    charts.arcs_era_chart(pd.read_json(SITE_DATA / "arcs_by_era.json"), [label for _, _, label in ERAS])
    return


if __name__ == "__main__":
    app.run()
