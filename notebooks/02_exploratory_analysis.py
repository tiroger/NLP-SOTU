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

    DERIVED = ROOT / "data" / "derived"
    SITE_DATA = ROOT / "site" / "data"
    return DERIVED, SITE_DATA, alt, charts, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 2 · Exploratory analysis

    Length, complexity, vocabulary and the words that define each president.
    Every chart here comes from `sotu/charts.py`, the same Altair code the
    website uses. Data comes from `uv run python -m sotu.build`.
    """)
    return


@app.cell
def _(DERIVED, SITE_DATA, pd):
    speeches = pd.read_csv(DERIVED / "speech_features.csv")
    distinctive = pd.read_json(SITE_DATA / "distinctive_words.json")
    party_words = pd.read_json(SITE_DATA / "party_words.json")
    return distinctive, party_words, speeches


@app.cell
def _(mo):
    mo.md(r"""
    ## Length and reading level

    The shaded band is 1801–1912, when every message was written. Spoken
    messages are shorter, and the reading level has dropped steadily since
    the early 20th century.
    """)
    return


@app.cell
def _(charts, mo, speeches):
    mo.ui.tabs({
        "Words": charts.length_chart(speeches),
        "Grade level": charts.grade_chart(speeches),
        "Lexical diversity": charts.diversity_chart(speeches),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the old “unique words” chart was misleading

    The original EDA ranked presidents by the number of distinct words across
    all their speeches. That number mostly measures **how much** a president
    wrote: more speeches, and longer ones, contain more distinct words.
    Below, each dot is a president. Unique-word count tracks total words
    almost perfectly. MTLD, the length-robust measure used now, doesn't.
    """)
    return


@app.cell
def _(SITE_DATA, alt, charts, mo, pd, speeches):
    _by_pres = pd.read_json(SITE_DATA / "presidents.json").merge(
        speeches.groupby("president").lexical_diversity.mean().rename("mtld").reset_index()
    )
    _corr = pd.DataFrame({
        "measure": ["unique words", "mean MTLD"],
        "Spearman ρ with total words": [
            round(_by_pres.total_words.corr(_by_pres.unique_words, method="spearman"), 2),
            round(_by_pres.total_words.corr(_by_pres.mtld, method="spearman"), 2),
        ],
    })
    _base = alt.Chart(_by_pres).mark_point(filled=True, size=70).encode(
        x=alt.X("total_words:Q", title="Total words across all messages", scale=alt.Scale(type="log")),
        color=alt.Color("party_group:N", scale=alt.Scale(domain=charts.PARTY_GROUPS, range=charts.PARTY_COLORS), title=None),
        tooltip=["president:N", alt.Tooltip("total_words:Q", format=","), alt.Tooltip("unique_words:Q", format=","),
                 alt.Tooltip("mtld:Q", format=".1f")],
    ).properties(width=300, height=260)
    mo.vstack([
        alt.hconcat(
            _base.encode(y=alt.Y("unique_words:Q", title="Unique words (old metric)", scale=alt.Scale(type="log"))),
            _base.encode(y=alt.Y("mtld:Q", title="Mean MTLD (length-robust)")),
        ),
        mo.ui.table(_corr, selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distinctive words

    Word clouds of raw frequency show every president saying *government*,
    *people* and *congress*. Here a **log-odds ratio with an informative
    Dirichlet prior** (Monroe, Colaresi & Quinn, 2008) compares each president
    with all the others. The prior shrinks rare words, so a word one president
    said twice doesn't outrank a word another said 200 times.
    """)
    return


@app.cell
def _(mo, speeches):
    _order = speeches.drop_duplicates("president").president.tolist()[::-1]
    president = mo.ui.dropdown(options=_order, value=_order[0], label="President")
    president
    return (president,)


@app.cell
def _(alt, charts, distinctive, president):
    alt.Chart(distinctive[distinctive.president == president.value]).mark_bar(
        color=charts.SERIES[0], cornerRadiusEnd=4
    ).encode(
        x=alt.X("z:Q", title="log-odds z-score"),
        y=alt.Y("word:N", sort="-x", title=None),
        tooltip=["word:N", "count:Q", alt.Tooltip("z:Q", format=".1f")],
    ).properties(width="container", height=alt.Step(20), title=president.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Democrats vs. Republicans, 1965–present

    The same statistic pits one party's presidents against the other's,
    including two-word phrases.
    """)
    return


@app.cell
def _(charts, party_words):
    charts.party_words_chart(party_words)
    return


@app.cell
def _(mo):
    mo.md("""
    ## “I” versus “we”
    """)
    return


@app.cell
def _(charts, speeches):
    charts.pronoun_chart(speeches)
    return


@app.cell
def _(mo):
    terms = mo.ui.text(value="slavery, freedom, liberty", label="Word trends (comma-separated)", full_width=True)
    terms
    return (terms,)


@app.cell
def _(SITE_DATA, charts, pd, terms):
    import json as _json

    _index = _json.loads((SITE_DATA / "word_index.json").read_text())
    _when = pd.Series(_index["when"]).astype(int)
    _words_by_year = pd.Series(_index["words"]).groupby(_when).sum()
    _rows = []
    for _term in [t.strip().lower() for t in terms.value.split(",") if t.strip()][:5]:
        if _term not in _index["terms"]:
            continue
        _counts = pd.Series(0.0, index=_words_by_year.index)
        for _pair in _index["terms"][_term].split():
            _i, _n = map(int, _pair.split(":"))
            _counts[_when[_i]] += _n
        _rate = 1e6 * _counts.rolling(5, center=True, min_periods=1).sum() / _words_by_year.rolling(5, center=True, min_periods=1).sum()
        _rows += [{"term": _term, "when": y, "rate": r} for y, r in _rate.items()]
    charts.trend_chart(pd.DataFrame(_rows, columns=["term", "when", "rate"]))
    return


if __name__ == "__main__":
    app.run()
