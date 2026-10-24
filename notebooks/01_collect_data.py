import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter

    import altair as alt
    import marimo as mo
    import pandas as pd

    from sotu import charts  # noqa: F401  (registers the shared Altair theme)
    from sotu import scrape
    from sotu.text import clean_paragraphs

    return Counter, alt, clean_paragraphs, json, mo, pd, scrape


@app.cell
def _(mo):
    mo.md(r"""
    # 1 · Collecting the State of the Union corpus

    Every annual message is scraped from
    [The American Presidency Project](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union)
    at UC Santa Barbara.

    The original version of this project drove a Selenium browser through the
    index once per year, clicking the first link labelled with that year. That
    approach was slow, and it silently dropped every year with **two** messages
    (1790, 1953, 1961, 1972–81) as well as 1790–92. The guidebook's table
    already links every message directly, and its columns say whether the
    message was **spoken or written**, so `sotu/scrape.py` just parses that
    table with `requests` + BeautifulSoup.

    Re-running the scraper only fetches messages that aren't in
    `data/speeches.jsonl` yet, so adding next year's address is one command:

    ```bash
    uv run python -m sotu.scrape && uv run python -m sotu.process && uv run python -m sotu.build
    ```
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Check the archive for new speeches")
    run
    return (run,)


@app.cell
def _(mo, run, scrape):
    if run.value:
        with mo.redirect_stdout():
            scrape.main()
    return


@app.cell
def _(json, pd, scrape):
    with scrape.OUT_PATH.open() as _f:
        speeches = [json.loads(line) for line in _f]
    corpus = pd.DataFrame([
        {
            "id": s["id"], "date": s["date"], "president": s["president"], "party": s["party"],
            "delivery": s["delivery"], "formal_sotu": s["formal_sotu"],
            "paragraphs": len(s["paragraphs"]), "words": sum(len(p.split()) for p in s["paragraphs"]),
        }
        for s in speeches
    ])
    corpus["year"] = corpus.date.str[:4].astype(int)
    return corpus, speeches


@app.cell
def _(corpus, mo):
    mo.hstack([
        mo.stat(f"{len(corpus)}", label="Messages"),
        mo.stat(f"{corpus.president.nunique()}", label="Presidents"),
        mo.stat(f"{corpus.words.sum() / 1e6:.2f}M", label="Words (raw)"),
        mo.stat(f"{(corpus.delivery == 'spoken').sum()} / {(corpus.delivery == 'written').sum()}", label="Spoken / written"),
        mo.stat(corpus.date.max(), label="Most recent"),
    ], justify="start")
    return


@app.cell
def _(corpus, mo):
    mo.ui.table(corpus, page_size=10, label="The corpus (one row per message)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spoken or written?

    Washington and Adams spoke to Congress in person. Jefferson thought that
    looked too much like a king's speech from the throne and sent a written
    report instead, and every president did the same until Woodrow Wilson in
    1913. The format matters for everything downstream: written reports are
    long, technical and formal.
    """)
    return


@app.cell
def _(alt, corpus):
    alt.Chart(corpus).mark_tick(thickness=2, size=22).encode(
        x=alt.X("year:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("delivery:N", title=None),
        color=alt.Color("delivery:N", legend=None),
        tooltip=["president:N", "date:N", "delivery:N", alt.Tooltip("words:Q", format=",")],
    ).properties(width="container", height=110)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cleaning

    Modern transcripts include things the president didn't say: `[Applause]`,
    `The President.` speaker tags, audience chants and heckles
    (“Audience members. U.S.A.!”, “Rep. Greene. Liar!”), editorial notes and
    signature lines. `sotu.text.clean_paragraphs` strips them but keeps
    punctuation and case. The old pipeline lower-cased everything and deleted
    punctuation *before* sentiment analysis, which threw away sentence
    boundaries, negation cues and emphasis that sentiment models rely on.
    """)
    return


@app.cell
def _(corpus, mo):
    pick = mo.ui.dropdown(options=corpus.id.tolist()[::-1], value="2024-biden", label="Speech")
    pick
    return (pick,)


@app.cell
def _(clean_paragraphs, mo, pick, speeches):
    _speech = next(s for s in speeches if s["id"] == pick.value)
    _raw = _speech["paragraphs"]
    _clean = clean_paragraphs(_raw)
    _removed = [p for p in _raw if p not in _clean]
    mo.vstack([
        mo.md(f"**{len(_raw)}** raw paragraphs → **{len(_clean)}** after cleaning. Paragraphs dropped or edited:"),
        mo.ui.table([{"paragraph": p} for p in _removed[:200]], page_size=8, selection=None),
    ])
    return


@app.cell
def _(Counter, corpus, mo):
    _by_pres = Counter(corpus.president)
    mo.md(
        "### Messages per president\n\n"
        + ", ".join(f"{p} ({n})" for p, n in sorted(_by_pres.items(), key=lambda kv: -kv[1]))
    )
    return


if __name__ == "__main__":
    app.run()
