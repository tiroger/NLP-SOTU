import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import re
    from collections import defaultdict

    import marimo as mo
    import pandas as pd

    from sotu.process import ROOT

    DERIVED = ROOT / "data" / "derived"
    return DERIVED, defaultdict, mo, pd, random, re


@app.cell
def _(mo):
    mo.md(r"""
    # 5 · A party-flavored speech generator

    The original notebook trained a character-level LSTM in Keras. It had three
    problems that made its output meaningless:

    1. `republican_speeches = ' '.join(democrat['clean_text'])`. The
       “Republican” corpus was the Democratic one.
    2. Each character went in as a single scaled integer (`x / vocab_size`)
       instead of an embedding or one-hot vector, so the model saw “a” and “b”
       as nearly the same input.
    3. Two epochs of training on stop-word-stripped, punctuation-free text.

    A modern fix would fine-tune a small transformer, but that's heavy
    machinery for a toy. A **word-level Markov chain** trained on each party's
    sentences runs in seconds, keeps punctuation, and reproduces each party's
    stock phrases well. Pick a party, an era and a seed below.
    """)
    return


@app.cell
def _(DERIVED, pd):
    sentences = pd.read_parquet(DERIVED / "sentences.parquet", columns=["year", "party", "text"])
    return (sentences,)


@app.cell
def _(mo):
    party = mo.ui.dropdown(["Democratic", "Republican"], value="Democratic", label="Party")
    years = mo.ui.range_slider(start=1913, stop=2030, step=1, value=[1981, 2030], label="Years", show_value=True)
    order = mo.ui.slider(1, 3, value=2, label="Chain order (higher = more verbatim)", show_value=True)
    n_sentences = mo.ui.slider(3, 20, value=8, label="Sentences", show_value=True)
    seed = mo.ui.number(start=0, stop=10_000, value=42, label="Random seed")
    mo.vstack([mo.hstack([party, years], justify="start"), mo.hstack([order, n_sentences, seed], justify="start")])
    return n_sentences, order, party, seed, years


@app.cell
def _(defaultdict, order, party, re, sentences, years):
    _sel = sentences[(sentences.party == party.value) & sentences.year.between(*years.value)]
    chain = defaultdict(list)
    for _text in _sel.text:
        _tokens = ["<s>"] * order.value + re.findall(r"[\w'’-]+|[.,!?;:—]", _text) + ["</s>"]
        for _i in range(len(_tokens) - order.value):
            chain[tuple(_tokens[_i:_i + order.value])].append(_tokens[_i + order.value])
    corpus_size = len(_sel)
    return chain, corpus_size


@app.cell
def _(
    chain,
    corpus_size,
    mo,
    n_sentences,
    order,
    party,
    random,
    re,
    seed,
    years,
):
    _rng = random.Random(seed.value)

    def _sentence(max_tokens=60):
        state, out = ("<s>",) * order.value, []
        while len(out) < max_tokens:
            nxt = _rng.choice(chain[state])
            if nxt == "</s>":
                break
            out.append(nxt)
            state = state[1:] + (nxt,)
        return re.sub(r" ([.,!?;:])", r"\1", " ".join(out))

    _speech = " ".join(_sentence() for _ in range(n_sentences.value))
    mo.vstack([
        mo.md(f"*Trained on {corpus_size:,} sentences from {party.value} presidents, {years.value[0]}–{years.value[1]}.*"),
        mo.callout(mo.md(f"“{_speech}”"), kind="neutral"),
    ])
    return


if __name__ == "__main__":
    app.run()
