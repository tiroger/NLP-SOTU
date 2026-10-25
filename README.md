# The State of the Union, in data

Natural-language analysis of **every U.S. State of the Union message, 1790–2026**: 243 messages from 43 presidents, about 2 million words.

**Live site: <https://tiroger.github.io/NLP-SOTU/>**  ·  [![Deploy site to GitHub Pages](https://github.com/tiroger/NLP-SOTU/actions/workflows/pages.yml/badge.svg)](https://github.com/tiroger/NLP-SOTU/actions/workflows/pages.yml)

![Screenshot of the site](docs/screenshot.png)

## Findings and their limits

- **Language and format change together.** Word counts, sentence lengths, pronouns, Flesch–Kincaid estimates and MTLD describe the corpus; they do not isolate the effects of broadcasting, audience or delivery format.
- **Tone and emotion are model estimates.** Net tone is mean P(positive) − P(negative). Class shares use each sentence’s highest-probability class, including the actual neutral class. Historical human validation is pending; these scores do not establish changes in presidential happiness or psychological emotion.
- **Topic attention is estimated.** Every word inherits its paragraph’s dominant topic. Complete text is embedded using sentence-aware chunks; paragraph vectors pool all chunks. Decade bootstrap intervals describe message sampling variation, excluding classification error. Weak-reference agreement does not establish unbiased topic prevalence.
- **Party comparisons are exploratory.** President-level exact tests and bootstrap intervals are exported in [`site/data/comparisons.json`](site/data/comparisons.json), with spoken-only comparisons, score-threshold sensitivity and similarity checks at 5/10/20-year era bandwidths. Non-significance does not establish equivalence. Similarity permutations relabel whole presidents and weight query presidents equally.

## What's in it

| Analysis | Method |
|---|---|
| Length, reading level, vocabulary richness | Word counts, Flesch–Kincaid grade, MTLD lexical diversity (length-robust) |
| Tone of every sentence | RoBERTa 3-class sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| Emotion | DistilRoBERTa, Ekman's six + neutral (`j-hartmann/emotion-english-distilroberta-base`) |
| Policy topics by decade | `all-mpnet-base-v2` complete-text chunk pooling → logistic regression trained on Policy Agendas Project hand codes + Claude-labeled paragraphs ([validation report](data/validation/report.md)) |
| Words that define each president / party | Log-odds ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn 2008) |
| Word & phrase trends | Per-speech unigram/bigram index, searchable in the browser |
| Speech similarity map | Speech embeddings → UMAP, plus an era-removed view (Gaussian-weighted era average subtracted); nearest neighbours and "kindred speeches from other eras" per speech |
| Places mentioned, “I” vs “we” | spaCy NER and pronoun rates |
| World map of attention | Place names resolved to present-day countries (pycountry + historical aliases), share of mentions by decade |
| Anatomy of every message | Each message as a strip of paragraphs colored by topic group |
| How messages open and close | Model-estimated sentence valence across each message, averaged by era |
| Words that arrived | Terms that first became common in each decade |
| Sentence length | Distribution of words per sentence by decade |
| President profiles | z-scores on six measures, against all presidents or each president's own era |

Every chart is defined once in Altair ([`sotu/charts.py`](sotu/charts.py)). The marimo notebooks render the charts directly, and the build step exports them as Vega-Lite specs for the website.

## Repository layout

```
sotu/                 Python package
  scrape.py           Scrape the UCSB American Presidency Project index (requests + BeautifulSoup)
  text.py             Transcript cleaning, spaCy sentences/entities, style metrics
  models.py           Transformer tone, emotion and embedding models
  topics.py           Topic taxonomy (edit to change topics)
  process.py          Per-speech NLP with an on-disk cache (.cache/processed)
  build.py            Aggregation, statistics, site data + chart specs
  validate_topics.py  Topic-classifier validation (Policy Agendas benchmark + Claude-labeled history) and the classifier the build uses
  charts.py           Altair charts + palette
  visuals.py          Data for the map, anatomy, arcs, new words, sentence lengths, profiles
  places.py           Place name -> present-day country (ISO numeric) resolution
  statistics.py       President-level party tests, exact graph-label permutations and sensitivity
  provenance.py       Model/dependency/code fingerprints for caches and builds
  human_review.py     Blind audit sampling and independent human-label scoring
  verify_site.py      Publication checks run before deployment
notebooks/            marimo notebooks (01 data · 02 EDA · 03 sentiment · 04 topics · 05 generator)
data/speeches.jsonl   The corpus: one JSON object per message (metadata + paragraphs)
data/derived/         Tidy outputs for analysis (speech features, sentence scores, paragraph topics)
data/validation/      Development labels, independent human audit and validation report
data/reference/       Compressed, checksum-pinned human-coded benchmark snapshot
site/                 Static website deployed to GitHub Pages
tests/                pytest suite
```

## Running it

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```bash
uv sync --locked                          # install the locked dependency environment
uv run python -m sotu.scrape              # fetch any new messages (incremental)
uv run python -m sotu.process             # spaCy + transformers; ~20 min the first time on Apple silicon, cached after
uv run python -m sotu.build               # write site/data and site/charts
python -m http.server -d site 8000        # preview at http://localhost:8000
```

### Notebooks (marimo)

| Notebook | Contents |
|---|---|
| [`01_collect_data.py`](notebooks/01_collect_data.py) | Scraping, corpus overview, spoken vs. written, cleaning before/after |
| [`02_exploratory_analysis.py`](notebooks/02_exploratory_analysis.py) | Length, reading level, MTLD vs. unique words, distinctive words, word trends |
| [`03_sentiment_analysis.py`](notebooks/03_sentiment_analysis.py) | Tone and emotion, party comparison with permutation test and bootstrap CI, VADER cross-check |
| [`04_topics_and_similarity.py`](notebooks/04_topics_and_similarity.py) | Topic heatmap, classifier spot-checks, interactive UMAP map |
| [`05_speech_generator.py`](notebooks/05_speech_generator.py) | Party-specific Markov-chain speech generator |

```bash
uv run marimo edit notebooks/03_sentiment_analysis.py   # interactive editor
uv run marimo run notebooks/05_speech_generator.py      # app mode
```

The notebooks read the committed files in `data/` and `site/data/`, so they work right after cloning without re-running the models.

Tests: `uv run pytest`. After rebuilding, run `uv run python -m sotu.verify_site`; GitHub Pages runs the same artifact checks before deployment. The committed benchmark snapshot permits rebuilding without depending on mutable upstream dataset bytes.

## Validating the topic classifier

The [validation report](data/validation/report.md) scores topic classifiers against the [Policy Agendas Project](https://comparativeagendas.net/us/datasets)'s hand-coded State of the Union sentences (1946–2025). Because nothing comparable exists before 1946, it also scores them against 750 paragraphs from every era labeled by Claude under a written rubric. Claude agrees with the human coders on **67.3% overall**, and 81% on its high-confidence subset. The 96.5% agreement between two labeling runs measures repeatability, not accuracy. Paragraph evaluation removes held-out presidents from **both** training sources. The original labeled paragraph sample excludes passages under 20 words; the new independent human sample includes them. Per-topic scores and production-hybrid human benchmark results are in the report.

```bash
# The development sample and its Claude labels are already committed.
uv run python -m sotu.validate_topics evaluate   # writes data/validation/report.md
```

Independent historical topic, sentiment and emotion validation remains pending. [`docs/annotation.md`](docs/annotation.md) describes a blind two-reader audit, with a committed unlabeled sample in `data/validation/human_review.jsonl`. Run `uv run python -m sotu.human_review evaluate` after independent annotation; it never substitutes LLM labels for human judgments.

With labels present, `sotu.build` uses the hybrid classifier. Without them it falls back to the description-centroid method.

## Validating sentiment

The frozen SOTU model was evaluated on **2,600 human-labeled English ParlaSent test sentences**: **69.3% accuracy / 0.690 macro-F1**, compared with **51.7% / 0.518** for VADER and **40.8% accuracy** for always neutral. It overassigns neutral and has a **+0.049 mean signed score error** against human class codes (95% document-bootstrap interval +0.029 to +0.069). These results describe modern UK parliamentary language; historical SOTU accuracy remains unestablished. The model and published SOTU scores have not been recalibrated using this test set.

[Full report](data/validation/sentiment/report.md) · [Probability reliability plots](data/validation/sentiment/reliability.html) · [Pinned reference and attribution](data/reference/parlasent/ATTRIBUTION.md).

```bash
uv run python -m sotu.sentiment_validation   # frozen model, VADER, fixed baselines, calibration and score diagnostics
uv run python -m sotu.sentiment_audit ingest # after two independent readers complete their blind files
uv run python -m sotu.sentiment_audit evaluate # development metrics; test metrics withheld
```

The expanded SOTU audit contains **1,000 blank human-review items**, with **200 development / 800 final-test sentences** and disjoint presidents. Reader files hide dates, parties, speakers and model outputs. Historical human validation is **pending**. Follow [the annotation protocol](docs/annotation.md); final-test metrics require complete adjudication and explicit `evaluate --release-test`. Preserve the earlier pilot, which must not be used to tune this final test.

## Adding next year's address

A scheduled workflow ([`check-new-speeches.yml`](.github/workflows/check-new-speeches.yml)) checks the archive monthly and opens an issue when a new message appears. Then:

```bash
uv run python -m sotu.scrape && uv run python -m sotu.process && uv run python -m sotu.build
git add data site && git commit -m "Add <year> address" && git push
```

The index is fetched fresh on every scrape; `--refresh` also bypasses transcript HTML caches. Unchanged speeches use caches keyed by complete input, preprocessing code, pinned model revisions and dependency versions. Changes to those inputs trigger recomputation. [`site/data/manifest.json`](site/data/manifest.json) records the build’s code/data hashes and model revisions. Pushing changes under `site/` redeploys the site.

## Data source and caveats

Use `uv sync --locked` for a reproducible dependency environment. Models use immutable Hugging Face commit revisions.

Texts come from [The American Presidency Project](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union), UC Santa Barbara. The models were trained on modern English, so read scores for 18th- and 19th-century prose as rough trends. Most messages from 1801–1912 were written reports, not speeches, so compare eras with care.
