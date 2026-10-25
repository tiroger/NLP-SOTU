# Independent human validation

External sentiment evaluation is complete on the frozen ParlaSent English test set; see [the report](../data/validation/sentiment/report.md). Independent historical SOTU topic, sentiment and emotion labels are still pending. An external benchmark and agreement between automated methods do not establish historical accuracy.

## Sentiment: the expanded audit

The new audit has **1,000 SOTU sentences**, with 50 development and 200 final-test sentences from each of four eras. Presidents are disjoint between development and final test: 9 development presidents and 34 test presidents. Allocation and sentence selection are fixed before annotation, seed 73. The master and its sampling manifest are `data/validation/sentiment_audit.jsonl` and `sentiment_audit_manifest.json`.

Reviewers can also label sentences on [the website’s annotation page](https://tiroger.github.io/NLP-SOTU/annotate/). The [owner dashboard](https://sotu-annotation-collection.tirogerl.chatgpt.site/admin) creates personal links, tracks submitted progress, and downloads each reader round. Sign in there with the ChatGPT account that owns the collection. External reviewer access requires the collection service to be publicly reachable; each assignment still requires its private link, and the dashboard and exports remain owner-only.

Create a distinct reviewer code for each person and assign them to **one round only**, A or B. Allocate manageable batches (50 sentences by default). Continue assigning the remaining sentences in both rounds until all 1,000 sentences have two independent readings. A reviewer code cannot be assigned across both rounds. Do not give the same person differently named codes in both rounds. The interface hides the other round, speaker metadata, and all model outputs. Use anonymous reviewer codes rather than personal details.

Share the generated personal URL privately. It is shown once; anyone with it can read and revise that assignment. Reviewers explicitly submit each label with **Save & next**. Saved answers resume through the same URL on another device; unsubmitted drafts remain in that browser. Concurrent edits require resolving a conflict. Keep a record of links you distribute. Notes are optional and should describe the sentence, without personal information.

Download both reader exports from the owner dashboard, preserve their filenames in a separate downloads folder, and merge them safely:

```bash
uv run python -m sotu.import_reviews a /path/to/downloads/sentiment_reader_a.jsonl
uv run python -m sotu.import_reviews b /path/to/downloads/sentiment_reader_b.jsonl
uv run python -m sotu.sentiment_audit ingest
uv run python -m sotu.sentiment_audit evaluate
```

The importer validates every sentence and label, keeps local judgments when a remote slot is blank, and refuses conflicting existing labels or notes. Resolve conflicts manually before importing again. Revisions made after ingestion also require manual reconciliation with the master; ingestion deliberately never overwrites an existing reader judgment. Export periodically as a local backup. The shared service is implemented in [`services/annotations`](../services/annotations/OPERATIONS.md); production QA assignments are excluded from study exports.

Each reader independently completes their own blind file:

- `data/validation/sentiment_reader_a.jsonl`
- `data/validation/sentiment_reader_b.jsonl`

These files contain shuffled text, opaque item IDs, a blank `sentiment` field and optional notes. They omit speaker, party, date, partition and model predictions. Do not consult the master metadata, other reader's judgments, or model output while labeling. Fill `sentiment` with `negative`, `neutral`, `positive`, or `ambiguous`; preserve the item IDs and text.

Label **expressed textual valence**, not the president's psychological state, party reputation, or approval of a policy. A factual description of harmful conditions can be negative even if it supports a policy the speaker likes. Use historical meanings. For example, “Unemployment has fallen” expresses positive conditions; “Many families cannot afford housing” expresses negative conditions; “The bill was introduced yesterday” is neutral. Use ambiguous when mixed valence or missing context prevents a defensible predominant label. Do not silently treat ambiguous as neutral.

Run:

```bash
uv run python -m sotu.sentiment_audit ingest
uv run python -m sotu.sentiment_audit evaluate
```

Ingestion preserves the two judgments and accepts agreements automatically. A third reader or documented discussion must adjudicate disagreements in the master's `adjudicated.sentiment` field. The evaluator requires both reader labels and the adjudication, reports ambiguity separately, and never invents human judgments. Reader notes remain in the separate reader files for adjudication.

The default evaluation exposes development metrics only. Use development labels for debugging, selecting a model, or fitting calibration. Freeze those decisions before releasing final-test metrics:

```bash
uv run python -m sotu.sentiment_audit evaluate --release-test
```

This requires all development and test items to have complete, adjudicated judgments. Do not tune on the final test. A later revision informed by test results requires another untouched test sample. The report includes per-era classification, classwise probability reliability, Brier/log loss, and mean signed error of P(positive) − P(negative) against human category codes −1/0/+1. Category codes do not establish an interval scale of psychological intensity.

Sentence selection is random within each era/partition's selected-president pool. Recorded weights correct the unequal era sampling rates **within those pools**; estimates for the final test are not estimates for all presidents. Inspect coverage by era and delivery and rare-class support before making subgroup claims. The 200-sentence development set is a small pilot; calibration estimates may themselves be uncertain.

The earlier 200 affect sentences in `human_review.jsonl` are preserved unchanged. They overlap final-test presidents, so their sentiment labels must not be used for model selection in this audit. The new sentiment workflow supersedes that pilot for final sentiment validation. The older emotion and topic workflow below remains available.

## Topics and emotion: the original audit

`data/validation/human_review.jsonl` contains 240 topic paragraphs and the earlier 200 affect sentences. The original preparation command never overwrites existing annotations. Topic sampling uses seed 23, four eras, and short (<20 words) and long paragraph strata. Inclusion weights are recorded.

Two readers independently complete `reviewer_a` and `reviewer_b`. Topic readers use `sotu/topics.py` and `sotu.validate_topics.rubric()`, assigning the dominant policy topic. Greetings and passages without a policy point are Other. Preserve historical meanings; use ambiguous if no defensible single label applies.

For emotion, label expressed anger, disgust, fear, joy, neutral, sadness or surprise. These are textual judgments, not diagnoses. Use ambiguous rather than inventing an emotion. Preserve both original labels and complete `adjudicated` after resolving disagreements.

Run `uv run python -m sotu.human_review evaluate` after a rebuild. Topic scoring removes each test president from both the benchmark and Claude paragraph training sources. Human labels never enter that training. Emotion uses the pinned pretrained model. The report records completion, ambiguity, reviewer agreement and weighted metrics; partial completion is provisional. The existing Claude labels remain weak supervision and a development reference.
