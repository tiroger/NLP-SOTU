"""Evaluate the frozen SOTU sentiment model on pinned external human labels.

Run: uv run python -m sotu.sentiment_validation
No fitting, threshold selection, or calibration is performed on the test set.
"""

import json
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sotu import models
from sotu.process import ROOT
from sotu.provenance import file_hash, fingerprint, pipeline_manifest

LABELS = ["negative", "neutral", "positive"]
VALUES = np.array([-1., 0., 1.])
REFERENCE = ROOT / "data" / "reference" / "parlasent"
OUTPUT = ROOT / "data" / "validation" / "sentiment"


def validate_probabilities(probs, n):
    if n == 0:
        raise ValueError("At least one labeled example is required")
    p = np.asarray(probs, dtype=float)
    if p.shape != (n, 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite (n, 3) probabilities in negative/neutral/positive order")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    return p / p.sum(axis=1, keepdims=True)


def reliability(y, probs, bins=10, weights=None):
    """One-vs-rest reliability, preserving empty bins as missing observations."""
    p = validate_probabilities(probs, len(y))
    y = np.asarray(y, dtype=int)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    for c, label in enumerate(LABELS):
        bucket = np.minimum((p[:, c] * bins).astype(int), bins - 1)
        for b in range(bins):
            mask = bucket == b
            if not mask.any():
                continue
            rows.append({"class": label, "bin_low": b / bins, "bin_high": (b + 1) / bins,
                "n": int(mask.sum()), "weight": float(weights[mask].sum()),
                "mean_probability": float(np.average(p[mask, c], weights=weights[mask])),
                "observed_frequency": float(np.average(y[mask] == c, weights=weights[mask]))})
    return rows


def evaluate_predictions(gold, prediction, *, probs=None, scores=None, weights=None):
    gold = list(gold)
    if not gold or any(label not in LABELS for label in gold) or any(label not in LABELS for label in prediction):
        raise ValueError("Nonempty gold/predictions must use the three sentiment labels")
    y = np.array([LABELS.index(label) for label in gold])
    prediction = list(prediction)
    if len(prediction) != len(y):
        raise ValueError("Prediction and label counts differ")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(y),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Weights must be positive, finite and aligned")
    report = classification_report(gold, prediction, labels=LABELS, output_dict=True,
                                   sample_weight=w, zero_division=0)
    result = {"n": len(y), "accuracy": float(np.average(np.array(gold) == prediction, weights=w)),
              "macro_f1": float(report["macro avg"]["f1-score"]), "per_class": {label: report[label] for label in LABELS},
              "confusion_matrix": confusion_matrix(gold, prediction, labels=LABELS, sample_weight=w).tolist(),
              "confusion_order": LABELS}
    if probs is not None:
        p = validate_probabilities(probs, len(y))
        result["brier_multiclass_sum"] = float(np.average(((p - np.eye(3)[y]) ** 2).sum(axis=1), weights=w))
        result["log_loss"] = float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)), weights=w))
        bins = reliability(y, p, weights=w)
        result["reliability"] = bins
        result["classwise_ece_10_bins"] = {label: float(sum(
            r["weight"] * abs(r["mean_probability"] - r["observed_frequency"]) for r in bins if r["class"] == label) / w.sum()) for label in LABELS}
        scores = p @ VALUES
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(y),) or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite and aligned")
        residual = scores - VALUES[y]
        result["score_mae_vs_class_code"] = float(np.average(np.abs(residual), weights=w))
        result["score_bias_vs_class_code"] = float(np.average(residual, weights=w))
        result["human_mean_class_code"] = float(np.average(VALUES[y], weights=w))
        result["model_mean_score"] = float(np.average(scores, weights=w))
    return result


def cluster_intervals(gold, probs, groups, seed=23, draws=2000):
    """Resample whole source documents, preserving dependence within speeches."""
    y = np.array([LABELS.index(label) for label in gold])
    p = validate_probabilities(probs, len(y))
    _, owner = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(owner.max()) + 1
    # Counts, nine confusion cells, score residual, absolute residual, Brier, NLL.
    stats = np.zeros((n_groups, 14))
    pred = p.argmax(axis=1)
    np.add.at(stats[:, 0], owner, 1)
    for a in range(3):
        for b in range(3):
            np.add.at(stats[:, 1 + 3*a + b], owner, ((y == a) & (pred == b)).astype(float))
    residual = p @ VALUES - VALUES[y]
    np.add.at(stats[:, 10], owner, residual)
    np.add.at(stats[:, 11], owner, np.abs(residual))
    np.add.at(stats[:, 12], owner, ((p - np.eye(3)[y]) ** 2).sum(axis=1))
    np.add.at(stats[:, 13], owner, -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))
    rng = np.random.default_rng(seed)
    sampled = stats[rng.integers(0, n_groups, (draws, n_groups))].sum(axis=1)
    cm = sampled[:, 1:10].reshape(-1, 3, 3)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(axis=1) + cm.sum(axis=2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean(axis=1)
    series = {"accuracy": tp.sum(axis=1) / sampled[:, 0], "macro_f1": f1,
              "score_bias_vs_class_code": sampled[:, 10] / sampled[:, 0],
              "score_mae_vs_class_code": sampled[:, 11] / sampled[:, 0],
              "brier_multiclass_sum": sampled[:, 12] / sampled[:, 0], "log_loss": sampled[:, 13] / sampled[:, 0]}
    return {"unit": "source document", "groups": n_groups, "draws": draws, "seed": seed,
            "percentile_95": {k: np.quantile(v, [.025, .975]).tolist() for k, v in series.items()}}


def load_reference():
    source = json.loads((REFERENCE / "source.json").read_text())
    for name, metadata in source["files"].items():
        if file_hash(REFERENCE / name) != metadata["sha256"]:
            raise ValueError(f"Reference snapshot checksum mismatch: {name}")
    test = pd.read_json(REFERENCE / "ParlaSent_EN_test.jsonl", lines=True)
    train = pd.read_json(REFERENCE / "ParlaSent_EN.jsonl", lines=True)
    test["label"] = test.label.str.lower()
    train["label"] = train.label.str.lower()
    if len(test) != 2600 or not set(test.label).issubset(LABELS):
        raise ValueError("Unexpected English test set size or label schema")
    if test.sentence.duplicated().any():
        raise ValueError("Duplicate English test sentences require explicit treatment")
    return source, train, test


def write_report(results):
    model = results["systems"]["sotu_roberta"]
    neutral_predictions = sum(row[1] for row in model["confusion_matrix"])
    bias_ci = model["bootstrap"]["percentile_95"]["score_bias_vs_class_code"]
    lines = ["# External sentiment validation", "", "## Protocol", "",
        "Frozen SOTU RoBERTa evaluated on ParlaSent's 2,600 English test sentences. No fitting, calibration, threshold tuning or model selection was performed on this test set.",
        "Labels measure textual valence in parliamentary speech. They do not measure presidential happiness, endorsement or psychological emotion.",
        "The English test sentences were annotated by one trained annotator; the sample covers mid-length UK parliamentary sentences, not historical U.S. addresses.",
        "", "## Classification", "", "| System | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name, r in results["systems"].items():
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    lines += ["", "### Frozen SOTU model, by class", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        r = model["per_class"][label]
        lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ["", "### Confusion matrix", "", "Rows are human labels; columns are predicted negative / neutral / positive.", "",
              "| Human label | Negative | Neutral | Positive |", "|---|---:|---:|---:|"]
    for label, row in zip(LABELS, model["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(int(x)) for x in row) + " |")
    lines += ["", "## Interpretation", "",
        f"The frozen model outperforms the fixed baselines on this benchmark. It predicts neutral for {int(neutral_predictions):,} sentences ({neutral_predictions/model['n']:.1%}), versus {int(model['per_class']['neutral']['support']):,} ({model['per_class']['neutral']['support']/model['n']:.1%}) in the human reference. Many positive and negative passages are compressed into neutral.",
        f"Its net score is higher than the human category-code average by {model['score_bias_vs_class_code']:+.3f} (bootstrap interval {bias_ci[0]:+.3f} to {bias_ci[1]:+.3f}). This bias is specific to this benchmark; subtracting it from historical SOTU scores would be unjustified without SOTU labels.",
        "", "## Probability calibration and net score", "",
        "Multiclass Brier is the sum of three squared probability errors (range 0–2); log loss uses natural logarithms. Lower values are better. VADER compound is not a class-probability distribution, so probability metrics are omitted for VADER.",
        "", "| System | Brier | Log loss | Mean score | Human mean class code | Mean signed error |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["sotu_roberta", "training_class_prior"]:
        r = results["systems"][name]
        lines.append(f"| {name} | {r['brier_multiclass_sum']:.3f} | {r['log_loss']:.3f} | {r['model_mean_score']:+.3f} | {r['human_mean_class_code']:+.3f} | {r['score_bias_vs_class_code']:+.3f} |")
    lines += ["", "The human comparison codes negative = −1, neutral = 0, positive = +1. Net score is the model's expected value of that category code; its magnitude has not been validated as psychological intensity.",
        "", "### Source-document bootstrap intervals (95%)", "", "| Metric | Lower | Upper |", "|---|---:|---:|"]
    for name, interval in model["bootstrap"]["percentile_95"].items():
        lines.append(f"| {name} | {interval[0]:+.3f} | {interval[1]:+.3f} |")
    lines += ["", f"Resampled {model['bootstrap']['groups']} source documents, 2,000 times, seed 23. Intervals describe this benchmark, not transfer to SOTU.",
        "", "[Classwise reliability plots](reliability.html) · [Full metrics](results.json) · [Per-sentence predictions](predictions.jsonl)",
        "", "## Limits and next test", "",
        "This is an external-domain diagnostic, not a measurement of historical SOTU accuracy. An error on this benchmark does not itself establish an error on SOTU; success would not establish historical validity either.",
        "Any model revision prompted by these results must be evaluated on a fresh untouched test, rather than repeatedly optimizing this score.",
        "Independent SOTU annotations remain pending. Use the president-separated development/test sample in sentiment_audit.jsonl to assess per-era signed score error and the historical trend.",
        "", "## Source and attribution", "",
        "Mochtak, Rupnik, Meden and Ljubešić, ParlaSent 1.0 (CC BY-SA 4.0): https://www.clarin.si/repository/xmlui/handle/11356/1868",
        f"Pinned dataset revision: `{results['reference']['revision']}`. Checksums, model revision and implementation hashes are recorded in results.json.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines))


def main():
    source, train, test = load_reference()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_key = fingerprint({"test": source["files"]["ParlaSent_EN_test.jsonl"]["sha256"], "pipeline": pipeline_manifest()})
    cache = ROOT / ".cache" / "sentiment" / f"{cache_key}.npy"
    if cache.exists():
        probs = np.load(cache)
    else:
        labels, raw = models.tone_probabilities(test.sentence.tolist())
        probs = raw[:, [labels.index(label) for label in LABELS]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, probs)
    probs = validate_probabilities(probs, len(test))
    vader = SentimentIntensityAnalyzer()
    compound = np.array([vader.polarity_scores(t)["compound"] for t in test.sentence])
    vader_labels = np.where(compound > .05, "positive", np.where(compound < -.05, "negative", "neutral"))
    prior = train.label.value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=float)
    prior /= prior.sum()
    prior_probs = np.tile(prior, (len(test), 1))
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    systems = {"sotu_roberta": evaluate_predictions(test.label, pred, probs=probs),
               "training_class_prior": evaluate_predictions(test.label, [LABELS[prior.argmax()]] * len(test), probs=prior_probs),
               "always_neutral": evaluate_predictions(test.label, ["neutral"] * len(test), scores=np.zeros(len(test))),
               "vader_default": evaluate_predictions(test.label, vader_labels, scores=compound)}
    systems["sotu_roberta"]["bootstrap"] = cluster_intervals(test.label, probs, test.document_id)
    results = {"schema": 1, "reference": source, "model": {"name": models.TONE_MODEL, "revision": models.REVISIONS[models.TONE_MODEL]},
        "protocol": "frozen external evaluation; no fitting or calibration on test", "systems": systems,
        "implementation_sha256": file_hash(Path(__file__)), "pipeline": pipeline_manifest(), "vader_version": importlib.metadata.version("vaderSentiment"),
        "scope": "modern UK parliamentary sentences; historical SOTU validation remains pending"}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    records = test[["sentence", "label", "document_id", "name", "date"]].copy()
    records["prediction"] = pred
    for c, label in enumerate(LABELS):
        records[f"p_{label}"] = probs[:, c]
    records["net_score"] = probs @ VALUES
    records["human_class_code"] = VALUES[[LABELS.index(x) for x in test.label]]
    records["signed_error"] = records.net_score - records.human_class_code
    records.to_json(OUTPUT / "predictions.jsonl", orient="records", lines=True, date_format="iso", force_ascii=False)
    records[records.prediction != records.label].assign(confidence=probs.max(axis=1)[pred != test.label]).sort_values(
        "confidence", ascending=False).head(30).to_json(OUTPUT / "confident_errors.jsonl", orient="records", lines=True, date_format="iso", force_ascii=False)
    import altair as alt
    frame = pd.DataFrame(systems["sotu_roberta"]["reliability"])
    diagonal = alt.Chart(pd.DataFrame({"x": [0., 1.], "y": [0., 1.]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    chart = alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X("mean_probability:Q", title="Predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_frequency:Q", title="Observed class frequency", scale=alt.Scale(domain=[0, 1])),
        tooltip=["class:N", "n:Q", "mean_probability:Q", "observed_frequency:Q"]).properties(width=260, height=260)
    panels = [(chart.transform_filter(alt.datum["class"] == label) + diagonal).properties(title=label) for label in LABELS]
    alt.hconcat(*panels).save(OUTPUT / "reliability.html")
    write_report(results)
    print(json.dumps({name: {k: v for k, v in r.items() if k in ["accuracy", "macro_f1", "brier_multiclass_sum", "score_bias_vs_class_code"]} for name, r in systems.items()}, indent=2))


if __name__ == "__main__":
    main()
