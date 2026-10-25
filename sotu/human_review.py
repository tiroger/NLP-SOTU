"""Prepare a blind human audit or score independently adjudicated labels.

Run `python -m sotu.human_review prepare` once, then follow docs/annotation.md.
This command never substitutes model-generated labels for human judgments.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from sotu.process import ROOT
from sotu.topics import TOPICS
from sotu.provenance import fingerprint

REVIEW = ROOT / "data" / "validation" / "human_review.jsonl"
RESULTS = ROOT / "data" / "validation" / "human_review_results.json"
ERAS = [1789, 1860, 1912, 1945, 2030]
LABELS = {"topic": list(TOPICS) + ["Other"], "sentiment": ["negative", "neutral", "positive"],
          "emotion": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]}


def audit_hash(rows):
    fields = ["key", "task", "speech_id", "unit", "year", "sampling_stratum", "sampling_weight", "text"]
    return fingerprint(sorted([{k: r[k] for k in fields} for r in rows], key=lambda r: r["key"]))


def prepare():
    if REVIEW.exists():
        raise FileExistsError(f"Preserve existing annotations: {REVIEW}")
    rows = []
    for task, filename, n in [("topic", "paragraph_topics.parquet", 30), ("affect", "sentences.parquet", 50)]:
        frame = pd.read_parquet(ROOT / "data" / "derived" / filename)
        frame["era"] = pd.cut(frame.year, ERAS).astype(str)
        frame["length_group"] = np.where(frame.text.str.split().str.len() < 20, "short", "long") if task == "topic" else "all"
        for (era, length), group in frame.groupby(["era", "length_group"], observed=True):
            selected = group.sample(min(n, len(group)), random_state=23)
            for record in selected.itertuples():
                unit = int(record.paragraph if task == "topic" else record.sentence)
                fields = ["topic"] if task == "topic" else ["sentiment", "emotion"]
                rows.append({"key": f"{task}-{record.speech_id}-{unit}", "task": task,
                    "speech_id": record.speech_id, "unit": unit, "year": int(record.year),
                    "sampling_stratum": f"{era}:{length}", "sampling_weight": len(group) / len(selected),
                    "text": record.text, "reviewer_a": {f: None for f in fields},
                    "reviewer_b": {f: None for f in fields}, "adjudicated": {f: None for f in fields}})
    REVIEW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    REVIEW.with_name("human_review_sample_manifest.json").write_text(json.dumps({"seed": 23, "items": len(rows), "sample_sha256": audit_hash(rows)}, indent=2) + "\n")
    print(f"Prepared {len(rows)} blind items; independent human annotations pending.")


def held_out_topic_predictions(rows):
    """Audit topics on presidents absent from every local supervised source.

    Human audit labels are never passed to a training routine. Even an audit
    paragraph overlapping a benchmark sentence is scored out of sample.
    """
    from sklearn.model_selection import GroupKFold
    from sotu import models, validate_topics as validation
    from sotu.process import load_speeches
    from sotu.presidents import canonical_name

    audit = [r for r in rows if r["task"] == "topic"]
    president_of = {s["id"]: s["president"] for s in load_speeches()}
    groups = np.array([canonical_name(president_of[r["speech_id"]]) for r in audit])
    cap = validation.load_cap()
    Xcap = validation.cap_embeddings(cap)
    items = pd.read_json(validation.SAMPLE_PATH, lines=True)
    labels = pd.read_json(validation.LABELS_PATH, lines=True)
    history = items[items.source == "paragraph"].merge(labels, on="key")
    history["president"] = history.speech_id.map(president_of).map(canonical_name)
    Xhistory = models.embed(history.text.tolist())
    Xaudit = models.embed([r["text"] for r in audit])
    predictions = np.empty(len(audit), dtype=object)
    if len(np.unique(groups)) < 2:
        folds = [(None, np.arange(len(audit)))]
    else:
        folds = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(Xaudit, groups=groups)
    for _, test in folds:
        held = set(groups[test])
        cap_keep = validation.benchmark_training_mask(cap, held)
        hist_keep = ~history.president.isin(held).to_numpy()
        weight = validation.HISTORICAL_WEIGHT
        clf = validation._supervised(Xcap[cap_keep], cap.label.to_numpy()[cap_keep],
              np.repeat(Xhistory[hist_keep], weight, axis=0), np.repeat(history.topic.to_numpy()[hist_keep], weight))
        predictions[test] = clf.predict(Xaudit[test])
    return {(r["speech_id"], r["unit"]): prediction for r, prediction in zip(audit, predictions)}


def evaluate():
    rows = [json.loads(line) for line in REVIEW.read_text().splitlines()]
    sample_manifest = json.loads(REVIEW.with_name("human_review_sample_manifest.json").read_text())
    if not rows or audit_hash(rows) != sample_manifest["sample_sha256"]:
        raise ValueError("Audit inputs changed or items are missing; preserve the full blind sample")
    sentences = pd.read_parquet(ROOT / "data" / "derived" / "sentences.parquet")
    topic = None  # Audit topics must use held-out predictions, not production training fits.
    affect = {(r.speech_id, r.sentence): {"sentiment": r.tone_label, "emotion": r.emotion} for r in sentences.itertuples()} if "tone_label" in sentences else {}
    report = {"status": "pending", "items": len(rows), "tasks": {},
              "limits": "Only independent, adjudicated human labels are scored. Partial completion may be selected; this audit is not a population estimate until all strata are complete."}
    all_complete = True
    for field, allowed in LABELS.items():
        subset = [r for r in rows if field in r["adjudicated"]]
        complete = []
        for r in subset:
            values = [r[reviewer].get(field) for reviewer in ("reviewer_a", "reviewer_b", "adjudicated")]
            if any(v is not None and v not in allowed + ["ambiguous"] for v in values):
                raise ValueError(f"Invalid {field} label: {r['key']}")
            if all(v is not None for v in values):
                complete.append(r)
        scored = [r for r in complete if r["adjudicated"][field] != "ambiguous"]
        all_complete &= len(complete) == len(subset)
        result = {"total": len(subset), "adjudicated": len(complete), "scored": len(scored),
                  "ambiguous": len(complete) - len(scored), "status": "complete" if len(complete) == len(subset) else "pending"}
        if complete:
            result["reviewer_agreement"] = float(np.mean([r["reviewer_a"][field] == r["reviewer_b"][field] for r in complete]))
        if scored:
            if field == "topic" and topic is None:
                topic = held_out_topic_predictions(rows)
            if field != "topic" and not affect:
                raise ValueError("Rebuild with python -m sotu.build before scoring affect annotations")
            predictions = [topic[(r["speech_id"], r["unit"])] if field == "topic" else affect[(r["speech_id"], r["unit"])][field] for r in scored]
            result["weighted_metrics"] = classification_report([r["adjudicated"][field] for r in scored], predictions,
                labels=allowed, sample_weight=[r["sampling_weight"] for r in scored], output_dict=True, zero_division=0)
            result["by_stratum"] = {}
            for stratum in sorted({r["sampling_stratum"] for r in scored}):
                idx = [i for i, r in enumerate(scored) if r["sampling_stratum"] == stratum]
                result["by_stratum"][stratum] = classification_report(
                    [scored[i]["adjudicated"][field] for i in idx], [predictions[i] for i in idx],
                    labels=allowed, output_dict=True, zero_division=0)
            result["prediction_protocol"] = "president-held-out across both supervised sources" if field == "topic" else "pinned pretrained model; no audit-label training"
        report["tasks"][field] = result
    report["status"] = "complete" if all_complete else "pending"
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "evaluate"])
    {"prepare": prepare, "evaluate": evaluate}[parser.parse_args().step]()
