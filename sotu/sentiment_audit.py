"""Prepare and score a blind, president-separated SOTU sentiment audit.

prepare: preserve the earlier pilot; create 1,000 new audit slots and blind reader files.
ingest: copy reader labels into the master, accepting agreements and preserving disagreements.
evaluate: score only independently annotated, adjudicated items; blanks remain pending.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sotu.process import ROOT
from sotu.provenance import file_hash, fingerprint
from sotu.sentiment_validation import LABELS, evaluate_predictions

AUDIT_DIR = ROOT / "data" / "validation"
AUDIT = AUDIT_DIR / "sentiment_audit.jsonl"
MANIFEST = AUDIT_DIR / "sentiment_audit_manifest.json"
RESULTS = AUDIT_DIR / "sentiment_audit_results.json"
ERAS = [1789, 1860, 1912, 1945, 2030]
SEED = 73
ALLOWED = LABELS + ["ambiguous"]


def audit_hash(rows):
    return fingerprint([{k: v for k, v in r.items() if k not in ["reviewer_a", "reviewer_b", "adjudicated", "notes"]} for r in sorted(rows, key=lambda r: r["key"])])


def make_sample(frame, seed=SEED, development_per_era=50, test_per_era=200):
    """Group allocation precedes annotation; simple random sentences within era/split.

    Inclusion weights describe the selected president pools, not all presidents.
    No predicted sentiment is used to draw or partition this sample.
    """
    frame = frame.copy()
    frame["era"] = pd.cut(frame.year, ERAS).astype(str)
    rng = np.random.default_rng(seed)
    assignment = {}
    for era, group in frame.groupby("era", sort=True):
        presidents = sorted(set(group.president) - set(assignment))
        if presidents:
            n_dev = min(len(presidents) - 1, max(1, round(.2 * len(presidents))))
            development = set(rng.choice(presidents, n_dev, replace=False))
            assignment.update({p: "development" if p in development else "test" for p in presidents})
    frame["partition"] = frame.president.map(assignment)
    picks = []
    for era, group in frame.groupby("era", sort=True):
        for partition, n in [("development", development_per_era), ("test", test_per_era)]:
            pool = group[group.partition == partition]
            if len(pool) < n:
                raise ValueError(f"Insufficient independent {partition} pool in era {era}")
            sample = pool.sample(n, random_state=seed).copy()
            sample["sampling_weight"] = len(pool) / n
            sample["pool_sentences"] = len(pool)
            picks.append(sample)
    selected = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in enumerate(selected.itertuples()):
        rows.append({"key": f"sentiment-{i+1:04}", "speech_id": r.speech_id, "unit": int(r.sentence),
            "year": int(r.year), "era": r.era, "partition": r.partition, "president": r.president,
            "party": r.party, "delivery": r.delivery, "text": r.text,
            "sampling_weight": float(r.sampling_weight), "pool_sentences": int(r.pool_sentences),
            "reviewer_a": {"sentiment": None}, "reviewer_b": {"sentiment": None},
            "adjudicated": {"sentiment": None}, "notes": ""})
    return rows, assignment


def prepare():
    targets = [AUDIT, MANIFEST, RESULTS] + [AUDIT_DIR / f"sentiment_reader_{reader}.jsonl" for reader in ["a", "b"]]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preserve existing audit inputs and annotations")
    frame = pd.read_parquet(ROOT / "data" / "derived" / "sentences.parquet")
    meta = pd.read_csv(ROOT / "data" / "derived" / "speech_features.csv", usecols=["id", "delivery"])
    frame = frame.merge(meta, left_on="speech_id", right_on="id", validate="many_to_one")
    rows, assignment = make_sample(frame)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for reader in ["a", "b"]:
        path = AUDIT_DIR / f"sentiment_reader_{reader}.jsonl"
        if path.exists():
            raise FileExistsError(f"Preserve existing reader annotations: {path}")
        # Opaque IDs and shuffled order hide date, party, speaker, model output and partition.
        path.write_text("".join(json.dumps({"key": r["key"], "text": r["text"], "sentiment": None, "notes": ""}, ensure_ascii=False) + "\n" for r in rows))
    MANIFEST.write_text(json.dumps({"schema": 1, "seed": SEED, "items": len(rows), "sample_sha256": audit_hash(rows),
        "corpus_sha256": file_hash(ROOT / "data" / "speeches.jsonl"),
        "sentences_sha256_at_sampling": file_hash(ROOT / "data" / "derived" / "sentences.parquet"),
        "implementation_sha256": file_hash(Path(__file__)), "president_partitions": assignment,
        "design": "50 development and 200 final-test sentences per era, disjoint presidents; conditional weights within era/partition president pools",
        "pilot_policy": "Earlier 200-sentence pilot retained unchanged; do not use it for model selection because it overlaps final-test presidents.",
    }, indent=2) + "\n")
    print(f"Prepared {len(rows)} blank human audit items; 200 development and 800 final-test sentences.")


def load_audit():
    rows = [json.loads(line) for line in AUDIT.read_text().splitlines()]
    manifest = json.loads(MANIFEST.read_text())
    if not rows or audit_hash(rows) != manifest["sample_sha256"]:
        raise ValueError("Audit inputs changed or items are missing")
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate audit keys")
    dev = {r["president"] for r in rows if r["partition"] == "development"}
    test = {r["president"] for r in rows if r["partition"] == "test"}
    if dev & test:
        raise ValueError("Development and test presidents overlap")
    for r in rows:
        for reviewer in ["reviewer_a", "reviewer_b", "adjudicated"]:
            if r[reviewer]["sentiment"] not in [None] + ALLOWED:
                raise ValueError(f"Invalid sentiment label in {r['key']}")
    return rows, manifest


def ingest():
    rows, _ = load_audit()
    by_key = {r["key"]: r for r in rows}
    for reader in ["a", "b"]:
        labels = [json.loads(line) for line in (AUDIT_DIR / f"sentiment_reader_{reader}.jsonl").read_text().splitlines()]
        if len(labels) != len(rows) or {r["key"] for r in labels} != set(by_key):
            raise ValueError("Reader file must preserve all unique audit items")
        for label in labels:
            r = by_key[label["key"]]
            if label["text"] != r["text"] or label["sentiment"] not in [None] + ALLOWED:
                raise ValueError(f"Changed text or invalid reader label: {r['key']}")
            old = r[f"reviewer_{reader}"]["sentiment"]
            if old is not None and old != label["sentiment"]:
                raise ValueError("Refusing to overwrite an existing reader judgment")
            r[f"reviewer_{reader}"]["sentiment"] = label["sentiment"]
    for r in rows:
        a, b = r["reviewer_a"]["sentiment"], r["reviewer_b"]["sentiment"]
        if a is not None and a == b and r["adjudicated"]["sentiment"] is None:
            r["adjudicated"]["sentiment"] = a
    AUDIT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("Imported reader judgments; agreements accepted, disagreements remain for human adjudication.")


def evaluate(release_test=False):
    rows, manifest = load_audit()
    if release_test and any(any(r[k]["sentiment"] is None for k in ["reviewer_a", "reviewer_b", "adjudicated"]) for r in rows):
        raise ValueError("Complete and adjudicate all reviews before releasing final-test metrics")
    sentences = pd.read_parquet(ROOT / "data" / "derived" / "sentences.parquet").set_index(["speech_id", "sentence"])
    report = {"status": "pending", "sample_sha256": manifest["sample_sha256"],
              "annotations_sha256": file_hash(AUDIT), "partitions": {},
              "limits": "Only independent human labels are scored. Partial completion can be selected. Final-test metrics describe held-out president pools, not all historical presidents. Human class codes measure categorical valence, not psychological intensity."}
    for partition in ["development", "test"]:
        subset = [r for r in rows if r["partition"] == partition]
        complete = [r for r in subset if all(r[k]["sentiment"] is not None for k in ["reviewer_a", "reviewer_b", "adjudicated"])]
        scored = [r for r in complete if r["adjudicated"]["sentiment"] != "ambiguous"]
        result = {"total": len(subset), "adjudicated": len(complete), "scored": len(scored),
                  "ambiguous": len(complete) - len(scored), "status": "complete" if len(complete) == len(subset) else "pending"}
        if complete:
            result["reader_agreement"] = float(np.mean([r["reviewer_a"] == r["reviewer_b"] for r in complete]))
        result["test_metrics_withheld"] = partition == "test" and not release_test
        if scored and not result["test_metrics_withheld"]:
            predictions = []
            for r in scored:
                source = sentences.loc[(r["speech_id"], r["unit"])]
                if source.text != r["text"]:
                    raise ValueError("Audit sentence text changed; align corpus before evaluation")
                predictions.append([source[f"tone_p_{label}"] for label in LABELS])
            probs = np.asarray(predictions)
            predicted = np.array(LABELS)[probs.argmax(axis=1)]
            gold = [r["adjudicated"]["sentiment"] for r in scored]
            weights = [r["sampling_weight"] for r in scored]
            result["metrics"] = evaluate_predictions(gold, predicted, probs=probs, weights=weights)
            result["by_era"] = {}
            for era in sorted({r["era"] for r in scored}):
                idx = [i for i, r in enumerate(scored) if r["era"] == era]
                result["by_era"][era] = evaluate_predictions([gold[i] for i in idx], predicted[idx], probs=probs[idx], weights=[weights[i] for i in idx])
        result["coverage"] = {"presidents": len({r["president"] for r in subset}), "speeches": len({r["speech_id"] for r in subset}),
            "by_era": {e: sum(r["era"] == e for r in subset) for e in sorted({r["era"] for r in subset})},
            "by_delivery": {d: sum(r["delivery"] == d for r in subset) for d in sorted({r["delivery"] for r in subset})}}
        report["partitions"][partition] = result
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["partitions"].values()) else "pending"
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "partitions": {k: {m: r[m] for m in ["total", "adjudicated", "scored"]} for k, r in report["partitions"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "ingest", "evaluate"])
    parser.add_argument("--release-test", action="store_true", help="Release final-test metrics only after all human judgments are complete")
    args = parser.parse_args()
    if args.step == "evaluate":
        evaluate(release_test=args.release_test)
    else:
        if args.release_test:
            parser.error("--release-test applies only to evaluate")
        {"prepare": prepare, "ingest": ingest}[args.step]()
