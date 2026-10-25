import json

import numpy as np
import pandas as pd
import pytest

from sotu import sentiment_audit as audit
from sotu.sentiment_validation import cluster_intervals, evaluate_predictions, load_reference, reliability


def test_probability_and_score_metrics_have_known_values():
    p = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    labels = ["negative", "neutral", "positive"]
    r = evaluate_predictions(labels, labels, probs=p)
    assert r["accuracy"] == r["macro_f1"] == 1
    assert r["brier_multiclass_sum"] == pytest.approx(.06)
    assert r["score_mae_vs_class_code"] == pytest.approx(.2)
    assert r["score_bias_vs_class_code"] == pytest.approx(0)
    assert r["log_loss"] == pytest.approx(-np.log(.8))
    assert r["classwise_ece_10_bins"]["positive"] == pytest.approx(2 / 15)


def test_weights_change_bias_and_accuracy_without_changing_label_alignment():
    r = evaluate_predictions(["negative", "positive"], ["neutral", "positive"],
                             scores=[0., 1.], weights=[3., 1.])
    assert r["accuracy"] == .25
    assert r["score_bias_vs_class_code"] == .75
    assert r["human_mean_class_code"] == -.5
    assert r["model_mean_score"] == .25


def test_reliability_handles_probability_one_and_does_not_invent_empty_bins():
    p = np.eye(3)
    rows = reliability(np.arange(3), p)
    assert len(rows) == 6
    assert all(row["mean_probability"] == row["observed_frequency"] for row in rows)
    assert sum(row["n"] for row in rows if row["class"] == "neutral") == 3
    assert any(row["bin_high"] == 1 for row in rows)


@pytest.mark.parametrize("p", [np.array([[.4, .4, .4]]), np.array([[np.nan, 0, 1]]), np.array([[-.1, .1, 1.]])])
def test_invalid_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        evaluate_predictions(["positive"], ["positive"], probs=p)


def test_source_document_bootstrap_keeps_perfect_groups_together():
    gold = ["negative", "neutral", "positive"] * 5
    p = np.tile(np.eye(3), (5, 1))
    r = cluster_intervals(gold, p, np.repeat(np.arange(5), 3), draws=100)
    assert r["groups"] == 5
    assert r["percentile_95"]["accuracy"] == [1., 1.]
    assert r["percentile_95"]["macro_f1"] == [1., 1.]
    assert r["percentile_95"]["score_bias_vs_class_code"] == [0., 0.]


def test_pinned_reference_keeps_official_labels_and_test_size():
    source, train, test = load_reference()
    assert len(test) == 2600 and len(train) == 2600
    assert test.label.value_counts().to_dict() == {"neutral": 1062, "negative": 841, "positive": 697}
    assert source["revision"] == "0587c2b6499fbc68a7623439c2af2b24748968dc"


def sample_frame():
    return pd.DataFrame([{"speech_id": f"speech-{era}-{pres}", "sentence": sentence,
                         "year": year, "president": f"president-{era}-{pres}", "party": "P", "delivery": "spoken",
                         "text": f"Text {era} {pres} {sentence}."}
                        for era, year in enumerate([1800, 1880, 1920, 2000]) for pres in range(5) for sentence in range(4)])


def test_audit_has_disjoint_presidents_and_known_sampling_weights():
    rows, assignment = audit.make_sample(sample_frame(), development_per_era=2, test_per_era=4)
    again, _ = audit.make_sample(sample_frame(), development_per_era=2, test_per_era=4)
    assert rows == again
    assert len(rows) == 24
    dev = {r["president"] for r in rows if r["partition"] == "development"}
    test = {r["president"] for r in rows if r["partition"] == "test"}
    assert not dev & test
    assert len(dev) == 4
    assert all(r["sampling_weight"] == (2 if r["partition"] == "development" else 4) for r in rows)
    assert all(r["reviewer_a"]["sentiment"] is None for r in rows)
    assert all(r["key"].startswith("sentiment-") for r in rows)


def setup_audit(monkeypatch, tmp_path):
    rows, _ = audit.make_sample(sample_frame(), development_per_era=2, test_per_era=4)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path)
    monkeypatch.setattr(audit, "AUDIT", tmp_path / "sentiment_audit.jsonl")
    monkeypatch.setattr(audit, "MANIFEST", tmp_path / "sentiment_audit_manifest.json")
    monkeypatch.setattr(audit, "RESULTS", tmp_path / "sentiment_audit_results.json")
    audit.AUDIT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    audit.MANIFEST.write_text(json.dumps({"sample_sha256": audit.audit_hash(rows)}))
    source = pd.DataFrame([{"speech_id": r["speech_id"], "sentence": r["unit"], "text": r["text"],
                            "tone_p_negative": .1, "tone_p_neutral": .8, "tone_p_positive": .1} for r in rows])
    path = tmp_path / "data" / "derived" / "sentences.parquet"
    path.parent.mkdir(parents=True)
    source.to_parquet(path)
    return rows


def test_blank_human_audit_remains_pending(monkeypatch, tmp_path):
    setup_audit(monkeypatch, tmp_path)
    audit.evaluate()
    r = json.loads(audit.RESULTS.read_text())
    assert r["status"] == "pending"
    assert r["partitions"]["test"]["scored"] == 0
    assert "metrics" not in r["partitions"]["test"]


def test_reader_disagreements_need_adjudication_and_agreements_are_scored(monkeypatch, tmp_path):
    rows = setup_audit(monkeypatch, tmp_path)
    for reader in ["a", "b"]:
        labels = [{"key": r["key"], "text": r["text"], "sentiment": "neutral"} for r in rows]
        if reader == "b":
            labels[0]["sentiment"] = "negative"
        (tmp_path / f"sentiment_reader_{reader}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    audit.ingest()
    updated, _ = audit.load_audit()
    assert updated[0]["adjudicated"]["sentiment"] is None
    assert updated[0]["reviewer_a"]["sentiment"] == "neutral"
    assert updated[0]["reviewer_b"]["sentiment"] == "negative"
    audit.evaluate()
    r = json.loads(audit.RESULTS.read_text())
    assert r["status"] == "pending"
    assert sum(p["scored"] for p in r["partitions"].values()) == len(rows) - 1
    assert r["partitions"]["development"]["metrics"]["accuracy"] == 1
    assert r["partitions"]["test"]["test_metrics_withheld"] is True
    assert "metrics" not in r["partitions"]["test"]
    with pytest.raises(ValueError, match="adjudicate"):
        audit.evaluate(release_test=True)
    updated[0]["adjudicated"]["sentiment"] = "neutral"
    audit.AUDIT.write_text("".join(json.dumps(row) + "\n" for row in updated))
    audit.evaluate(release_test=True)
    released = json.loads(audit.RESULTS.read_text())
    assert released["status"] == "complete"
    assert released["partitions"]["test"]["metrics"]["accuracy"] == 1
