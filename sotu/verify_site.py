"""Validate committed publication artifacts using only the Python standard library."""

import hashlib
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify(root=ROOT):
    site = root / "site"
    read = lambda name: json.loads((site / "data" / name).read_text())
    manifest = read("manifest.json")
    expected_id = manifest.pop("build_id")
    actual_id = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert expected_id == actual_id, "Build manifest fingerprint mismatch"
    for name, digest in manifest["analysis_sources"].items():
        assert hashlib.sha256((root / "sotu" / name).read_bytes()).hexdigest() == digest, f"Stale build source: {name}"
    assert hashlib.sha256((root / "data" / "speeches.jsonl").read_bytes()).hexdigest() == manifest["corpus_sha256"], "Stale corpus"
    for name, digest in manifest.get("publication_sources", {}).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f"Stale publication source: {name}"
    for name, digest in manifest["topic_inputs"].items():
        if name == "US-Exec_SOTU_2025.csv":
            content = gzip.decompress((root / "data" / "reference" / (name + ".gz")).read_bytes())
        else:
            content = (root / "data" / "validation" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest, f"Stale topic input: {name}"
    sentiment = read("sentiment_validation.json")
    artifacts = manifest["sentiment_validation"]
    validation_dir = root / "data" / "validation" / "sentiment"
    for path, expected in [(validation_dir / "results.json", artifacts["results_sha256"]),
            (site / "data" / "sentiment_validation.json", artifacts["results_sha256"]),
            (validation_dir / "reliability.html", artifacts["reliability_sha256"]),
            (site / "data" / "sentiment_reliability.html", artifacts["reliability_sha256"]),
            (validation_dir / "report.md", artifacts["report_sha256"])]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Stale sentiment artifact: {path.name}"
    assert sentiment["implementation_sha256"] == hashlib.sha256((root / "sotu" / "sentiment_validation.py").read_bytes()).hexdigest()
    assert sentiment["model"]["revision"] == manifest["pipeline"]["models"][sentiment["model"]["name"]]
    reference = root / "data" / "reference" / "parlasent"
    assert hashlib.sha256((reference / "source.json").read_bytes()).hexdigest() == artifacts["reference_source_sha256"]
    for name, metadata in sentiment["reference"]["files"].items():
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == metadata["sha256"]
    metrics = sentiment["systems"]["sotu_roberta"]
    assert metrics["n"] == sum(sum(row) for row in metrics["confusion_matrix"]) == 2600
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["macro_f1"] <= 1
    assert hashlib.sha256((root / "data" / "validation" / "sentiment_audit_manifest.json").read_bytes()).hexdigest() == manifest["sentiment_audit"]["sample_manifest_sha256"]
    speeches, summary = read("speeches.json"), read("summary.json")
    assert summary["build_id"] == expected_id
    ids = {s["id"] for s in speeches}
    assert len(ids) == len(speeches) == summary["speeches"]
    assert sum(s["words"] for s in speeches) == summary["words"]
    for s in speeches:
        assert abs(sum(s[f"{label}_share"] for label in ["positive", "neutral", "negative"]) - 1) <= .0002
        assert all(sid in ids for sid in s["similar"] + s["kindred"])
        text = read(f"text/{s['id']}.json")
        assert sum(map(len, text)) == s["sentences"], f"Missing sentence text: {s['id']}"
    for row in read("topics_by_decade.json"):
        assert 0 <= row["ci_low"] <= row["share"] <= row["ci_high"] <= 1
        assert row["n_speeches"] >= 1
    comparisons = read("comparisons.json")
    assert len(comparisons["party"]) == 14
    for row in comparisons["party"]:
        assert row["assignments"] == math.comb(row["presidents_democratic"] + row["presidents_republican"], row["presidents_democratic"])
        assert 0 <= row["exact_p"] <= 1
    assert {row["era_sigma_years"] for row in comparisons["neighbor_party"]} == {5, 10, 20}

    def check(node):
        if isinstance(node, dict):
            url = node.get("url")
            if isinstance(url, str) and url.startswith("data/"):
                assert (site / url).is_file(), f"Missing chart data: {url}"
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)
        elif isinstance(node, float):
            assert math.isfinite(node), "Non-finite publication number"
    for path in (site / "charts").glob("*.json"):
        check(json.loads(path.read_text()))
    for path in (site / "data").glob("*.json"):
        check(json.loads(path.read_text()))
    print(f"Verified {len(speeches)} speeches, sentiment classes, topic intervals, comparison tests and build {expected_id[:12]}")


if __name__ == "__main__":
    verify()
