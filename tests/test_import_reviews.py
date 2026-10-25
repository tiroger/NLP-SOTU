import pytest
from sotu.import_reviews import merge_reviews


def test_merge_preserves_local_judgments_and_validates_before_writing():
    audit = [{"key": "one", "text": "Words"}, {"key": "two", "text": "Other"}]
    old = [{**r, "sentiment": "neutral" if r["key"] == "one" else None, "notes": ""} for r in audit]
    incoming = [{**r, "sentiment": "positive" if r["key"] == "two" else None, "notes": ""} for r in audit]
    merged = merge_reviews(old, incoming, audit)
    assert [r["sentiment"] for r in merged] == ["neutral", "positive"]
    assert old[1]["sentiment"] is None
    for invalid in [[incoming[0], incoming[0]], [{**incoming[0], "text": "changed"}, incoming[1]], [{**incoming[0], "sentiment": "negative"}, incoming[1]]]:
        with pytest.raises(ValueError):
            merge_reviews(old, invalid, audit)
