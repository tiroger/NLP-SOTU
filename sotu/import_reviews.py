"""Merge an owner-downloaded web annotation export without erasing local judgments."""
import argparse
import json
from pathlib import Path

from sotu.sentiment_audit import ALLOWED, AUDIT_DIR, load_audit


def merge_reviews(existing, incoming, audit):
    expected = {r["key"]: r["text"] for r in audit}
    def validated(rows):
        if len(rows) != len(expected) or {r.get("key") for r in rows} != set(expected):
            raise ValueError("Export must contain every audit item exactly once")
        for r in rows:
            if r.get("text") != expected[r["key"]] or r.get("sentiment") not in [None] + ALLOWED or not isinstance(r.get("notes"), str) or len(r["notes"]) > 2000:
                raise ValueError(f"Changed text, invalid label or notes: {r['key']}")
        return {r["key"]: r for r in rows}
    local, remote = validated(existing), validated(incoming)
    result = []
    for key in expected:
        old, new = local[key], remote[key]
        if old["sentiment"] is not None and new["sentiment"] is not None and old["sentiment"] != new["sentiment"]:
            raise ValueError(f"Conflicting existing judgment: {key}. Resolve it manually before importing")
        if old["notes"] and new["notes"] and old["notes"] != new["notes"]:
            raise ValueError(f"Conflicting notes: {key}. Resolve them manually before importing")
        result.append({"key": key, "text": expected[key], "sentiment": new["sentiment"] or old["sentiment"], "notes": new["notes"] or old["notes"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reader", choices=["a", "b"])
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    # Download filename is the round identifier; cross-round imports would invalidate independence.
    if args.export.name != f"sentiment_reader_{args.reader}.jsonl":
        parser.error("Use the unrenamed dashboard export for this reader round")
    rows, _ = load_audit()
    target = AUDIT_DIR / f"sentiment_reader_{args.reader}.jsonl"
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    merged = merge_reviews(read(target), read(args.export), rows)
    pending = target.with_suffix(".jsonl.tmp")
    pending.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    pending.replace(target)
    print(f"Imported reader {args.reader.upper()}: {sum(r['sentiment'] is not None for r in merged)}/{len(merged)} labeled. Run sentiment_audit ingest next.")


if __name__ == "__main__":
    main()
