"""Validate the topic classifier, including before 1946 where no human codes exist.

The Policy Agendas Project (comparativeagendas.net) hand-coded every State of
the Union statement from 1946 on. Nothing comparable exists for 1790-1945, so
Claude labels a stratified sample of paragraphs under a written rubric. Claude
also labels a sample of the human-coded sentences, which measures how far its
labels can be trusted before they are used to judge anything else.

Usage:
    uv run python -m sotu.validate_topics sample     # draw the validation sample
    uv run python -m sotu.validate_topics submit     # send it to Claude (Message Batches, ~$5)
    uv run python -m sotu.validate_topics collect    # download labels once the batch has ended
    uv run python -m sotu.validate_topics label      # or: label directly, in parallel (~$9, minutes)
    uv run python -m sotu.validate_topics evaluate   # compare classifiers era by era

`evaluate` also reports the benchmark-only results without any Claude labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sotu import models, topics
from sotu.process import ROOT

VALIDATION_DIR = ROOT / "data" / "validation"
SAMPLE_PATH = VALIDATION_DIR / "sample.jsonl"
LABELS_PATH = VALIDATION_DIR / "claude_labels.jsonl"
BATCH_PATH = VALIDATION_DIR / "batch_id.txt"
REPORT_PATH = VALIDATION_DIR / "report.md"
CAP_URL = "https://minio.la.utexas.edu/compagendas/datasetfiles/US-Exec_SOTU_2025.csv"
CAP_PATH = ROOT / ".cache" / "external" / "US-Exec_SOTU_2025.csv"
CAP_EMB_PATH = ROOT / ".cache" / "external" / "cap_embeddings.npy"

MODEL = "claude-opus-5-5"
ERAS = [(1790, 1860), (1861, 1912), (1913, 1945), (1946, 2030)]
PER_ERA = {(1790, 1860): 200, (1861, 1912): 200, (1913, 1945): 200, (1946, 2030): 150}
CAP_SAMPLE = 300
LABELS = list(topics.TOPICS) + ["Other"]

# --- Policy Agendas Project codes -> this project's taxonomy ----------------
CAP_MAJOR = {
    2: "Civil rights", 3: "Health care", 4: "Agriculture", 5: "Economy & jobs", 6: "Education",
    7: "Energy & environment", 8: "Energy & environment", 9: "Immigration", 10: "Infrastructure",
    12: "Crime & justice", 13: "Social welfare", 14: "Social welfare", 16: "Defense & war",
    17: "Science & technology", 18: "Trade & tariffs", 19: "Foreign affairs", 20: "Government & administration",
}


def cap_to_topic(major: int, sub: int) -> str:
    if major == 1:  # macroeconomics: split out budget/taxes and monetary policy
        return {104: "Banking & currency", 105: "Taxes & budget", 107: "Taxes & budget"}.get(sub, "Economy & jobs")
    if major == 15:  # domestic commerce: banking and securities vs the rest
        return "Banking & currency" if sub in (1501, 1502) else "Economy & jobs"
    if major == 21:  # public lands also holds Native American affairs and territories
        return {2102: "Native American affairs", 2105: "Territories & expansion"}.get(sub, "Energy & environment")
    return CAP_MAJOR.get(major, "Other")  # 0 / -555 = no policy content; 23 = culture


def load_cap() -> pd.DataFrame:
    if not CAP_PATH.exists():
        CAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CAP_URL, timeout=120)
        r.raise_for_status()
        CAP_PATH.write_bytes(r.content)
    cap = pd.read_csv(CAP_PATH, encoding_errors="replace").dropna(subset=["description"]).reset_index(drop=True)
    cap["label"] = [cap_to_topic(int(m), int(s)) for m, s in zip(cap.majortopic, cap.subtopic)]
    return cap


def cap_embeddings(cap: pd.DataFrame) -> np.ndarray:
    if CAP_EMB_PATH.exists():
        emb = np.load(CAP_EMB_PATH)
        if len(emb) == len(cap):
            return emb.astype(np.float32)
    emb = models.embed(cap.description.tolist())
    np.save(CAP_EMB_PATH, emb.astype(np.float16))
    return emb.astype(np.float32)


# --- Sampling ------------------------------------------------------------------

def sample() -> None:
    paras = pd.read_parquet(ROOT / "data" / "derived" / "paragraph_topics.parquet")
    paras = paras[paras.text.str.split().str.len() >= 20]
    rows = []
    for era in ERAS:
        pool = paras[paras.year.between(*era)]
        pick = pool.sample(PER_ERA[era], random_state=7)
        rows += [{"key": f"p-{r.speech_id}-{r.paragraph}", "source": "paragraph", "speech_id": r.speech_id,
                  "year": int(r.year), "era": f"{era[0]}-{min(era[1], 2026)}", "text": r.text} for r in pick.itertuples()]
    cap = load_cap()
    pick = cap.sample(CAP_SAMPLE, random_state=7)
    rows += [{"key": f"c-{int(r.id)}", "source": "benchmark", "speech_id": str(r.date), "year": int(r.year),
              "era": "benchmark", "text": r.description, "human_label": r.label} for r in pick.itertuples()]
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {SAMPLE_PATH.relative_to(ROOT)}")


# --- Labeling with Claude --------------------------------------------------------

def rubric() -> str:
    topic_lines = "\n".join(f"- {name}: " + " ".join(desc) for name, desc in topics.TOPICS.items())
    return f"""You label passages from U.S. presidents' annual messages to Congress (State of the Union), 1790 to today, by policy topic.

Choose exactly one topic: the policy area the passage is mainly about. Topics, with descriptions:
{topic_lines}
- Other: no policy content. Greetings, thanks, applause lines, ceremonial or patriotic passages, transitions, personal stories with no policy point, and general statements about the nation.

Conventions (these match the Policy Agendas Project codebook, so labels are comparable to its human codes):
- Labor, wages, unemployment, and general business conditions are Economy & jobs. Inflation is Economy & jobs.
- Budgets, deficits, debt, appropriations, Treasury receipts, and taxes are Taxes & budget. Tariffs as trade policy are Trade & tariffs.
- Currency, coinage, gold and silver, banks, the Federal Reserve, interest rates, and securities are Banking & currency.
- Housing and poverty programs, pensions, and Social Security are Social welfare. Veterans' benefits and the armed services are Defense & war.
- Public lands, conservation, forests, water, pollution, and energy are Energy & environment, except land policy toward Indian tribes (Native American affairs) and acquiring or governing territories (Territories & expansion).
- Diplomacy, treaties, claims against foreign governments, and foreign aid are Foreign affairs unless the passage is mainly about war or military forces.
- Slavery, emancipation, voting rights, and discrimination are Civil rights.
- Elections, the Constitution, Congress's powers, the civil service, the Post Office, and executive departments are Government & administration.

Judge the passage by its subject, not by its tone, and use 19th-century meanings for 19th-century text. If two topics apply, choose the one the passage spends more words on."""


SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "enum": LABELS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["topic", "confidence"],
    "additionalProperties": False,
}


def _request_params(item: dict) -> dict:
    return dict(
        model=MODEL,
        max_tokens=2048,
        system=[{"type": "text", "text": rubric(), "cache_control": {"type": "ephemeral"}}],
        # Classification: low effort holds quality at a fraction of the cost.
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Passage ({item['year']}):\n\n{item['text']}"}],
    )


def _write_labels(labels: list[dict], problems: dict) -> None:
    with LABELS_PATH.open("w") as f:
        for row in sorted(labels, key=lambda r: r["key"]):
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(labels)} labels to {LABELS_PATH.relative_to(ROOT)}; not labeled: {len(problems)} {problems or ''}")


def label(concurrency: int = 16) -> None:
    """Label the sample with direct (non-batch) requests: full price, but done in minutes."""
    import asyncio

    import anthropic

    items = [json.loads(line) for line in SAMPLE_PATH.open()]
    client = anthropic.AsyncAnthropic(max_retries=6)
    gate = asyncio.Semaphore(concurrency)
    labels, problems = [], {}

    async def one(item):
        async with gate:
            try:
                msg = await client.messages.create(**_request_params(item))
            except anthropic.APIStatusError as e:
                problems[item["key"]] = f"HTTP {e.status_code}"
                return
            except anthropic.APIConnectionError:
                problems[item["key"]] = "connection error"
                return
        if msg.stop_reason == "refusal":
            problems[item["key"]] = "refusal"
            return
        text = next(b.text for b in msg.content if b.type == "text")
        labels.append({"key": item["key"], **json.loads(text)})
        if len(labels) % 100 == 0:
            print(f"  {len(labels)}/{len(items)} labeled")

    async def run():
        await asyncio.gather(*(one(i) for i in items))

    asyncio.run(run())
    _write_labels(labels, problems)


def submit() -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    items = [json.loads(line) for line in SAMPLE_PATH.open()]
    reqs = [Request(custom_id=item["key"], params=MessageCreateParamsNonStreaming(**_request_params(item))) for item in items]
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    BATCH_PATH.write_text(batch.id)
    print(f"Submitted batch {batch.id} with {len(reqs)} requests. Run `collect` once it has ended (usually < 1 hour).")


def collect() -> None:
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(BATCH_PATH.read_text().strip())
    if batch.processing_status != "ended":
        print(f"Batch {batch.id} is {batch.processing_status}: {batch.request_counts}")
        return
    labels, problems = [], {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            problems[result.custom_id] = result.result.type
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            problems[result.custom_id] = "refusal"
            continue
        text = next(b.text for b in msg.content if b.type == "text")
        labels.append({"key": result.custom_id, **json.loads(text)})
    _write_labels(labels, problems)


# --- The classifier the build uses ----------------------------------------------

CLASSIFIER_PATH = ROOT / ".cache" / "topic_classifier.joblib"
HISTORICAL_WEIGHT = 5  # each Claude-labeled paragraph counts like 5 benchmark sentences


def topic_classifier():
    """Return ``classify(embeddings) -> (labels, confidence)`` for paragraph embeddings.

    With Claude labels present: logistic regression on the benchmark's human
    codes plus the Claude-labeled historical paragraphs (the "hybrid" that won
    every era in `evaluate`). Without them: the description-centroid method.
    """
    if not LABELS_PATH.exists():
        names, cents = topics.centroids(models.embed)
        return lambda emb: topics.assign(emb, names, cents)

    import hashlib

    import joblib

    key = hashlib.sha1(b"".join([
        LABELS_PATH.read_bytes(), SAMPLE_PATH.read_bytes(), json.dumps(topics.TOPICS).encode(),
        str(HISTORICAL_WEIGHT).encode(), CAP_URL.encode(),
    ])).hexdigest()
    if CLASSIFIER_PATH.exists():
        cached = joblib.load(CLASSIFIER_PATH)
        if cached["key"] == key:
            clf = cached["model"]
            return lambda emb: _predict(clf, emb)

    cap = load_cap()
    items = pd.DataFrame([json.loads(line) for line in SAMPLE_PATH.open()])
    labels = pd.DataFrame([json.loads(line) for line in LABELS_PATH.open()])
    hist = items[items.source == "paragraph"].merge(labels, on="key")
    Xh = models.embed(hist.text.tolist()).astype(np.float32)
    clf = _supervised(cap_embeddings(cap), cap.label.to_numpy(),
                      np.repeat(Xh, HISTORICAL_WEIGHT, axis=0), np.repeat(hist.topic.to_numpy(), HISTORICAL_WEIGHT))
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "model": clf}, CLASSIFIER_PATH)
    return lambda emb: _predict(clf, emb)


def _predict(clf, emb):
    proba = clf.predict_proba(np.asarray(emb, dtype=np.float32))
    return list(clf.classes_[proba.argmax(axis=1)]), proba.max(axis=1)


# --- Evaluation -------------------------------------------------------------------

def _scores(gold, pred) -> dict:
    from sklearn.metrics import f1_score

    gold, pred = np.asarray(gold), np.asarray(pred)
    return {"n": len(gold), "accuracy": float(np.mean(gold == pred)),
            "macro_F1": float(f1_score(gold, pred, average="macro", zero_division=0))}


def _supervised(X, y, extra_X=None, extra_y=None, C=2.0):
    from sklearn.linear_model import LogisticRegression

    desc = [(t, d) for t, ds in topics.TOPICS.items() for d in ds]
    Xd = models.embed([d for _, d in desc]).astype(np.float32)
    yd = np.array([t for t, _ in desc])
    parts_X, parts_y = [X] + [Xd] * 20, [y] + [yd] * 20  # descriptions as anchors for rare historical topics
    if extra_X is not None:
        parts_X.append(extra_X)
        parts_y.append(extra_y)
    return LogisticRegression(C=C, max_iter=3000).fit(np.vstack(parts_X), np.concatenate(parts_y))


def evaluate() -> None:
    from sklearn.model_selection import GroupKFold

    cap = load_cap()
    X_cap = cap_embeddings(cap)
    names, cents = topics.centroids(models.embed)
    lines = ["# Topic classifier validation", ""]

    # 1. Benchmark (1946-2025), leave-president-out.
    centroid_pred, _ = topics.assign(X_cap, names, cents)
    oof = np.empty(len(cap), dtype=object)
    for tr, te in GroupKFold(n_splits=6).split(X_cap, cap.label, cap.president):
        oof[te] = _supervised(X_cap[tr], cap.label.to_numpy()[tr]).predict(X_cap[te])
    lines += ["## Policy Agendas Project benchmark, 1946-2025 (sentence level, leave-president-out)", "",
              "| classifier | n | accuracy | macro-F1 |", "|---|---|---|---|"]
    for name, pred in [("centroid (original method)", centroid_pred), ("supervised (benchmark-trained)", oof)]:
        s = _scores(cap.label, pred)
        lines.append(f"| {name} | {s['n']:,} | {s['accuracy']:.3f} | {s['macro_F1']:.3f} |")

    if not LABELS_PATH.exists():
        lines += ["", "_No Claude labels yet: run `submit` and `collect` to validate the pre-1946 era._"]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    items = pd.DataFrame([json.loads(line) for line in SAMPLE_PATH.open()])
    labels = pd.DataFrame([json.loads(line) for line in LABELS_PATH.open()])
    df = items.merge(labels, on="key")

    # 2. How far do Claude's labels agree with human coders, and with themselves?
    bench = df[df.source == "benchmark"]
    s = _scores(bench.human_label, bench.topic)
    lines += ["", "## Claude labels vs. human coders (benchmark sentences)", "",
              f"Agreement {s['accuracy']:.1%} on {s['n']} sentences (macro-F1 {s['macro_F1']:.3f}), by Claude's stated confidence:", ""]
    for conf in ("high", "medium", "low"):
        g = bench[bench.confidence == conf]
        if len(g):
            lines.append(f"- {conf}: {np.mean(g.topic == g.human_label):.0%} of {len(g)}")
    rerun = VALIDATION_DIR / "claude_labels_batch_run.jsonl"
    if rerun.exists():
        other = pd.DataFrame([json.loads(line) for line in rerun.open()]).merge(df[["key", "topic"]], on="key", suffixes=("", "_direct"))
        lines += ["", f"Run-to-run consistency (two independent labeling runs): {np.mean(other.topic == other.topic_direct):.1%} "
                      f"of {len(other)} items get the same label."]
    lines.append("")

    # 3. Paragraph classifiers vs. Claude labels, era by era.
    paras = df[df.source == "paragraph"].reset_index(drop=True)
    Xp = models.embed(paras.text.tolist()).astype(np.float32)
    yp = paras.topic.to_numpy()
    cen = np.array(topics.assign(Xp, names, cents)[0])
    sup = _supervised(X_cap, cap.label.to_numpy()).predict(Xp)
    # Benchmark + Claude-labeled historical paragraphs, cross-validated by speech.
    hyb = np.empty(len(paras), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(Xp, yp, paras.speech_id):
        hyb[te] = _supervised(X_cap, cap.label.to_numpy(), np.repeat(Xp[tr], 5, axis=0), np.repeat(yp[tr], 5)).predict(Xp[te])
    lines += ["## Paragraph classifiers vs. Claude labels, by era", "",
              "| era | n | centroid (original method) | benchmark-trained | benchmark + historical labels (used by the site) |", "|---|---|---|---|---|"]
    best = {}
    for era, g in paras.groupby("era", sort=True):
        idx = g.index.to_numpy()
        row = {k: _scores(yp[idx], p[idx]) for k, p in [("centroid", cen), ("supervised", sup), ("hybrid", hyb)]}
        best[era] = max(row, key=lambda k: row[k]["macro_F1"])
        cells = " | ".join(f"acc {row[k]['accuracy']:.2f} / F1 {row[k]['macro_F1']:.2f}" for k in ("centroid", "supervised", "hybrid"))
        lines.append(f"| {era} | {len(idx)} | {cells} |")
    lines += ["", "Best classifier by macro-F1 per era: " + ", ".join(f"{e}: **{b}**" for e, b in best.items()), ""]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY; .env is git-ignored
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["sample", "submit", "collect", "label", "evaluate"])
    {"sample": sample, "submit": submit, "collect": collect, "label": label, "evaluate": evaluate}[parser.parse_args().step]()
