from collections import Counter

from sotu.build import log_odds_z, tone_arc
from sotu.scrape import parse_index
from sotu.text import clean_paragraphs, mtld

INDEX_HTML = """
<table><thead><tr><td><a name="Table of SOTU"></a></td></tr></thead><tbody>
<tr><td><strong>Barack Obama</strong></td><td>2013-2017</td>
  <td><a href="https://x/2013">2013</a></td><td></td><td></td><td></td><td></td>
  <td></td><td></td><td></td><td></td><td></td></tr>
<tr><td> </td><td>2009-2013</td>
  <td><a href="https://x/2009">2009</a>*</td><td></td><td></td><td></td><td></td>
  <td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><strong>William Howard Taft</strong></td><td>1909-1913</td>
  <td></td><td></td><td></td><td></td><td></td>
  <td><a href="https://x/1911">1911</a>**</td><td><a href="https://x/fn">‡</a></td><td></td><td></td><td></td></tr>
</tbody></table>
"""


def test_parse_index_carries_names_and_reads_markers():
    entries = {e["index_year"]: e for e in parse_index(INDEX_HTML)}
    assert set(entries) == {2013, 2009, 1911}  # footnote link ignored
    assert entries[2009]["president"] == "Barack Obama"  # blank name cell inherits
    assert entries[2009]["formal_sotu"] is False  # single asterisk
    assert entries[1911]["formal_sotu"] is True  # double asterisk is a footnote
    assert entries[1911]["delivery"] == "written"
    assert entries[2013]["delivery"] == "spoken"


def test_clean_paragraphs_removes_non_presidential_text():
    raw = [
        "The President. Thank you. [Applause] Madam Speaker,",
        "Audience members. U.S.A.! U.S.A.!",
        "Rep. Greene. Liar!",
        "NOTE: The President spoke at 9:07 p.m.",
        "DWIGHT D. EISENHOWER",
        "We will (Laughter) keep going.",
    ]
    assert clean_paragraphs(raw) == ["Thank you. Madam Speaker,", "We will keep going."]


def test_mtld_is_length_robust():
    words = "the quick brown fox jumps over the lazy dog while a cat sleeps nearby".split()
    short, long = mtld(words * 3), mtld(words * 30)
    assert abs(short - long) / long < 0.25
    assert mtld(["same"] * 200) < 5


def test_log_odds_sign():
    a = Counter({"tax": 50, "war": 5, "the": 500})
    b = Counter({"tax": 5, "war": 50, "the": 500})
    z = log_odds_z(a, b, a + b)
    assert z["tax"] > 3 and z["war"] < -3 and abs(z["the"]) < 1


def test_tone_arc_shape():
    import numpy as np

    arc = tone_arc(np.linspace(-1, 1, 500))
    assert len(arc) == 40 and arc[0] < 0 < arc[-1]


def test_cap_mapping_splits_macro_and_public_lands():
    from sotu.validate_topics import LABELS, cap_to_topic

    assert cap_to_topic(1, 105) == "Taxes & budget"
    assert cap_to_topic(1, 104) == "Banking & currency"
    assert cap_to_topic(1, 103) == "Economy & jobs"
    assert cap_to_topic(21, 2102) == "Native American affairs"
    assert cap_to_topic(21, 2105) == "Territories & expansion"
    assert cap_to_topic(21, 2101) == "Energy & environment"
    assert cap_to_topic(-555, -555) == "Other"
    assert all(cap_to_topic(m, m * 100) in LABELS for m in range(0, 24))


def test_era_adjustment_removes_shared_time_trend():
    import numpy as np

    from sotu.build import era_adjusted

    rng = np.random.default_rng(0)
    when = np.linspace(1790, 2026, 200)
    trend = np.outer(np.sin(when / 40), np.ones(16))  # language drifting with time
    speech = rng.normal(size=(200, 16))
    e = trend * 5 + speech
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    adj = era_adjusted(e, when)
    corr = lambda x: abs(np.corrcoef(x @ np.ones(16), np.sin(when / 40))[0, 1])
    assert corr(e) > 0.9 and corr(adj) < 0.3


def test_place_names_resolve_to_modern_countries():
    from sotu.places import country_code

    assert country_code("Prussia") == 276  # Germany
    assert country_code("Soviet Union") == country_code("Russia") == 643
    assert country_code("London") == 826
    assert country_code("Georgia") is None  # usually the U.S. state
    assert country_code("Texas") is None and country_code("United States") is None


def test_anatomy_runs_tile_each_speech():
    import pandas as pd

    from sotu.visuals import anatomy

    df = pd.DataFrame({"id": ["a", "b"]})
    paras = pd.DataFrame({
        "speech_id": ["a"] * 4 + ["b"] * 2,
        "topic": ["Other", "Foreign affairs", "Defense & war", "Health care", "Education", "Education"],
        "text": ["one two", "x " * 6, "y " * 2, "z", "w", "v v v"],
    })
    runs = anatomy(df, paras)
    for sid, g in runs.groupby("id"):
        g = g.sort_values("s")
        assert g.s.iloc[0] == 0 and abs(g.e.iloc[-1] - 1) < 1e-9
        assert (g.s.iloc[1:].to_numpy() == g.e.iloc[:-1].to_numpy()).all()  # contiguous
    # Foreign affairs + Defense & war share a group, so they merge into one run.
    assert len(runs[runs.id == "a"]) == 3 and len(runs[runs.id == "b"]) == 1


def test_new_words_lines_wrap():
    import pandas as pd

    from sotu.visuals import new_words_lines

    rows = pd.DataFrame({"decade": [1900] * 3, "rank": [1, 2, 3], "term": ["monroe doctrine", "hague", "forest reserves"]})
    line = new_words_lines(rows, width=20).line.iloc[0]
    assert all(len(part) <= 20 for part in line.split("\n")) and "hague" in line


class WordTokenizer:
    """Deterministic tokenizer for exercising coverage without model downloads."""
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        import re
        words = list(re.finditer(r"\S+", text))
        return {"input_ids": list(range(len(words))), "offset_mapping": [m.span() for m in words]}


def test_complete_chunks_preserve_long_text_and_sentence_boundaries():
    from sotu.models import split_text

    tokenizer = WordTokenizer()
    text = "First sentence ends. " + "a long sentence without punctuation " * 20 + "Final words."
    chunks = split_text(text, tokenizer, max_length=10)
    assert "".join(chunks) == text
    assert chunks[0] == "First sentence ends. "
    assert all(len(tokenizer(c)["input_ids"]) + 2 <= 10 for c in chunks)
    assert chunks[-1].endswith("Final words.")


def test_chunk_pooling_uses_all_chunks_and_normalizes_weights():
    import numpy as np
    from sotu.models import _pool

    vectors = np.array([[1., 0.], [0., 1.], [0.2, 0.8]])
    pooled = _pool(vectors, np.array([0, 0, 1]), np.array([1., 3., 1.]), 2)
    np.testing.assert_allclose(pooled, [[0.25, 0.75], [0.2, 0.8]])
    np.testing.assert_allclose(pooled.sum(axis=1), 1)


def test_president_holdout_excludes_benchmark_sentences_from_same_president():
    import pandas as pd
    from sotu.validate_topics import benchmark_training_mask

    cap = pd.DataFrame({"president": ["Donald J. Trump", "Donald J. Trump", "Barack Obama", "Harry S Truman"]})
    keep = benchmark_training_mask(cap, ["Donald J. Trump - II", "Harry S. Truman"])
    assert keep.tolist() == [False, False, True, False]


def test_sentiment_class_share_is_not_a_net_score_band():
    import numpy as np
    from sotu.build import speech_table, sentence_table

    speech = {"id": "x", "date": "2000-01-01", "president": "P", "party": "Democratic",
              "delivery": "spoken", "formal_sotu": True, "title": "T", "url": "https://example.test"}
    processed = {"paragraph_embeddings": np.array([[1., 0.]]), "paragraphs": ["A passage."],
                 "stats": {"words": 2}, "tone": np.array([0., .6]),
                 "tone_labels": ["negative", "neutral", "positive"],
                 "tone_probs": np.array([[.49, .02, .49], [.1, .2, .7]]),
                 "emotions": np.array([[.8, .2], [.2, .8]]), "emotion_labels": ["neutral", "joy"],
                 "entities": Counter(), "sentences": ["A.", "B."], "sentence_paragraph": [0, 0]}
    df, _, _, _ = speech_table([speech], [processed], ["Health care"], lambda e: (["Health care"], np.array([.7])))
    assert df.neutral_share.iloc[0] == 0
    assert df.negative_share.iloc[0] == .5 and df.positive_share.iloc[0] == .5
    result = sentence_table(df, [processed])
    np.testing.assert_allclose(result[["tone_p_negative", "tone_p_neutral", "tone_p_positive"]].sum(axis=1), 1)
    assert result.tone_label.tolist() == ["negative", "positive"]


def test_model_revision_changes_processed_cache_key(monkeypatch, tmp_path):
    from sotu import process

    monkeypatch.setattr(process, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(process, "pipeline_manifest", lambda: {"revision": "one"})
    speech = {"id": "test", "paragraphs": ["The complete text."]}
    first = process._cache_path(speech)
    monkeypatch.setattr(process, "pipeline_manifest", lambda: {"revision": "two"})
    assert process._cache_path(speech) != first
    speech["paragraphs"].append("A changed input.")
    assert process._cache_path(speech) != first


def test_benchmark_embedding_cache_detects_same_length_content_changes(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    from sotu import validate_topics

    calls = []
    def embed(texts):
        calls.append(texts)
        return np.array([[len(t), 1.] for t in texts])
    monkeypatch.setattr(validate_topics, "CAP_EMB_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr(validate_topics.models, "embed", embed)
    a = pd.DataFrame({"description": ["first", "second"]})
    validate_topics.cap_embeddings(a)
    validate_topics.cap_embeddings(a)
    b = pd.DataFrame({"description": ["changed", "second"]})
    new = validate_topics.cap_embeddings(b)
    assert len(calls) == 2
    assert new[0, 0] == len("changed")


def test_scrape_always_refreshes_index_and_refresh_bypasses_transcript_cache(monkeypatch, tmp_path):
    from unittest.mock import Mock
    import hashlib
    from sotu import scrape

    monkeypatch.setattr(scrape, "ROOT", tmp_path)
    monkeypatch.setattr(scrape, "CACHE_DIR", tmp_path / "html")
    monkeypatch.setattr(scrape, "OUT_PATH", tmp_path / "speeches.jsonl")
    monkeypatch.setattr(scrape, "REQUEST_DELAY", 0)
    response = Mock(text=INDEX_HTML)
    get = Mock(return_value=response)
    monkeypatch.setattr(scrape.session, "get", get)
    scrape.CACHE_DIR.mkdir()
    index = scrape.CACHE_DIR / (hashlib.sha1(scrape.INDEX_URL.encode()).hexdigest() + ".html")
    index.write_text("stale and invalid index")
    monkeypatch.setattr(scrape, "parse_speech", lambda html: {"date": "2000-01-01", "title": "Test", "paragraphs": ["Text."]})
    scrape.main()
    assert scrape.OUT_PATH.exists()
    assert get.call_args_list[0].args[0] == scrape.INDEX_URL
    count = get.call_count
    scrape.main(refresh=True)
    assert get.call_count == count + 4  # fresh index and all three transcripts


def test_party_inference_does_not_treat_more_speeches_as_more_presidents():
    import pandas as pd
    from sotu.statistics import party_test

    df = pd.DataFrame({"president": list("abcd"), "party_group": ["Democratic"] * 2 + ["Republican"] * 2,
                       "year": [2000] * 4, "delivery": ["spoken"] * 4, "tone": [0., .2, .1, .4]})
    expected = party_test(df)
    duplicated = pd.concat([df, df.iloc[[0]]] * 3, ignore_index=True)
    result = party_test(duplicated)
    import numpy as np
    for key in ("democratic_mean", "republican_mean", "difference_R_minus_D", "exact_p", "ci_95"):
        np.testing.assert_allclose(result[key], expected[key])
    assert result["presidents_democratic"] == result["presidents_republican"] == 2
    assert expected["assignments"] == 6


def test_neighbor_null_permutates_whole_presidents():
    import numpy as np
    import pandas as pd
    from sotu.statistics import neighbor_party_test

    df = pd.DataFrame({"president": list("abcd"), "party_group": ["Democratic"] * 2 + ["Republican"] * 2, "year": [2000] * 4})
    vectors = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    result = neighbor_party_test(df, vectors, k=1)
    assert result["assignments"] == 6
    assert abs(result["null_mean"] - 1 / 3) < 1e-9
    assert result["same_party_share"] == 1


def test_topic_intervals_preserve_hidden_other_denominator():
    import pandas as pd
    from sotu.build import decade_topics

    df = pd.DataFrame({"id": ["a", "b"], "year": [2000, 2001], "words": [100, 100]})
    topics = pd.DataFrame({"id": ["a", "b"], "year": [2000, 2001], "topic": ["Health care"] * 2, "words": [10., 10.]})
    result = decade_topics(topics, df, ["Health care"]).iloc[0]
    assert result.share == .1 and result.ci_low == .1 and result.ci_high == .1
    assert result.n_speeches == 2


def test_human_audit_stays_pending_and_refuses_deleted_items(monkeypatch, tmp_path):
    import json
    import pandas as pd
    import pytest
    from sotu import human_review

    derived = tmp_path / "data" / "derived"
    derived.mkdir(parents=True)
    pd.DataFrame({"speech_id": ["x"], "sentence": [0], "tone_label": ["neutral"], "emotion": ["neutral"]}).to_parquet(derived / "sentences.parquet")
    row = {"key": "affect-x-0", "task": "affect", "speech_id": "x", "unit": 0, "year": 1800,
           "sampling_stratum": "early:all", "sampling_weight": 1., "text": "A sentence.",
           "reviewer_a": {"sentiment": None, "emotion": None},
           "reviewer_b": {"sentiment": None, "emotion": None},
           "adjudicated": {"sentiment": None, "emotion": None}}
    review = tmp_path / "human_review.jsonl"
    review.write_text(json.dumps(row) + "\n")
    review.with_name("human_review_sample_manifest.json").write_text(json.dumps({"sample_sha256": human_review.audit_hash([row])}))
    monkeypatch.setattr(human_review, "ROOT", tmp_path)
    monkeypatch.setattr(human_review, "REVIEW", review)
    monkeypatch.setattr(human_review, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(human_review, "held_out_topic_predictions", lambda rows: pytest.fail("Unlabeled audit must not train models"))
    human_review.evaluate()
    result = json.loads(human_review.RESULTS.read_text())
    assert result["status"] == "pending" and result["tasks"]["sentiment"]["scored"] == 0
    review.write_text("")
    with pytest.raises(ValueError, match="missing"):
        human_review.evaluate()


def test_human_scoring_uses_held_out_topic_predictions_and_keeps_reviewer_labels(monkeypatch, tmp_path):
    import json
    import pandas as pd
    from sotu import human_review

    derived = tmp_path / "data" / "derived"
    derived.mkdir(parents=True)
    pd.DataFrame({"speech_id": ["x"], "sentence": [0], "tone_label": ["positive"], "emotion": ["joy"]}).to_parquet(derived / "sentences.parquet")
    common = {"speech_id": "x", "unit": 0, "year": 2000, "sampling_stratum": "modern:long", "sampling_weight": 2., "text": "A sentence."}
    rows = [dict(common, key="topic-x-0", task="topic", reviewer_a={"topic": "Health care"},
                 reviewer_b={"topic": "Social welfare"}, adjudicated={"topic": "Health care"}),
            dict(common, key="affect-x-0", task="affect", reviewer_a={"sentiment": "positive", "emotion": "joy"},
                 reviewer_b={"sentiment": "positive", "emotion": "joy"}, adjudicated={"sentiment": "positive", "emotion": "joy"})]
    review = tmp_path / "human_review.jsonl"
    original = "".join(json.dumps(r) + "\n" for r in rows)
    review.write_text(original)
    review.with_name("human_review_sample_manifest.json").write_text(json.dumps({"sample_sha256": human_review.audit_hash(rows)}))
    monkeypatch.setattr(human_review, "ROOT", tmp_path)
    monkeypatch.setattr(human_review, "REVIEW", review)
    monkeypatch.setattr(human_review, "RESULTS", tmp_path / "results.json")
    calls = []
    def held_out(rows):
        calls.append(rows)
        return {("x", 0): "Health care"}
    monkeypatch.setattr(human_review, "held_out_topic_predictions", held_out)
    human_review.evaluate()
    result = json.loads(human_review.RESULTS.read_text())
    assert len(calls) == 1 and result["status"] == "complete"
    assert result["tasks"]["topic"]["weighted_metrics"]["Health care"]["f1-score"] == 1
    assert result["tasks"]["topic"]["reviewer_agreement"] == 0
    assert result["tasks"]["sentiment"]["weighted_metrics"]["positive"]["f1-score"] == 1
    assert review.read_text() == original


def test_all_real_benchmark_presidents_match_corpus_holdout_names():
    from sotu.validate_topics import load_cap, benchmark_training_mask
    from sotu.process import load_speeches

    corpus_names = {r["president"] for r in load_speeches()}
    cap = load_cap()
    assert set(cap.president) <= corpus_names
    for president in cap.president.unique():
        keep = benchmark_training_mask(cap, [president])
        assert not (cap.president[keep] == president).any()
        assert (cap.president[~keep] == president).all()
