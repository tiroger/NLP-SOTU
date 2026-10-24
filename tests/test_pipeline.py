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
