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
