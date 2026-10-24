"""Scrape every State of the Union message from The American Presidency Project.

The UCSB guidebook page has a table with one row per presidential term and a
link per message. Columns 3-7 are messages delivered as a speech, columns 8-12
are written messages, so delivery mode comes straight from the table.

Usage:
    uv run python -m sotu.scrape            # fetch only speeches not yet in data/speeches.jsonl
    uv run python -m sotu.scrape --refresh  # re-fetch everything
    uv run python -m sotu.scrape --check    # list new speeches without fetching; exit 1 if any
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from sotu.presidents import canonical_name, party_of

INDEX_URL = (
    "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/"
    "annual-messages-congress-the-state-the-union"
)
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "speeches.jsonl"
CACHE_DIR = ROOT / ".cache" / "html"
HEADERS = {"User-Agent": "NLP-SOTU research project (https://github.com/tiroger/NLP-SOTU)"}
REQUEST_DELAY = 1.0  # seconds between uncached requests

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str) -> str:
    """GET a URL, caching the body on disk so re-runs don't hit the server."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if path.exists():
        return path.read_text()
    time.sleep(REQUEST_DELAY)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


def parse_index(html: str) -> list[dict]:
    """Return one entry per message linked from the guidebook table."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("a", attrs={"name": "Table of SOTU"}).find_parent("table")
    entries, president = [], None
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 12:
            continue
        # Rows for a president's other terms leave the name cell blank.
        if cells[0].find("strong"):
            president = canonical_name(cells[0].get_text())
        if president is None:
            continue
        for col, cell in enumerate(cells[2:12]):
            delivery = "spoken" if col < 5 else "written"
            for link in cell.find_all("a"):
                label = link.get_text(strip=True)
                if not re.fullmatch(r"\d{4}", label):
                    continue  # footnote markers such as "‡" or "+"
                # A single asterisk after the link marks a first-year address to a
                # joint session, which is not formally a State of the Union. Double
                # and triple asterisks are footnotes about Taft's multi-part messages.
                sibling = link.next_sibling
                marker = sibling.strip() if isinstance(sibling, str) else ""
                formal = marker != "*"
                entries.append({
                    "president": president,
                    "index_year": int(label),
                    "delivery": delivery,
                    "formal_sotu": formal,
                    "url": link["href"],
                })
    return entries


def parse_speech(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="field-docs-content")
    paragraphs = [" ".join(p.get_text(" ").split()) for p in content.find_all("p")]
    if not any(paragraphs):
        # A few early documents keep their text directly in the div.
        paragraphs = [line.strip() for line in content.get_text("\n").split("\n")]
    date_text = soup.find("span", class_="date-display-single").get_text(strip=True)
    return {
        "title": soup.find("div", class_="field-ds-doc-title").get_text(strip=True),
        "date": datetime.strptime(date_text, "%B %d, %Y").date().isoformat(),
        "paragraphs": [p for p in paragraphs if p],
    }


def speech_id(entry: dict, date: str, taken: set[str]) -> str:
    surname = entry["president"].split()[-1].lower()
    base = f"{date[:4]}-{surname}"
    sid, n = base, 2
    while sid in taken:
        sid, n = f"{base}-{n}", n + 1
    return sid


def load_existing() -> dict[str, dict]:
    if not OUT_PATH.exists():
        return {}
    with OUT_PATH.open() as f:
        return {s["url"]: s for s in map(json.loads, f)}


def check() -> int:
    """Print messages listed in the index but missing locally. Returns how many."""
    existing = load_existing()
    new = [e for e in parse_index(session.get(INDEX_URL, timeout=30).text) if e["url"] not in existing]
    for e in new:
        print(f"NEW: {e['index_year']} {e['president']} {e['url']}")
    print(f"{len(new)} new message(s)")
    return len(new)


def main(refresh: bool = False) -> None:
    existing = {} if refresh else load_existing()
    entries = parse_index(fetch(INDEX_URL))
    print(f"Index lists {len(entries)} messages; {len(existing)} already scraped")

    speeches, taken = [], set()
    for entry in sorted(entries, key=lambda e: (e["index_year"], e["url"])):
        if entry["url"] in existing:
            speech = existing[entry["url"]]
        else:
            print(f"  fetching {entry['index_year']} {entry['president']}")
            speech = {**entry, **parse_speech(fetch(entry["url"]))}
            speech["party"] = party_of(entry["president"])
        speech["id"] = speech_id(speech, speech["date"], taken)
        taken.add(speech["id"])
        speeches.append(speech)

    speeches.sort(key=lambda s: (s["date"], s["id"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for s in speeches:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(speeches)} speeches to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-scrape every speech")
    parser.add_argument("--check", action="store_true", help="only report speeches not yet scraped")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(1 if check() else 0)
    main(args.refresh)
