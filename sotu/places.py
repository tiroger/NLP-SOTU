"""Resolve place names from spaCy NER to present-day countries (ISO 3166 numeric).

Historical names map to the country that occupies the territory today
(Prussia -> Germany, Persia -> Iran), which is what a modern world map can
show. Names that are ambiguous with U.S. places are skipped.
"""

import pycountry

ALIASES = {
    # Britain and its parts
    "great britain": "GBR", "britain": "GBR", "england": "GBR", "united kingdom": "GBR", "scotland": "GBR",
    "the united kingdom": "GBR", "british empire": "GBR",
    # Historical and common names
    "russia": "RUS", "soviet union": "RUS", "the soviet union": "RUS", "u.s.s.r.": "RUS", "ussr": "RUS",
    "prussia": "DEU", "west germany": "DEU", "east germany": "DEU", "federal republic of germany": "DEU",
    "persia": "IRN", "iran": "IRN", "siam": "THA", "burma": "MMR", "ceylon": "LKA", "formosa": "TWN", "taiwan": "TWN",
    "turkey": "TUR", "ottoman empire": "TUR", "the ottoman empire": "TUR", "the porte": "TUR",
    "holland": "NLD", "the netherlands": "NLD", "netherlands": "NLD",
    "austria-hungary": "AUT", "austria hungary": "AUT",
    "naples": "ITA", "sardinia": "ITA", "the two sicilies": "ITA", "two sicilies": "ITA", "sicily": "ITA",
    "tripoli": "LBY", "algiers": "DZA", "tunis": "TUN",
    "hayti": "HTI", "santo domingo": "DOM", "san domingo": "DOM", "dominican republic": "DOM",
    "new granada": "COL", "colombia": "COL", "the argentine republic": "ARG", "argentine republic": "ARG",
    "korea": "KOR", "south korea": "KOR", "republic of korea": "KOR", "north korea": "PRK",
    "vietnam": "VNM", "viet-nam": "VNM", "viet nam": "VNM", "south vietnam": "VNM", "north vietnam": "VNM",
    "laos": "LAO", "cambodia": "KHM", "syria": "SYR", "venezuela": "VEN", "bolivia": "BOL", "tanzania": "TZA",
    "czechoslovakia": "CZE", "yugoslavia": "SRB", "bosnia": "BIH", "kosovo": "SRB",
    "the philippines": "PHL", "philippines": "PHL", "philippine islands": "PHL", "the philippine islands": "PHL",
    "china": "CHN", "people's republic of china": "CHN", "the people's republic of china": "CHN",
    "congo": "COD", "the congo": "COD", "zaire": "COD", "rhodesia": "ZWE", "abyssinia": "ETH",
    "palestine": "PSE", "gaza": "PSE", "the west bank": "PSE", "west bank": "PSE",
    "moldova": "MDA", "macedonia": "MKD", "ivory coast": "CIV", "cote d'ivoire": "CIV",
    # Capitals, which 19th-century diplomatic passages use for their governments
    "london": "GBR", "paris": "FRA", "berlin": "DEU", "madrid": "ESP", "rome": "ITA", "vienna": "AUT",
    "st. petersburg": "RUS", "saint petersburg": "RUS", "moscow": "RUS", "peking": "CHN", "beijing": "CHN",
    "tokyo": "JPN", "constantinople": "TUR", "havana": "CUB", "baghdad": "IRQ", "kabul": "AFG", "tehran": "IRN",
    "hanoi": "VNM", "saigon": "VNM", "seoul": "KOR", "managua": "NIC", "panama city": "PAN",
}
# Display names for codes whose ISO short name is awkward on a map tooltip.
NAMES = {643: "Russia (incl. USSR)", 826: "United Kingdom", 410: "South Korea", 408: "North Korea", 158: "Taiwan",
         364: "Iran", 862: "Venezuela", 68: "Bolivia", 760: "Syria", 704: "Vietnam", 418: "Laos", 498: "Moldova",
         834: "Tanzania", 180: "DR Congo", 275: "Palestine"}
# U.S. states, cities and regions that pycountry would otherwise match to a country.
SKIP = {"georgia", "jersey", "panama canal", "mexico city", "new mexico", "guinea", "chad", "jordan river"}

_cache: dict[str, int | None] = {}


def country_code(name: str) -> int | None:
    """ISO 3166-1 numeric code for a place name, or None if it isn't a foreign country."""
    key = " ".join(name.lower().replace("’", "'").split())
    if key in _cache:
        return _cache[key]
    code = None
    if key not in SKIP:
        alpha3 = ALIASES.get(key)
        try:
            country = pycountry.countries.get(alpha_3=alpha3) if alpha3 else pycountry.countries.lookup(key)
            if country and country.alpha_3 != "USA":
                code = int(country.numeric)
        except LookupError:
            pass
    _cache[key] = code
    return code


def country_name(code: int) -> str:
    if code in NAMES:
        return NAMES[code]
    country = pycountry.countries.get(numeric=f"{code:03d}")
    return getattr(country, "common_name", None) or country.name
