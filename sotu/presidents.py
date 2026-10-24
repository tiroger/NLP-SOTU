"""President metadata: canonical names and party affiliation."""

# The UCSB index table uses a few spellings that differ from how presidents are
# usually written; map them to a single canonical display name.
NAME_FIXES = {
    "Donald J. Trump - I": "Donald J. Trump",
    "Donald J. Trump - II": "Donald J. Trump",
    "Harry S Truman": "Harry S. Truman",
    "George Bush": "George H. W. Bush",
}

PARTY = {
    "George Washington": "Unaffiliated",
    "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican",
    "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican",
    "John Quincy Adams": "Democratic-Republican",
    "Andrew Jackson": "Democratic",
    "Martin Van Buren": "Democratic",
    "John Tyler": "Whig",
    "James K. Polk": "Democratic",
    "Zachary Taylor": "Whig",
    "Millard Fillmore": "Whig",
    "Franklin Pierce": "Democratic",
    "James Buchanan": "Democratic",
    "Abraham Lincoln": "Republican",
    "Andrew Johnson": "National Union",
    "Ulysses S. Grant": "Republican",
    "Rutherford B. Hayes": "Republican",
    "Chester A. Arthur": "Republican",
    "Grover Cleveland": "Democratic",
    "Benjamin Harrison": "Republican",
    "William McKinley": "Republican",
    "Theodore Roosevelt": "Republican",
    "William Howard Taft": "Republican",
    "Woodrow Wilson": "Democratic",
    "Warren G. Harding": "Republican",
    "Calvin Coolidge": "Republican",
    "Herbert Hoover": "Republican",
    "Franklin D. Roosevelt": "Democratic",
    "Harry S. Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican",
    "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic",
    "Richard M. Nixon": "Republican",
    "Gerald R. Ford": "Republican",
    "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican",
    "George H. W. Bush": "Republican",
    "William J. Clinton": "Democratic",
    "George W. Bush": "Republican",
    "Barack Obama": "Democratic",
    "Donald J. Trump": "Republican",
    "Joseph R. Biden": "Democratic",
}


def canonical_name(name: str) -> str:
    name = " ".join(name.split())
    return NAME_FIXES.get(name, name)


def party_of(name: str) -> str:
    try:
        return PARTY[name]
    except KeyError:
        raise KeyError(f"No party recorded for {name!r}; add it to sotu/presidents.py") from None
