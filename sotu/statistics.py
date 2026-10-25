"""Reproducible descriptive comparisons, with presidents as inference units."""

from itertools import combinations

import numpy as np


def party_test(df, column="tone", spoken_only=False, seed=0):
    modern = df[(df.year >= 1965) & df.party_group.isin(["Democratic", "Republican"])]
    if spoken_only:
        modern = modern[modern.delivery == "spoken"]
    groups = modern.groupby(["president", "party_group"])[column].mean().reset_index()
    d = groups.loc[groups.party_group == "Democratic", column].to_numpy()
    r = groups.loc[groups.party_group == "Republican", column].to_numpy()
    pooled, observed = np.r_[d, r], r.mean() - d.mean()
    diffs = []
    for idx in combinations(range(len(pooled)), len(d)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        diffs.append(pooled[~mask].mean() - pooled[mask].mean())
    rng = np.random.default_rng(seed)
    boot = rng.choice(r, (10000, len(r))).mean(axis=1) - rng.choice(d, (10000, len(d))).mean(axis=1)
    return {"measure": column, "spoken_only": spoken_only, "presidents_democratic": len(d),
            "presidents_republican": len(r), "democratic_mean": float(d.mean()),
            "republican_mean": float(r.mean()), "difference_R_minus_D": float(observed),
            "ci_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "exact_p": float(np.mean(np.abs(diffs) >= abs(observed) - 1e-12)),
            "assignments": len(diffs),
            "interpretation": "Observational comparison assuming exchangeable presidents; non-significance does not establish equivalence."}


def neighbor_party_test(df, adjusted, k=5):
    """Fixed high-dimensional graph; exact label permutations of whole presidents."""
    keep = (df.year >= 1965) & df.party_group.isin(["Democratic", "Republican"])
    modern, vectors = df.loc[keep], adjusted[keep.to_numpy()]
    presidents, owner = np.unique(modern.president, return_inverse=True)
    party = modern.groupby("president").party_group.first().reindex(presidents).to_numpy()
    sim = vectors @ vectors.T
    sim[owner[:, None] == owner[None, :]] = -np.inf
    nn = np.argsort(-sim, axis=1)[:, :k]
    if not np.isfinite(np.take_along_axis(sim, nn, axis=1)).all():
        raise ValueError("Not enough eligible cross-president neighbors")
    # Give each query president equal weight, regardless of number of speeches.
    def statistic(labels):
        hits = (labels[owner[:, None]] == labels[owner[nn]]).mean(axis=1)
        return float(np.mean([hits[owner == p].mean() for p in range(len(presidents))]))
    observed = statistic(party)
    null = []
    for idx in combinations(range(len(presidents)), int((party == "Democratic").sum())):
        labels = np.zeros(len(presidents), int)
        labels[list(idx)] = 1
        null.append(statistic(labels))
    null = np.array(null)
    center = float(null.mean())
    return {"start_year": 1965, "k": k, "speeches": len(modern), "presidents": len(presidents),
            "same_party_share": observed, "null_mean": center,
            "exact_two_sided_p": float((np.abs(null - center) >= abs(observed - center) - 1e-12).mean()),
            "assignments": len(null), "query_weight": "equal president", "space": "era-adjusted cosine embeddings",
            "interpretation": "Conditional on this representation and graph; no claim of party equivalence."}


def comparison_report(df, embeddings, adjust):
    metrics = ["tone", "positive_share", "negative_share", "neutral_share", "grade_level", "we_per_1k", "i_per_1k"]
    return {
        "seed": 0, "party": [party_test(df, col, spoken) for spoken in (False, True) for col in metrics],
        "neighbor_party": [{"era_sigma_years": sigma, **neighbor_party_test(
            df, adjust(embeddings, df["when"].to_numpy(), sigma=sigma))} for sigma in (5, 10, 20)],
        "limits": "Exploratory, unadjusted multiple comparisons. Neither party nor delivery mode was randomized; era adjustment does not identify a causal party effect.",
        "tone_threshold_sensitivity": [
            {"threshold": t, "decade": int(d),
             "strong_positive_share": float(g[f"strong_positive_{t}"].mean()),
             "strong_negative_share": float(g[f"strong_negative_{t}"].mean())}
            for t in (0.25, 0.5, 0.75) for d, g in df.groupby(df.year // 10 * 10)
        ],
    }
