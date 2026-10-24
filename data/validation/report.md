# Topic classifier validation

## Policy Agendas Project benchmark, 1946-2025 (sentence level, leave-president-out)

| classifier | n | accuracy | macro-F1 |
|---|---|---|---|
| centroid (original method) | 26,077 | 0.477 | 0.453 |
| supervised (benchmark-trained) | 26,077 | 0.623 | 0.552 |

## Claude labels vs. human coders (benchmark sentences)

Agreement 67.3% on 300 sentences (macro-F1 0.577), by Claude's stated confidence:

- high: 81% of 152
- medium: 60% of 114
- low: 32% of 34

Run-to-run consistency (two independent labeling runs): 96.5% of 1045 items get the same label.

## Paragraph classifiers vs. Claude labels, by era

| era | n | centroid (original method) | benchmark-trained | benchmark + historical labels (used by the site) |
|---|---|---|---|---|
| 1790-1860 | 200 | acc 0.57 / F1 0.37 | acc 0.54 / F1 0.37 | acc 0.69 / F1 0.46 |
| 1861-1912 | 200 | acc 0.56 / F1 0.48 | acc 0.62 / F1 0.53 | acc 0.76 / F1 0.65 |
| 1913-1945 | 200 | acc 0.64 / F1 0.66 | acc 0.74 / F1 0.67 | acc 0.76 / F1 0.69 |
| 1946-2026 | 150 | acc 0.59 / F1 0.57 | acc 0.79 / F1 0.79 | acc 0.81 / F1 0.81 |

Best classifier by macro-F1 per era: 1790-1860: **hybrid**, 1861-1912: **hybrid**, 1913-1945: **hybrid**, 1946-2026: **hybrid**

