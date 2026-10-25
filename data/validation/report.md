# Topic classifier validation

## Policy Agendas Project benchmark, 1946-2025 (sentence level, president-grouped folds)

| classifier | n | accuracy | macro-F1 |
|---|---|---|---|
| centroid (original method) | 26,077 | 0.477 | 0.453 |
| supervised (benchmark-trained) | 26,077 | 0.622 | 0.553 |

## Claude labels vs. human coders (benchmark sentences)

Agreement 67.3% on 300 sentences (macro-F1 0.577), by Claude's stated confidence:

- high: 81% of 152
- medium: 60% of 114
- low: 32% of 34

Run-to-run consistency (two independent labeling runs): 96.5% of 1045 items get the same label.

## Paragraph classifiers vs. Claude labels, by era

| era | n | centroid (original method) | benchmark-trained | benchmark + historical labels (used by the site) |
|---|---|---|---|---|
| 1790-1860 | 200 | acc 0.57 / F1 0.37 | acc 0.53 / F1 0.34 | acc 0.68 / F1 0.45 |
| 1861-1912 | 200 | acc 0.56 / F1 0.49 | acc 0.61 / F1 0.52 | acc 0.77 / F1 0.67 |
| 1913-1945 | 200 | acc 0.64 / F1 0.66 | acc 0.73 / F1 0.66 | acc 0.77 / F1 0.73 |
| 1946-2026 | 150 | acc 0.59 / F1 0.57 | acc 0.77 / F1 0.72 | acc 0.79 / F1 0.76 |

Five folds grouped by president. Every test president is removed from both the benchmark and paragraph training data.
The reference is Claude, not human historical ground truth. These labels were also used for method selection; this is not an untouched final test set.

## Per-topic paragraph agreement (out-of-fold, all eras)

| Topic | Precision | Recall | F1 | Reference support |
|---|---|---|---|---|
| Economy & jobs | 0.738 | 0.738 | 0.738 | 65 |
| Taxes & budget | 0.750 | 0.825 | 0.786 | 40 |
| Trade & tariffs | 0.714 | 0.833 | 0.769 | 30 |
| Banking & currency | 0.947 | 0.667 | 0.783 | 27 |
| Foreign affairs | 0.811 | 0.831 | 0.821 | 160 |
| Defense & war | 0.744 | 0.763 | 0.753 | 114 |
| Health care | 0.875 | 0.778 | 0.824 | 9 |
| Education | 0.875 | 0.778 | 0.824 | 9 |
| Immigration | 0.500 | 1.000 | 0.667 | 3 |
| Energy & environment | 0.750 | 0.667 | 0.706 | 18 |
| Crime & justice | 0.727 | 0.667 | 0.696 | 24 |
| Civil rights | 0.636 | 0.467 | 0.538 | 15 |
| Social welfare | 0.750 | 0.375 | 0.500 | 16 |
| Agriculture | 0.714 | 0.882 | 0.789 | 17 |
| Infrastructure | 0.788 | 0.839 | 0.812 | 31 |
| Native American affairs | 0.833 | 0.789 | 0.811 | 19 |
| Science & technology | 1.000 | 0.714 | 0.833 | 7 |
| Government & administration | 0.671 | 0.723 | 0.696 | 65 |
| Territories & expansion | 0.773 | 0.630 | 0.694 | 27 |
| Other | 0.603 | 0.648 | 0.625 | 54 |

## Production hybrid vs. human benchmark labels

Six president-grouped folds, removing each test president from both sources: accuracy 0.620, macro-F1 0.575 (n=26,077).

## Limits and independent review

Claude-human agreement is 67.3% overall on the original 300 benchmark sentences; high-confidence agreement describes only a subset. Repeatability is not accuracy.
The existing labeled paragraph sample excludes passages shorter than 20 words. New sampling includes them, but their human validation remains pending.
No independent human historical topic, sentiment or emotion evaluation has been completed. See human_review.jsonl and the annotation protocol.
Paragraph accuracy does not establish unbiased topic prevalence. Per-decade bootstrap intervals describe speech sampling variation only, not classifier error.

