# External sentiment validation

## Protocol

Frozen SOTU RoBERTa evaluated on ParlaSent's 2,600 English test sentences. No fitting, calibration, threshold tuning or model selection was performed on this test set.
Labels measure textual valence in parliamentary speech. They do not measure presidential happiness, endorsement or psychological emotion.
The English test sentences were annotated by one trained annotator; the sample covers mid-length UK parliamentary sentences, not historical U.S. addresses.

## Classification

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| sotu_roberta | 0.693 | 0.690 |
| training_class_prior | 0.323 | 0.163 |
| always_neutral | 0.408 | 0.193 |
| vader_default | 0.517 | 0.518 |

### Frozen SOTU model, by class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.837 | 0.622 | 0.714 | 841 |
| neutral | 0.607 | 0.824 | 0.699 | 1062 |
| positive | 0.758 | 0.580 | 0.657 | 697 |

### Confusion matrix

Rows are human labels; columns are predicted negative / neutral / positive.

| Human label | Negative | Neutral | Positive |
|---|---:|---:|---:|
| negative | 523 | 295 | 23 |
| neutral | 81 | 875 | 106 |
| positive | 21 | 272 | 404 |

## Interpretation

The frozen model outperforms the fixed baselines on this benchmark. It predicts neutral for 1,442 sentences (55.5%), versus 1,062 (40.8%) in the human reference. Many positive and negative passages are compressed into neutral.
Its net score is higher than the human category-code average by +0.049 (bootstrap interval +0.029 to +0.069). This bias is specific to this benchmark; subtracting it from historical SOTU scores would be unjustified without SOTU labels.

## Probability calibration and net score

Multiclass Brier is the sum of three squared probability errors (range 0–2); log loss uses natural logarithms. Lower values are better. VADER compound is not a class-probability distribution, so probability metrics are omitted for VADER.

| System | Brier | Log loss | Mean score | Human mean class code | Mean signed error |
|---|---:|---:|---:|---:|---:|
| sotu_roberta | 0.415 | 0.698 | -0.007 | -0.055 | +0.049 |
| training_class_prior | 0.706 | 1.151 | -0.238 | -0.055 | -0.182 |

The human comparison codes negative = −1, neutral = 0, positive = +1. Net score is the model's expected value of that category code; its magnitude has not been validated as psychological intensity.

### Source-document bootstrap intervals (95%)

| Metric | Lower | Upper |
|---|---:|---:|
| accuracy | +0.675 | +0.710 |
| macro_f1 | +0.671 | +0.707 |
| score_bias_vs_class_code | +0.029 | +0.069 |
| score_mae_vs_class_code | +0.383 | +0.410 |
| brier_multiclass_sum | +0.397 | +0.435 |
| log_loss | +0.665 | +0.730 |

Resampled 2600 source documents, 2,000 times, seed 23. Intervals describe this benchmark, not transfer to SOTU.

[Classwise reliability plots](reliability.html) · [Full metrics](results.json) · [Per-sentence predictions](predictions.jsonl)

## Limits and next test

This is an external-domain diagnostic, not a measurement of historical SOTU accuracy. An error on this benchmark does not itself establish an error on SOTU; success would not establish historical validity either.
Any model revision prompted by these results must be evaluated on a fresh untouched test, rather than repeatedly optimizing this score.
Independent SOTU annotations remain pending. Use the president-separated development/test sample in sentiment_audit.jsonl to assess per-era signed score error and the historical trend.

## Source and attribution

Mochtak, Rupnik, Meden and Ljubešić, ParlaSent 1.0 (CC BY-SA 4.0): https://www.clarin.si/repository/xmlui/handle/11356/1868
Pinned dataset revision: `0587c2b6499fbc68a7623439c2af2b24748968dc`. Checksums, model revision and implementation hashes are recorded in results.json.
