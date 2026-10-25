# ParlaSent English reference snapshots

Creators: Michal Mochtak, Peter Rupnik, Katja Meden and Nikola Ljubešić.

Source: [The multilingual sentiment dataset of parliamentary debates ParlaSent 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1868), distributed by CLARIN.SI. [Paper](https://aclanthology.org/2024.lrec-main.1393/).

License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). The raw files are unchanged; source.json records immutable Hugging Face URLs, revision and SHA-256 checksums. This attribution and license apply to the ParlaSent data and reproduced sentence text in the derived prediction/error files, not to unrelated repository content.

The 2,600 English test sentences were sampled from mid-length UK parliamentary speech and annotated by one trained annotator. The English training sample is enriched using sentiment lexicons and has two annotators plus reconciliation. Its class prior is therefore a baseline, not an estimate of population class prevalence. Raw labels are title case; evaluation normalizes their case only, retaining the dataset's official three-way mapping.

The current frozen SOTU model is evaluated without fitting on these annotations. VADER uses its existing compound thresholds ±0.05. Training-set counts supply a fixed prior/majority baseline; an always-neutral baseline is reported separately. No dataset agreement subset or test threshold was selected after inspecting outcomes.
