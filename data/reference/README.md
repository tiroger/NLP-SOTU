# Pinned benchmark snapshot

`US-Exec_SOTU_2025.csv.gz` is a compressed, unchanged snapshot of the Policy Agendas Project's U.S. State of the Union dataset (1946–2025).

Source: https://minio.la.utexas.edu/compagendas/datasetfiles/US-Exec_SOTU_2025.csv

Dataset project: https://www.comparativeagendas.net/us/datasets

Uncompressed SHA-256: `dfa6801aaf93a795f7e8ecb51e7a9f4f38983f5198e3ad0869207e84998b97ee`

The loader restores this committed snapshot into the external cache and checks the hash. It rejects changed upstream or cached bytes, avoiding a silent change to training or evaluation data. Human codes remain the Policy Agendas Project's work; the project's topic taxonomy is mapped in `sotu/validate_topics.py`.
