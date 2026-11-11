# Data

The datasets are public and are **not** redistributed here.

| Dataset | Files used | Required fields |
|---|---|---|
| UNSW-NB15 | `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv` and `NUSW-NB15_features.csv` (column names) | `srcip`, `dstip`, `Stime`, `Label`, `attack_cat` and the flow features |
| CICIDS2017 | the labelled flow CSVs with flow identifiers (`TrafficLabelling`) | `Source IP`, `Destination IP`, `Timestamp`, `Label` and the flow features |

Graph construction (source-IP nodes, 60-second windows, trust label from the share of attack flows) needs the source IP and the timestamp of every flow. The UNSW-NB15 training/testing partition files do not contain these fields, so the loader reads files that do. `attack_cat` / the CICIDS2017 label string is used only for the per-attack-type evaluation and is never a model input.

```
python scripts/prepare_data.py --dataset unsw   --raw-dir /path/to/UNSW-NB15  --out data/processed/unsw
python scripts/prepare_data.py --dataset cicids --raw-dir /path/to/CICIDS2017 --out data/processed/cicids
```

Each run writes `windows_{train,val,test}.pkl`, `splits.npz` (flow indices of the three partitions), `meta.json` (input dimension, window and node counts) and `stats.json` (class counts, number of duplicates removed, split sizes) to the output directory.

`data/paper_dataset_stats.json` is the single source of the counts reported in Table 7 (raw corpus, duplicate removal, benign ratio and the 70/10/20 split sizes). `python scripts/compute_split_counts.py` checks it against the split arithmetic in `sentinelxlm/splits.py`:

| Dataset | Records used | Train | Validation | Test |
|---|---|---|---|---|
| UNSW-NB15 (93,000 benign / 164,673 attack, 36.09% benign) | 257,673 | 180,370 | 25,768 | 51,535 |
| CICIDS2017 after removing 8,867 duplicate benign rows (2,264,230 benign / 557,646 attack, 80.24% benign) | 2,821,876 | 1,975,312 | 282,188 | 564,376 |
