# SENTINEL-XLM: reproducibility package

Reference implementation (plain PyTorch, no graph-library dependency) of SENTINEL-XLM (TT-GAT + neuro-symbolic reasoning engine + self-evolution module), the ten baselines, the data pipeline and one script per table of the manuscript.

## Layout
```
sentinelxlm/     data adapters, graph construction, splits, SMOTE, layers, SENTINEL-XLM, baselines, training, streaming, statistics
configs/         sentinel_xlm.yaml (Table 8), baselines.yaml (Table 8b), datasets.yaml, edge_projection.yaml
rules/           trust_rules.yaml (R = 15 initial rules)
scripts/         prepare_data.py, make_smoke_data.py, compute_split_counts.py
experiments/     one script per table + make_tables.py (see docs/table_map.md)
llm_simulation/  LangGraph agent simulation, feature mapping (Table B1), evaluation
tests/           unit and pipeline tests
data/            data README, paper_dataset_stats.json (counts of Table 7)
results/smoke/   outputs of the smoke run (pipeline check only)
```

## Install and test
```
pip install -r requirements.txt
make test          # 32 tests
make smoke         # synthetic data, 2-epoch training
```

## Reproduce the tables
1. Download the raw datasets and run `scripts/prepare_data.py` for both datasets (see `data/README.md`).
2. Run the scripts listed in `docs/table_map.md` (for example `python experiments/run_main.py --unsw data/processed/unsw --cicids data/processed/cicids --device cuda --out results/table9_per_run.csv`), then aggregate with `experiments/make_tables.py`.
3. Five seeds (42, 123, 456, 789, 1024) are used throughout; seed 42 also fixes the data split. The five runs share one split and differ in initialisation and mini-batch order.

## Files that document choices
`DEVIATIONS.md` (implementation choices and differences from the manuscript description), `AI_USE.md`, `docs/table_map.md`, `CHECKSUMS.sha256` (`make checksums`).

## License
MIT (see `LICENSE`).
