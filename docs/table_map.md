# Manuscript table / experiment -> script map

| Manuscript item | Script (run from `experiments/`) | Aggregation (`make_tables.py --kind`) |
|---|---|---|
| Table 5 – rule quality (full / 30% noisy / no rules) | `run_rules.py` | `rules` |
| Table 7 – dataset statistics and splits | `scripts/prepare_data.py` (writes `stats.json`), `data/paper_dataset_stats.json` | `scripts/compute_split_counts.py` |
| Table 8 – SENTINEL-XLM hyperparameters | `configs/sentinel_xlm.yaml` (checked by `tests/test_config.py`) | – |
| Table 8b – baseline hyperparameters | `configs/baselines.yaml`, `run_baseline_search.py` | – |
| Tables 9, 9b, 20 – comparison with ten baselines, SDs, CIs, paired t-tests | `run_main.py` | `main` |
| Tables 10–12 – module and design ablations | `run_ablation.py` | `variants` |
| Table 13 – edge injection / deletion robustness | `run_robustness.py` | `robustness` |
| Table 14 – measured latency, memory, parameters | `run_efficiency.py` | `efficiency` |
| Table 14b – scalability on synthetic graphs, log-log slopes | `run_scalability.py` | `scalability` |
| Table 15 – edge projection (analytical) | `run_efficiency.py` with `configs/edge_projection.yaml` | `efficiency` |
| Table 16 – per-attack-category results | `run_attack_types.py` | `attack_types` |
| Table 17 – temporal window size | `run_window.py` | `window` |
| Table 18 – explainability ratings | `analyze_explainability.py` (input `results/explainability/ratings.csv`) | – |
| Table 19 – streaming comparison with / without SEM | `run_streaming.py` | `streaming` |
| Table A1 – trust-label threshold sensitivity | `run_label_threshold.py` | `threshold` |
| Node-disjoint evaluation | `run_node_disjoint.py` | `main` |
| Appendix B, Table B1 – LLM-agent simulation | `llm_simulation/run_simulation.py`, `llm_simulation/evaluate_simulation.py` | – |

All scripts write one CSV row per run (seed); `make_tables.py` reports the mean, the sample standard deviation and the t-based 95% confidence interval (t = 2.776 for five runs) and, for `main`, the paired t-tests of every model against EdgeTrust-6G.
