PY ?= python
test:
	$(PY) -m pytest tests -q
smoke:
	$(PY) scripts/make_smoke_data.py --out data/smoke_raw
	$(PY) scripts/prepare_data.py --dataset unsw --raw-dir data/smoke_raw/unsw --out data/smoke_processed/unsw
	$(PY) scripts/prepare_data.py --dataset cicids --raw-dir data/smoke_raw/cicids --out data/smoke_processed/cicids
	$(PY) -m sentinelxlm.train --data data/smoke_processed/unsw --model sentinel_xlm --epochs 2
stats:
	$(PY) scripts/compute_split_counts.py
checksums:
	sha256sum -c CHECKSUMS.sha256
