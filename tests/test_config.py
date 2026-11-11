from sentinelxlm.config import sentinel_cfg, load_yaml

def test_table8_hyperparameters():
    c = sentinel_cfg(); m, s, t = c["model"], c["sem"], c["training"]
    assert (m["embedding_dim"], m["num_layers"], m["temporal_window"], m["num_heads"]) == (128, 3, 10, 4)
    assert m["lambda_init"] == 0.6 and s["gamma"] == 0.95 and s["rho"] == 0.3
    assert t["learning_rate"] == 0.001 and t["batch_size"] == 256 and t["epochs"] == 100 and t["early_stopping_patience"] == 20
    assert t["logic_loss_weight"] == 0.1 and t["regularization_weight"] == 0.01
    assert t["seeds"] == [42, 123, 456, 789, 1024]
    assert c["graph"]["window_seconds"] == 60 and c["graph"]["label_threshold"] == 0.30 and c["graph"]["split_seed"] == 42

def test_table8b_baseline_hyperparameters():
    b = load_yaml("baselines.yaml")
    assert len(b) == 10
    assert (b["gcn_trust"]["d"], b["gcn_trust"]["batch"]) == (64, 128)
    assert (b["dysat_trust"]["d"], b["edgetrust_6g"]["layers"], b["llm_gnn"]["lr"]) == (256, 3, 0.0001)
