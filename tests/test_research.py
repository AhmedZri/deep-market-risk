from market_risk.data import synthetic_prices
from market_risk.research import walk_forward


def test_walk_forward_excludes_frozen_test():
    report = walk_forward(synthetic_prices(220), folds=3, epochs=1)
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["last_train_target"] < fold["validation_start"]
        assert fold["last_validation_target"] < report["frozen_test_start"]
        assert set(fold["metrics"]) == {"lstm", "cnn", "historical20"}
