import json

from churn.data import make_customers
from churn.model import evaluate, log_run


def test_data_shape_and_binary_target():
    df = make_customers(500)
    assert len(df) == 500 and set(df["churned"]) == {0, 1}
    assert 0.1 < df["churned"].mean() < 0.9


def test_models_beat_chance_clearly():
    df = make_customers()
    for kind in ("logreg", "gboost"):
        assert evaluate(df, kind)["roc_auc"] > 0.7


def test_results_reproducible():
    df = make_customers()
    assert evaluate(df, "logreg") == evaluate(df, "logreg")


def test_run_logging(tmp_path):
    r = evaluate(make_customers(600), "logreg")
    path = log_run(r, str(tmp_path))
    assert json.loads(path.read_text()) == r
