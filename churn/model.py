import json
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["tenure_months", "monthly_charge", "support_calls"]
CAT = ["contract"]


def build(kind: str) -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(), CAT)])
    clf = LogisticRegression(max_iter=1000) if kind == "logreg" else GradientBoostingClassifier(random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(df, kind: str, seed: int = 0) -> dict:
    X, y = df[NUM + CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    model = build(kind).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return {"model": kind, "roc_auc": round(roc_auc_score(yte, proba), 3),
            "pr_auc": round(average_precision_score(yte, proba), 3),
            "churn_rate": round(float(y.mean()), 3), "n_test": int(len(yte))}


def log_run(result: dict, folder: str = "runs") -> Path:
    """Minimal experiment tracking: one JSON file per run (no MLflow dependency)."""
    Path(folder).mkdir(exist_ok=True)
    path = Path(folder) / f"{result['model']}_{int(time.time()*1000)}.json"
    path.write_text(json.dumps(result, indent=2))
    return path
