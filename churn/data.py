import numpy as np
import pandas as pd


def make_customers(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Synthetic customers. Churn probability is a known function of the features plus noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charge": rng.normal(60, 20, n).clip(10, 150).round(2),
        "support_calls": rng.poisson(1.5, n),
        "contract": rng.choice(["monthly", "yearly"], n, p=[0.6, 0.4]),
    })
    z = (-1.0 - 0.04 * df["tenure_months"] + 0.02 * df["monthly_charge"]
         + 0.4 * df["support_calls"] + 1.2 * (df["contract"] == "monthly"))
    p = 1 / (1 + np.exp(-z))
    df["churned"] = (rng.random(n) < p).astype(int)
    return df
