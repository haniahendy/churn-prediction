from churn.data import make_customers
from churn.model import evaluate, log_run

df = make_customers()
for kind in ("logreg", "gboost"):
    r = evaluate(df, kind)
    log_run(r)
    print(r)
