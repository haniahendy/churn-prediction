# churn-prediction

Customer churn classifier with a scikit-learn pipeline (scaling + one-hot encoding), a logistic regression and a gradient boosting model compared on ROC-AUC and PR-AUC, and a simple JSON run log for experiment tracking.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q     # 4 tests
python train.py         # prints metrics, writes runs/*.json
```

## Measured result (23 Nov 2026, seed 0, 500 test rows)
| model | ROC-AUC | PR-AUC |
|---|---|---|
| logistic regression | 0.771 | 0.778 |
| gradient boosting | 0.747 | 0.749 |

Logistic regression wins here because the synthetic labels are generated from a logistic function of the features. That is not a general statement about the two model types.

## Data and limits
Synthetic customers (`churn/data.py`), churn rate about 51 percent, so it is not realistic (real churn is usually far lower and the classes imbalanced). No hyperparameter tuning, no calibration, no cross-validation, one split. Tracking is a plain JSON file, not MLflow.

## Provenance
Built with AI assistance as a learning/portfolio project.
