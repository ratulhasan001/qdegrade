"""Supplementary analysis: an on-device arm that uses ONLY the quantized model's prefill
(no FP16 reference), including its last-position entropy and margin, which the notebook's
`edge` arm did not include. Same XGBoost config and 5x5 CV (seed 0) as the notebook.
Writes results/extra_edge_last.csv.
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 0

def xgb(y):
    spw = float((len(y) - y.sum()) / y.sum())
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.85,
                         colsample_bytree=0.85, reg_lambda=1.5, eval_metric="logloss",
                         n_jobs=4, random_state=SEED, scale_pos_weight=spw)

LAB = pd.read_parquet(os.path.join(ROOT, "feats", "labels.parquet"))
T0 = pd.read_parquet(os.path.join(ROOT, "feats", "tier0.parquet"))
F_T0 = [c for c in T0.columns if c.startswith("t0_")]
rows = []
for prec in ["nf4", "fp4", "int8"]:
    T1 = pd.read_parquet(os.path.join(ROOT, "feats", f"tier1__qwen1_5b__{prec}__g3.parquet"))
    D = LAB[LAB.precision == prec].merge(T0, on="uid").merge(T1, on="uid")
    F_EDGE = [c for c in T1.columns if c.startswith("t1q_")]
    F_EDGE_X = F_EDGE + ["t1_ent_q_mean", "t1_ent_q_last", "t1_margin_q_last"]
    y = D.y.values
    arms = {"edge+last (quantized only)": F_EDGE_X,
            "tier0+edge+last (quantized only)": F_T0 + F_EDGE_X}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    for arm, feats in arms.items():
        X = D[feats].values
        aucs = []
        for tr, te in cv.split(X, y):
            c = xgb(y[tr]); c.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
        oof = cross_val_predict(xgb(y), X, y, method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
        per = {t: roc_auc_score(y[(D.task == t).values], oof[(D.task == t).values])
               for t in sorted(D.task.unique())}
        # router: share of quantization loss recovered / FP16 parity at each escalation budget
        order = np.argsort(-oof); route = {}
        for cov in [0.05, 0.10, 0.15, 0.25]:
            k = int(round(cov * len(y))); hit = y[order[:k]].sum()
            route[f"rec@{int(cov*100)}"] = hit / y.sum()
            route[f"parity@{int(cov*100)}"] = 1 - (y.sum() - hit) / len(y)
        rows.append(dict(precision=prec, arm=arm, n_feat=len(feats), auc=np.mean(aucs),
                         sd=np.std(aucs, ddof=1), within_macro=np.mean(list(per.values())),
                         **route, **per))
    for feat, sign in [("t1_margin_q_last", -1), ("t1_ent_q_last", 1)]:
        s = sign * D[feat].values
        per = {t: roc_auc_score(y[(D.task == t).values], s[(D.task == t).values])
               for t in sorted(D.task.unique())}
        rows.append(dict(precision=prec, arm=f"single:{feat}", n_feat=1, auc=roc_auc_score(y, s),
                         sd=np.nan, within_macro=np.mean(list(per.values())), **per))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "results", "extra_edge_last.csv"), index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
