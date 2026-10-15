"""Supplementary analysis for the paper: how much of the Tier-1 signal comes from
the final prompt position (the distribution over the first generated token)?

Uses only saved artifacts (feats/), the same XGBoost configuration as the notebook,
and the same 5x5 RepeatedStratifiedKFold (seed 0). Writes results/extra_last_position.csv.
"""
import os, sys
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
    F_T1 = [c for c in T1.columns if c.startswith("t1")]
    F_T1_nolast = [c for c in F_T1 if not c.endswith("_last")]
    y = D.y.values
    arms = {"tier0+tier1": F_T0 + F_T1, "tier0+tier1 (no last-pos.)": F_T0 + F_T1_nolast}
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
        rows.append(dict(precision=prec, arm=arm, n_feat=len(feats), auc=np.mean(aucs),
                         sd=np.std(aucs, ddof=1), within_macro=np.mean(list(per.values())), **per))
    # single-feature, model-free scores
    for feat, sign in [("t1_top1_agree_last", -1), ("t1_kl_last", 1), ("t1_kl_mean", 1)]:
        s = sign * D[feat].values
        per = {t: roc_auc_score(y[(D.task == t).values], s[(D.task == t).values])
               for t in sorted(D.task.unique())}
        rows.append(dict(precision=prec, arm=f"single:{feat}", n_feat=1,
                         auc=roc_auc_score(y, s), sd=np.nan,
                         within_macro=np.mean(list(per.values())), **per))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "results", "extra_last_position.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
