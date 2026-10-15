"""Router operating points at EXACT escalation budgets, recomputed from the saved
out-of-fold scores (results/oof_<prec>.npy). The notebook's T6 table picked the nearest
point of a 2%-step grid, so its nominal 5/15/25% rows are really 6/14/24%.
Writes results/router_exact.csv.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = pd.read_parquet(os.path.join(ROOT, "feats", "labels.parquet"))
T0 = pd.read_parquet(os.path.join(ROOT, "feats", "tier0.parquet"))
rows = []
for prec in ["nf4", "fp4", "int8"]:
    T1 = pd.read_parquet(os.path.join(ROOT, "feats", f"tier1__qwen1_5b__{prec}__g3.parquet"))
    D = LAB[(LAB.model == "qwen1_5b") & (LAB.precision == prec)].merge(T0, on="uid") \
                                                                  .merge(T1, on="uid")
    y = D.y.values; n, P = len(y), y.sum()
    arms = json.load(open(os.path.join(ROOT, "results", f"oof_{prec}_arms.json")))
    oof = dict(zip(arms, np.load(os.path.join(ROOT, "results", f"oof_{prec}.npy"))))
    # alignment check against the notebook's within-task table
    m = (D.task == "arc_challenge").values
    chk = roc_auc_score(y[m], oof["tier0+tier1"][m])
    scores = {k: oof[k] for k in ["task_id_only", "tier0", "tier0+tier1"]}
    rng = np.random.RandomState(0)
    for cov in [0.05, 0.10, 0.15, 0.25]:
        k = int(round(cov * n))
        out = dict(precision=prec, coverage=cov, base_parity=1 - P / n, check_arcc=chk,
                   random_rec=cov, oracle_rec=min(k, P) / P)
        for name, s in scores.items():
            # random tie-breaking (task_id_only is constant within a task)
            order = np.lexsort((rng.rand(n), -s))
            hit = y[order[:k]].sum()
            out[f"{name}_rec"] = hit / P
            out[f"{name}_parity"] = 1 - (P - hit) / n
        rows.append(out)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "results", "router_exact.csv"), index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
