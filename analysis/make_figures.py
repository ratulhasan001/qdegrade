"""Paper figures (vector PDF, IEEE column widths) built from the saved project artifacts.

  fig_flips.pdf   degradation rate per task, and churn vs. net accuracy change
  fig_within.pdf  within-task ROC-AUC per task for three feature sets, per precision
  fig_router.pdf  recovered share of quantization errors vs. escalation budget

Tier-0 / Tier-0+1 / task-ID numbers come from the notebook's saved out-of-fold scores
(results/oof_*.npy). The quantized-only arm is recomputed here with the notebook's XGBoost
configuration and the same 5-fold split (seed 0), exactly as in extra_edge_last.py.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
SEED = 0

# ---- style: Times-like serif to match IEEEtran, thin marks, recessive axes and grid
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
# categorical slots 1-3 of the validated reference palette, in fixed order
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
PREC_COLOR = {"INT8": C1, "NF4": C2, "FP4": C3}
ARM_COLOR = {"Tier-0": C1, "Q-only": C2, "Tier-0+1": C3}
COL1, COL2 = 3.45, 7.16      # IEEE single / double column width, inches

TASKS = ["gsm8k", "gsm8k_direct", "arc_easy", "arc_challenge", "sst2"]
TASK_LABEL = {"gsm8k": "GSM8K\nCoT", "gsm8k_direct": "GSM8K\ndirect", "arc_easy": "ARC\nEasy",
              "arc_challenge": "ARC\nChall.", "sst2": "SST-2"}
SHORT = {"gsm8k": "CoT", "gsm8k_direct": "Direct", "arc_easy": "ARC-E",
         "arc_challenge": "ARC-C", "sst2": "SST-2"}
PRECS = [("int8", "INT8"), ("nf4", "NF4"), ("fp4", "FP4")]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def xgb(y):
    spw = float((len(y) - y.sum()) / y.sum())
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.85,
                         colsample_bytree=0.85, reg_lambda=1.5, eval_metric="logloss",
                         n_jobs=4, random_state=SEED, scale_pos_weight=spw)


# ---------------------------------------------------------------- data
LAB = pd.read_parquet(os.path.join(ROOT, "feats", "labels.parquet"))
T0 = pd.read_parquet(os.path.join(ROOT, "feats", "tier0.parquet"))
F_T0 = [c for c in T0.columns if c.startswith("t0_")]
DATA = {}
for prec, _ in PRECS:
    T1 = pd.read_parquet(os.path.join(ROOT, "feats", f"tier1__qwen1_5b__{prec}__g3.parquet"))
    D = LAB[(LAB.model == "qwen1_5b") & (LAB.precision == prec)].merge(T0, on="uid") \
                                                                  .merge(T1, on="uid")
    arms = json.load(open(os.path.join(ROOT, "results", f"oof_{prec}_arms.json")))
    oof = dict(zip(arms, np.load(os.path.join(ROOT, "results", f"oof_{prec}.npy"))))
    f_q = [c for c in T1.columns if c.startswith("t1q_")] + \
          ["t1_ent_q_mean", "t1_ent_q_last", "t1_margin_q_last"]
    y = D.y.values
    q_only = cross_val_predict(xgb(y), D[F_T0 + f_q].values, y, method="predict_proba",
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
    DATA[prec] = dict(D=D, y=y, scores={"Task-ID only": oof["task_id_only"],
                                        "Tier-0": oof["tier0"], "Q-only": q_only,
                                        "Tier-0+1": oof["tier0+tier1"]})

# ---------------------------------------------------------------- Fig: flips
F = pd.read_csv(os.path.join(ROOT, "results", "flip_decomposition.csv"))
fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw=dict(width_ratios=[1.35, 1]))
w = 0.26
x = np.arange(len(TASKS))
for i, (prec, lab) in enumerate(PRECS):
    v = [100 * F[(F.task == t) & (F.precision == prec)].degrade_rate.iloc[0] for t in TASKS]
    a.bar(x + (i - 1) * w, v, w - 0.03, color=PREC_COLOR[lab], label=lab, linewidth=0)
a.set_xticks(x, [SHORT[t] for t in TASKS], fontsize=6.3, rotation=30, ha="right",
             rotation_mode="anchor")
a.set_ylabel("degradation rate $\\pi$ (%)")
a.set_ylim(0, 75)
a.legend(loc="upper right", ncol=1, handlelength=1, handletextpad=0.4, borderaxespad=0.1)
a.set_title("(a) Broken answers per task", loc="left")
a.tick_params(axis="x", length=0)

for prec, lab in PRECS:
    g = F[F.precision == prec]
    b.scatter(100 * g.net_delta.abs(), 100 * g.churn, s=16, color=PREC_COLOR[lab],
              edgecolor="white", linewidth=0.6, zorder=3, label=lab)
lim = 40
b.plot([0, lim], [0, lim], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
b.text(20.5, 15.5, "churn = |net|", rotation=38, fontsize=6.3, color=INK2, ha="center")
b.set_xlim(-1, 25); b.set_ylim(-1, 42)
b.set_xlabel("|net accuracy change| (pp)")
b.set_ylabel("churn $\\kappa$ (pp)")
b.grid(axis="both")
r = F[(F.task == "gsm8k") & (F.precision == "int8")].iloc[0]
b.annotate("INT8 GSM8K\nnet 0.2, churn 14.8", (100 * abs(r.net_delta), 100 * r.churn),
           xytext=(1.0, 31), fontsize=6.3, color=INK,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
b.set_title("(b) Accuracy hides flips", loc="left")
fig.tight_layout(w_pad=0.6)
save(fig, "fig_flips.pdf")

# ---------------------------------------------------------------- Fig: within-task AUC
fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.72), sharey=True)
groups = TASKS + ["macro"]
xg = np.arange(len(groups))
w = 0.26
for ax, (prec, lab) in zip(axes, PRECS):
    D, y = DATA[prec]["D"], DATA[prec]["y"]
    for i, arm in enumerate(["Tier-0", "Q-only", "Tier-0+1"]):
        s = DATA[prec]["scores"][arm]
        per = [roc_auc_score(y[(D.task == t).values], s[(D.task == t).values]) for t in TASKS]
        vals = per + [np.mean(per)]
        ax.bar(xg + (i - 1) * w, vals, w - 0.03, color=ARM_COLOR[arm], label=arm,
               linewidth=0, bottom=0)
        if arm == "Tier-0+1":
            ax.text(xg[-1] + (i - 1) * w, vals[-1] + 0.012, f"{vals[-1]:.2f}", ha="center",
                    fontsize=6, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.axvspan(len(TASKS) - 0.5, len(TASKS) + 0.5, color="#f1f0ec", zorder=0, lw=0)
    ax.set_xticks(xg, [TASK_LABEL.get(t, "Macro") for t in groups], fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05)
    ax.set_title(lab, loc="left")
axes[0].set_ylabel("within-task ROC-AUC")
axes[2].legend(loc="lower right", ncol=3, handlelength=1, handletextpad=0.4,
               columnspacing=0.9, bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
fig.tight_layout(w_pad=0.8)
save(fig, "fig_within.pdf")

# ---------------------------------------------------------------- Fig: router
fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.85), sharey=True)
betas = np.linspace(0, 0.5, 101)
style = {"Task-ID only": dict(color="#9a9893", lw=1.0),
         "Tier-0": dict(color=C1, lw=1.4), "Q-only": dict(color=C2, lw=1.4),
         "Tier-0+1": dict(color=C3, lw=1.6)}
for ax, (prec, lab) in zip(axes, [("nf4", "NF4"), ("fp4", "FP4")]):
    y = DATA[prec]["y"]; n, P = len(y), y.sum()
    rng = np.random.RandomState(0)
    for arm, sty in style.items():
        s = DATA[prec]["scores"][arm]
        order = np.lexsort((rng.rand(n), -s))           # random tie-break (task-ID)
        csum = np.concatenate([[0], np.cumsum(y[order])])
        rec = [csum[int(round(bt * n))] / P for bt in betas]
        ax.plot(100 * betas, rec, label=arm, **sty)
    ax.plot(100 * betas, np.minimum(1, betas / (P / n)), color=INK, lw=0.8,
            ls=(0, (4, 2)), label="Oracle")
    ax.plot(100 * betas, betas, color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="Random")
    ax.set_xlim(0, 50); ax.set_ylim(0, 1.02)
    ax.set_xlabel("escalated to FP16, $\\beta$ (%)")
    ax.set_title(lab, loc="left")
axes[0].set_ylabel("errors recovered $\\rho(\\beta)$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.53, -0.18),
           handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(w_pad=0.5)
save(fig, "fig_router.pdf")

# ---------------------------------------------------------------- Fig: where the signal lives
E1 = pd.read_csv(os.path.join(ROOT, "results", "extra_last_position.csv"))
E2 = pd.read_csv(os.path.join(ROOT, "results", "extra_edge_last.csv"))
E = pd.concat([E1, E2], ignore_index=True)
ROWS = [("tier0+tier1", "Tier-0+1, all 73 features"),
        ("tier0+tier1 (no last-pos.)", "Tier-0+1 without last position"),
        ("single:t1_kl_last", "KL at last position only"),
        ("single:t1_kl_mean", "KL averaged over prompt only"),
        ("single:t1_ent_q_last", "Q: entropy $H(q_T)$ only"),
        ("tier0+edge+last (quantized only)", "Q: Tier-0 + edge + last pos.")]
fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.95), sharey=True)
yy = np.arange(len(ROWS))[::-1]
for ax, col, ttl in zip(axes, ["auc", "within_macro"], ["(a) pooled AUC", "(b) within-task macro AUC"]):
    for j, (prec, lab) in enumerate(PRECS):
        v = [E[(E.precision == prec) & (E.arm == k)][col].iloc[0] for k, _ in ROWS]
        ax.scatter(v, yy + (1 - j) * 0.17, s=14, color=PREC_COLOR[lab], edgecolor="white",
                   linewidth=0.5, zorder=3, label=lab)
    ax.axvline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(1.5, color=GRID, lw=0.8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title(ttl, loc="left", fontsize=7.5)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(yy, [r for _, r in ROWS], fontsize=6.6)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.62, -0.07),
           handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(w_pad=0.4)
save(fig, "fig_signal.pdf")

# ---------------------------------------------------------------- Fig: cross-precision transfer
from matplotlib.colors import LinearSegmentedColormap
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec",
                                                    "#3987e5", "#256abf", "#184f95", "#0d366b"])
rows = []
for line in open(os.path.join(ROOT, "tables", "T11_cross_precision.tex")):
    parts = [p.strip() for p in line.replace("\\\\", "").split("&")]
    if len(parts) == 5 and parts[0] in ("nf4", "fp4", "int8"):
        rows.append(dict(train=parts[0], test=parts[1], arm=parts[2], auc=float(parts[3])))
X = pd.DataFrame(rows)
order = ["int8", "nf4", "fp4"]; names = ["INT8", "NF4", "FP4"]
fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.55))
for ax, arm, ttl in zip(axes, ["tier0", "tier0+tier1"], ["(a) Tier-0 (text)", "(b) Tier-0+1 (prefill)"]):
    M = X[X.arm == arm].pivot(index="train", columns="test", values="auc").loc[order, order]
    ax.imshow(M.values, cmap=BLUES, vmin=0.6, vmax=1.0)
    for i in range(3):
        for j in range(3):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.835 else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("tested on"); ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(ttl, loc="left", fontsize=7.5)
axes[0].set_ylabel("trained on")
fig.tight_layout(w_pad=1.2)
save(fig, "fig_transfer.pdf")
