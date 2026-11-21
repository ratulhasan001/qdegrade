# QDEGRADE: Predicting Per-Prompt Quantization Degradation in Edge LLMs

Research code that asks whether you can tell, before a quantized language model generates anything, which prompts it will get wrong that the full-precision model gets right. If you can, an edge device could send only those risky prompts to a full-precision (or cloud) model and answer the rest locally.

The experiment uses Qwen2.5-1.5B-Instruct at four precisions (fp16 reference, and int8, nf4 and fp4 from `bitsandbytes`) on 5,372 prompts from five tasks: GSM8K with step-by-step reasoning, the same GSM8K problems with a direct answer only, ARC-Easy, ARC-Challenge and SST-2. The repository holds the notebook, the outputs of the final run (generation tag `g3`), the trained router models and four follow-up analysis scripts.

## Contents

| Path | What it is |
|---|---|
| [qdegrade.ipynb](qdegrade.ipynb) | The full pipeline, executed on Google Colab (GPU), with its outputs: generation, labels, features, classifiers, confound controls, router simulation, cost, figures and LaTeX tables. |
| [analysis/](analysis) | Follow-up scripts that reuse the saved features and out-of-fold scores (no GPU needed): `extra_last_position.py`, `extra_edge_last.py`, `router_exact.py` and `make_figures.py`, plus the figures the last one writes (`analysis/figs/`). |
| [runs/](runs) | `sweep_qwen1_5b.parquet`: correctness of every prompt at every precision (`uid, task, gold, correct, precision, model, run_tag`); `sweep_qwen1_5b_pilot.parquet`: the same for the 100-prompt pilot. |
| [feats/](feats) | Degradation labels (`labels.parquet`), Tier-0 text features (`tier0.parquet`) and Tier-1 prefill features per precision (`tier1__qwen1_5b__<precision>__g3.parquet`). |
| [models/](models) | Trained XGBoost routers per precision (Tier-0 only and Tier-0 + Tier-1) as XGBoost JSON, each with its feature list. |
| [results/](results) | `summary.json`, cross-validation scores (`cv_<precision>.csv`), out-of-fold predictions (`oof_<precision>.npy` with arm names in `oof_<precision>_arms.json`), flip decomposition, pilot gate, truncation check, cost, and the outputs of the analysis scripts. |
| [tables/](tables) | LaTeX tables T0–T12 written by the notebook. |
| [figs/](figs) | Figures F1–F16 written by the notebook (PNG and PDF), and two methodology overview figures (F0, PNG/PDF/SVG). |
| [datasets/](datasets) | The GSM8K, GSM8K-direct, ARC-Easy and ARC-Challenge prompts used in the run (`id, question, correct_answer, full_prompt_given_to_model`). SST-2 is not included (see [Data and licences](#data-and-licences)). |

## Pipeline (notebook sections)

1. **Datasets (§1).** Loads `openai/gsm8k` (test), `allenai/ai2_arc` (ARC-Easy and ARC-Challenge, test) and `stanfordnlp/sst2` (validation) with Hugging Face `datasets`, shuffles with seed 0 and keeps 1,000 GSM8K problems (used for both GSM8K tasks), 1,500 ARC-Easy, 1,000 ARC-Challenge and all 872 SST-2 items. Each item is wrapped in a fixed task instruction. Exports the prompt set and an example table.
2. **Scoring (§2).** Exact match on the extracted answer (`#### <number>` for GSM8K, the option letter for ARC, the label word for SST-2).
3. **Model loading and generation (§3–5, §7).** Greedy decoding at each precision with per-task token budgets (512 tokens for GSM8K with reasoning, 6–16 for the rest). fp16 is run twice with different batching to measure GPU non-determinism. Every run is cached as a parquet file, so a dropped Colab session can resume.
4. **Go/no-go gate (§6).** A 100-prompt pilot per task checks that fp16 accuracy is non-trivial and that degradation is well above the fp16-vs-fp16 noise floor.
5. **Labels (§8).** On prompts the fp16 model answers correctly, `y = 1` if the quantized model gets it wrong. Also a flip decomposition (degraded vs recovered prompts per task) and a length-versus-content comparison (GSM8K with and without reasoning).
6. **Features (§9–10).** *Tier-0*: text statistics computed on the CPU before inference (length, lexical, numeric, structural and arithmetic features). *Tier-1*: one forward pass over the prompt with both the fp16 and the quantized model, giving the exact KL and JS divergence between their next-token distributions, top-1 agreement, entropies and margins; `t1q_` features use the quantized model alone.
7. **Classifiers and controls (§11–14).** XGBoost and other arms under 5 × 5 repeated stratified cross-validation (seed 0), compared on identical folds with Wilcoxon tests. Controls: a task-ID-only classifier, a length rule, within-task AUC, leave-one-task-out AUC, and an fp16-vs-fp16 negative control. Also a pooled model across precisions, a consistent-label variant and a feature-group ablation.
8. **Router simulation and cost (§15–16).** Escalates the highest-risk prompts to fp16 and reports how much of the fp16 accuracy is kept and how much of the quantization loss is recovered at each escalation budget. Measures router overhead, prefill latency and peak VRAM.
9. **Outputs (§17–20).** Figures, cross-precision transfer (train on one precision, test on another), LaTeX tables and `summary.json`.

## Results (final run)

| | nf4 | fp4 | int8 |
|---|---|---|---|
| fp16-correct prompts that degrade (n = 3,625) | 11.9 % | 17.2 % | 3.6 % |
| Tier-0 AUC | 0.695 | 0.741 | 0.724 |
| Task-ID-only control AUC | 0.722 | 0.760 | 0.763 |
| Within-task macro AUC, Tier-0 | 0.536 | 0.505 | 0.552 |
| Within-task macro AUC, Tier-0 + Tier-1 | 0.774 | 0.822 | 0.724 |
| Loss recovered when 15 % of prompts are escalated, Tier-0 + Tier-1 | 46 % | 53 % | 40 % |

- Text-only (Tier-0) features did not beat a classifier that only knows which task a prompt comes from, and within each task they were close to chance. Most of their signal is task identity.
- Tier-1 features from a single prefill pass carried real within-task signal on ARC and SST-2, but little on GSM8K (within-task AUC 0.50–0.58).
- The fp16 model disagreed with itself on 9 of 3,625 prompts (0.25 %), so decoding was effectively deterministic.
- The Tier-0 router costs about 1.1 ms per prompt. Tier-1 needs a prefill with both models (35 ms in fp16 and 57–179 ms quantized on the Colab GPU used).

Per-task accuracies, all arms and confidence intervals are in `results/` and `tables/`.

## Data and licences

| Source | Licence | In this repository |
|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT | `datasets/gsm8k.csv`, `datasets/gsm8k_direct.csv` |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) (Allen Institute for AI) | CC BY-SA 4.0 | `datasets/arc_easy.csv`, `datasets/arc_challenge.csv`; these derived files are shared under CC BY-SA 4.0 |
| [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) | not stated ("unknown" on Hugging Face) | Not included |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache 2.0 | Downloaded by the notebook |

The SST-2 prompts, the full prompt export (`results/prompt_set.csv`, `runs/prompts.parquet`) and the raw generations (`runs/qwen1_5b__<precision>__<task>__<run>__g3.parquet`, which contain the prompt text and the model outputs) are left out. Running §1 of the notebook rebuilds the prompt set from Hugging Face (same dataset splits, seed 0), and §5 and §7 regenerate the outputs on a GPU. The notebook builds its labels from those generations, so rerunning it needs them; the analysis scripts need only the labels, features and out-of-fold scores included here.

## How to run

**Notebook.** Open `qdegrade.ipynb` in Google Colab with a GPU runtime (the final run used an A100; the notebook scales batch sizes to the GPU memory). The first cell installs `transformers`, `accelerate`, `bitsandbytes`, `datasets`, `xgboost` and `pyarrow`. Settings are in the configuration cell (§0):

- `SMOKE = True` runs the whole analysis on synthetic data without a GPU or downloads in about two minutes. Use it to check the pipeline first.
- `USE_DRIVE = True` stores everything in `/content/drive/MyDrive/qdegrade/` so that runs can resume. With `USE_DRIVE = False` the output goes to `./qdegrade/`.
- The output folder has the same layout as this repository (`runs/`, `feats/`, `models/`, `results/`, `tables/`, `figs/`). Copy the folders from here into it to reuse the saved results.

**Analysis scripts.** They read `feats/` and `results/` relative to the repository root and need `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost` and `matplotlib`:

```bash
python analysis/router_exact.py          # writes results/router_exact.csv
python analysis/extra_last_position.py   # writes results/extra_last_position.csv
python analysis/extra_edge_last.py       # writes results/extra_edge_last.csv
python analysis/make_figures.py          # writes analysis/figs/*.pdf
```

Run `make_figures.py` last, because it reads the CSV files written by the other two `extra_*` scripts.

## Authors

Ratul Hasan, with co-authors Samiha Farjana and Umme Habiba.

Ratul Hasan · [ratulhasan.dev](https://ratulhasan.dev)
