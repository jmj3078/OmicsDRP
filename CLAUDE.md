# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

OmicsDRP is a dual-branch deep-learning model that predicts drug response from
cellular multi-omics profiles + drug structure, trained/evaluated on GDSC2.

Two code trees coexist — do not conflate them:

- **`code/`** — the *original* paper implementation (`Train_5fold_CV.py`,
  `Save_Results.py`, `Model.py`, `Utils.py`). Kept **untouched** for reference.
- **`omicsdrp/`** — a full **Stage-1 review refactor** (`omicsdrp/src/omicsdrp/`
  package + `omicsdrp/scripts/`). **This is where all current work happens.** It
  reimplements training as one configurable framework covering every Stage-1
  item in `plans/Review_plan.md`. See `omicsdrp/README.md` for the long-form
  walkthrough.

## Environment

All commands run in the **`omicsdrp`** conda env (built from `README.md` deps:
python 3.9.19, torch 2.3.0+cu121, pandas 1.5.3, numpy 1.26.4, scipy 1.13.1,
sklearn 1.5.2; GPU = RTX 4090). The env additionally has rdkit + torch-geometric
(for GIN/GCN). Always `conda activate omicsdrp` before running anything.

Do **not** reuse `pytorch_env` — the user explicitly wants this isolated env.

## Common commands

All scripts are run from `omicsdrp/scripts/`, dataset lives at `../../data`:

```bash
conda activate omicsdrp
cd omicsdrp/scripts

# Fast end-to-end sanity check (tiny epochs/folds) — use this as the "test":
python run_stage1.py --smoke

# Staged Stage-1 run in order (BASELINE→FEATURE→ATTENTION→DRUG), live progress + email:
python run_stage1.py --dataset_path ../../data --out_root ./Results \
    --email-to jmj3078@gmail.com          # --email-per stage|experiment|none
python run_stage1.py --split_mode unseen_drug    # run the whole ladder unseen

# OFAT ablation grid (alternative to staged):
python run_ablations.py --groups split --smoke --dataset_path ../../data
python run_ablations.py --dataset_path ../../data --out_root ./Results

# Cluster-diagnostic plots to choose the unseen-split threshold k (user decides):
python inspect_clusters.py --target both --k_min 2 --k_max 30 \
    --chosen_k_cell 6 --chosen_k_drug 8 --dataset_path ../../data

# Deployable ENSEMBLE inference models (SEPARATE track from nested CV, own folder):
python train_inference_models.py --dataset_path ../../data \
    --out_root ./InferenceModels --email-to jmj3078@gmail.com   # 12 mixed conditions
python train_inference_models.py --smoke                        # fast end-to-end check
RUNNER=train_inference_models.py OUT_ROOT=./InferenceModels \
    EMAIL_TO=jmj3078@gmail.com ./run_sweep.sh --email-per condition   # detached/resumable

# Long, crash/reboot-resilient detached sweep (auto-restart, auto-resume):
EMAIL_TO=jmj3078@gmail.com ./run_sweep.sh          # runs run_stage1.py by default
RUNNER=run_ablations.py ./run_sweep.sh --groups drug
GPU=0 ./run_sweep.sh --split_mode unseen_drug
# monitor:  tail -f Results/sweep_logs/sweep_*.log   OR   cat Results/progress.md
# stop:     kill "$(cat Results/sweep.pid)"          (progress kept; rerun to resume)
```

There is no unit-test suite or linter configured; `--smoke` is the standard
end-to-end verification path.

## Architecture (omicsdrp package)

The whole framework is driven by a single **`ExperimentConfig`** dataclass
(`config.py`): one config object = one point in the ablation space (omics subset
× cell encoder × drug encoder × split mode × hyperparams). Adding an ablated
variant means adding a config, never touching the engine.

Data flow: `data.load_raw` → `RawData` → `nested_cv.run_nested_cv` → per-fold
`run_fold` → `DRPModel` (cell encoder + drug encoder + response head) →
`recorder` streams everything to disk.

Key modules:
- **`config.py`** — `ExperimentConfig`; `OMICS_ORDER = [SNP, MET, CNV, RNA]` is
  fixed by preprocessing. `DRUG_ENCODERS` registry (7) + `DRUG_ENCODER_FAMILY`.
- **`data.py`** — loads per-gene omics dict `[N_cell, 4]`, drug meta, IC50 matrix;
  `merge_duplicate_drugs` (data-level dedup, see below); `select_omics` (column
  slice); `scale_gene_data` (StandardScaler **fit on inner-train cells only** —
  the leakage boundary); `stack_gene_data` → one `[N_cell, n_gene, n_omics]`
  tensor; `OmicsDrugDataset` yields cheap tensor slices + indices.
- **`splits.py`** — `mixed` / `unseen_cell` / `unseen_drug`. Unseen splits
  KMeans-cluster the groups and stratify clusters across outer folds so a whole
  similarity-cluster never lands entirely in test. Guardrail warns if
  `min_cluster_size < outer_folds`.
- **`cell_encoders.py`** — `attention` vs `mlp`. `PerGeneLinear` vectorises the
  per-gene embedding via einsum (was a 909-module Python loop — the old
  bottleneck). Uses **LayerNorm** per gene, not BatchNorm.
- **`drug_encoders.py` / `pretrained_drug_encoder.py` / `graph_drug_encoders.py`**
  — `build_drug_encoder` routes to Morgan (baseline), frozen pretrained
  embeddings (chemberta/molformer/graphormer/unimol), or end-to-end GIN/GCN.
- **`nested_cv.py`** — the core CV fix (see below).
- **`inference_models.py`** — SEPARATE track from `nested_cv` for producing
  *deployable* models. Plain 5-fold CV where the held-out fold is used as the
  early-stopping set (reuses `build_folds`: outer train+val → training pool,
  outer-test → early-stopping fold); trains K models per condition and saves each
  with its **per-fold gene scaler** (mean/scale) to a **separate folder**
  (`InferenceModels/<tag>/fold_k.pt`, never `Results/`). `InferenceEnsemble.load`
  → `.predict()` averages the K models on new data (each fold applies its own
  saved scaler). These models' fold scores are early-stopping-optimistic and are
  **NOT** performance numbers — report nested-CV for that; use these only for
  inference on genuinely external data. Runner: `scripts/train_inference_models.py`
  (12 mixed conditions, idempotent/resumable).
- **`recorder.py`** — append-only `events.jsonl` (fsync'd), rebuilds
  `history.csv`/`summary.json`; fold-completion markers drive resume.
- **`pipeline.py` / `ablations.py`** — staged runner + OFAT grid definitions.
- **`diagnostics.py`** — cluster k-sweep / embedding / OOD-distance plots.
- **`notify.py`** — per-stage email via msmtp (see below).

## Critical invariants — preserve these

**Nested CV metric-bias fix (the whole point of Stage 1).** For each outer fold:
`outer-train → inner-train + inner-val` (inner-val drives early stopping),
`outer-test` is untouched until a **single** final evaluation of the
best-inner-val model. Never report the early-stopping/selection set as the score.
Feature scaling is fit on inner-train cells only. See `nested_cv.py` docstring.

**Leakage boundary.** `scale_gene_data` must only ever fit on training cells; the
cell encoder is batch-independent (LayerNorm, not per-gene BatchNorm) so eval is
not coupled to batch statistics. Any change here must be re-verified for leakage.

**`ExperimentConfig.tag()` is identity-only.** It excludes `name`/`out_root` and
canonicalises omics order via `omics_indices()`, so the baseline that recurs
across stages collapses to one tag → **trained exactly once**, reused via
resume/cache. Do not add cosmetic fields to the hash. NOTE: internal
architecture refactors do **not** change `tag()`, so wipe old `Results/` before
rerunning if the model architecture changed (else resume reuses stale folds).

**Duplicate-drug merge is at the DATA level.** GDSC2 has 10 duplicate-SMILES
pairs among 241 drugs. `merge_duplicate_drugs` (called by `load_raw`) collapses
each to one drug (IC50 = NaN-aware mean, label `name1 (id1)/name2 (id2)`) →
**231 unique drugs**. A `_source_row` column remaps the 241-aligned pretrained
`.npy` tables to 231 at load. There is **no** split-level duplicate grouping —
splits operate on 231 distinct molecules.

**Fair-ablation grid = 5 frozen encoders only** (`morgan` + chemberta/molformer/
graphormer/unimol): all "fixed representation → trained projection". `gin`/`gcn`
are end-to-end (extra trainable capacity) → **excluded** from the grid
(`ablations.ABLATION_DRUG_ENCODERS`) but kept runnable via explicit config.

**Feature ablation is RNA-anchored** (RNA always present so the cell branch never
collapses to one modality): RNA+one ×3, RNA+two ×3, all(=baseline, not re-run).

**`drop_last=True` on the train loader** — Morgan/response-head BatchNorm1d still
crash on a trailing batch of size 1. Keep it.

## Data & outputs

- Dataset at `data/`: `PGKB_Gene_data_dict.pth` (per-gene omics), `IC50_GDSC2.csv`,
  `TargetDrugs_with_MorganFingerprint_GDSC2_512.txt`, `gene_list.txt`.
- Pretrained drug embeddings: `data/drug_embeddings/<model>.npy` float32 `[241, D]`
  in drug-index order (+ `.meta.json` provenance). chemberta D=384, molformer/
  graphormer D=768, unimol D=512. Regenerate via extraction scripts in the
  scratchpad under isolated `drugemb_*` envs (kept out of the training env).
- Per-experiment outputs under `out_root/<tag>/`: `config.json`, `events.jsonl`,
  `history.csv`, `folds/fold_k_*.{parquet,json}`, `summary.json`; plus
  `out_root/ablation_summary.csv` and `Results/progress.{md,json}`.
- **`.gitignore` excludes `*.csv *.json *.jsonl *.log *.pkl`** — most results and
  data artefacts are untracked by design. Don't assume `git status` shows them.

## Email notifications

Per-stage emails go through **msmtp** (Gmail SMTP relay, App Password) — Gmail
rejects unauthenticated direct sends. Config: `~/.msmtprc` (600) +
`~/.omicsdrp_smtp_pass` (600, App Password, user-filled). `notify.py` builds the
MIME message and pipes to `msmtp` directly. **Never** paste the App Password into
chat; the user fills the pass file themselves. `EMAIL_TO` env or `--email-to`
selects the recipient; sending is best-effort and never blocks training.
