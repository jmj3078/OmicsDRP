# OmicsDRP — Stage-1 Refactor Skeleton

Full renewal of the training/evaluation code, built to run **all Stage-1 review
items** from `plans/Review_plan.md` from a single configurable framework. The
original scripts under `../code/` are left untouched for reference.

## What this fixes / adds (Stage 1)

| Review item | Where |
|---|---|
| ① CV metric bias → **nested CV (outer-test / inner-val)** | `src/omicsdrp/nested_cv.py` |
| ② Ablation: **omics combination** (2-modality baselines incl.) | `config.omics`, `data.select_omics` |
| ② Ablation: **attention vs. MLP** cell encoder | `src/omicsdrp/cell_encoders.py` |
| ② Ablation: **drug representation** — 1 baseline + 6 candidates (see below) | `src/omicsdrp/drug_encoders.py` |
| ③ **Unseen cell / unseen drug** clustering-stratified splits | `src/omicsdrp/splits.py` |
| ③ **Cluster diagnostics / visualisation** to pick the split threshold | `src/omicsdrp/diagnostics.py`, `scripts/inspect_clusters.py` |
| Extra: **exhaustive logging** of every loss/metric | `src/omicsdrp/recorder.py` |

## Layout
```
src/omicsdrp/
  config.py         ExperimentConfig — one object = one ablated run
  data.py           loading + omics-subset + leakage-safe scaling
  splits.py         mixed / unseen_cell / unseen_drug (cluster-stratified)
  cell_encoders.py  attention vs. mlp
  drug_encoders.py  morgan (+ pretrained/GNN stubs with a fixed interface)
  models.py         DRPModel = cell + drug + head  (param count for complexity)
  engine.py         train/eval loops, early stopping, metrics
  metrics.py        rmse/mae/r2/pearson/spearman
  recorder.py       streams EVERY epoch & eval to events.jsonl + csv/parquet
  nested_cv.py      outer-test / inner-val engine
  experiment.py     run one config
  ablations.py      the Stage-1 OFAT grid
scripts/
  run_ablations.py  sequential sweep runner
```

## Choosing the unseen-split threshold (k)
The unseen-cell / unseen-drug splits cluster the groups and stratify clusters
across folds. **You** pick the number of clusters `k` from the diagnostics:
```bash
cd omicsdrp/scripts
python inspect_clusters.py --target both --k_min 2 --k_max 30 \
    --chosen_k_cell 6 --chosen_k_drug 8 --dataset_path ../../data
```
Then read, under `ClusterDiagnostics/<target>/`:
- `*_k_sweep.png` — inertia (elbow) + silhouette vs. k, with `min_cluster_size`
  in the CSV. **Rule of thumb: keep `min_cluster_size >= outer_folds` (5)** so
  every cluster can be spread across folds; the split builder warns otherwise.
- `*_ood_distance_k{K}.png` — how far each held-out test group is from its
  nearest train group (small = conservative, long tail = genuinely OOD).
- `*_embedding_k{K}.png`, `*_cluster_fold_table.csv`.

Set `n_cluster_cell` / `n_cluster_drug` in the config to your choice. Every
unseen-split experiment also auto-saves these diagnostics under
`Results/<tag>/cluster_diag/` so each result documents its own threshold.

**Duplicate molecules are merged at the data level.** GDSC2 registers 10
molecules twice under different DRUG_IDs (e.g. Docetaxel 1007/1819) with slightly
different per-batch IC50s. `data.merge_duplicate_drugs` (called by `load_raw`)
collapses each to one drug: IC50 = NaN-aware **mean** of the duplicate columns,
label = `name1 (id1)/name2 (id2)`, giving **231 unique drugs**. A `_source_row`
column maps each merged drug to its representative row in the original 241-drug
order, so the pretrained embedding tables (stored 241-aligned) are remapped to
231 automatically at load. Splits therefore operate on 231 distinct molecules
with no duplicate-leakage handling needed.

## Running a long sweep safely (GPU-hours, crash/reboot-resilient)
A full sweep is many GPU-hours. Use the detached driver — it survives terminal/SSH
close, auto-restarts on failure, and **resumes** exactly where it stopped:
```bash
cd omicsdrp/scripts
./run_sweep.sh                 # all groups, detached, auto-resume
./run_sweep.sh --groups drug   # any run_ablations.py args pass through
GPU=0 ./run_sweep.sh           # pin a GPU
# monitor: tail -f Results/sweep_logs/sweep_*.log
# stop:    kill "$(cat Results/sweep.pid)"   (progress kept; rerun to resume)
```
Resilience guarantees:
- **Fold-level resume** — each finished outer fold writes `folds/fold_k_test_metrics.json`; on restart those folds are skipped. A crash mid-experiment loses at most the current fold.
- **Experiment-level skip** — fully-finished experiments return instantly as `cached`.
- **Crash-safe logging** — `events.jsonl` is append-only + fsync'd every event; `history.csv`/`summary.json` are rebuilt from it, so outputs stay complete/correct across any number of resumes.
- **Reproducible** — per-fold seed (`seed + fold`) makes a fold's result identical whether run fresh or on resume.
- **GPU cleanup** between folds/experiments; **single-instance lock** (`Results/sweep.lock`) prevents two sweeps clobbering one output.
- After a power-off, just rerun `./run_sweep.sh` — it continues.

## Run (foreground / quick)
```bash
conda activate omicsdrp            # env built from ../README.md deps

cd omicsdrp/scripts
# quick end-to-end validation:
python run_ablations.py --groups split --smoke --dataset_path ../../data

# real sweep (stubbed drug encoders auto-skip & are recorded):
python run_ablations.py --dataset_path ../../data --out_root ./Results
```

## Outputs (per experiment, under `out_root/<tag>/`)
- `config.json` — exact config
- `events.jsonl` — append-only stream of **every** logged event (crash-safe)
- `history.csv` — per-epoch train & val loss + all metrics
- `folds/fold_k_predictions.parquet`, `fold_k_test_metrics.json`
- `summary.json` — mean/std of unbiased outer-test metrics across folds
- `out_root/ablation_summary.csv` — one row per experiment (checkpointed)

## Drug representation candidates (finalised)
`morgan` is the implemented baseline; the 6 test candidates are stubs with a
fixed interface (auto-skipped until built):

All 7 are **implemented and verified** end-to-end (DRPModel forward/backward +
nested-CV smoke):

| family | encoders | how it's built | status |
|---|---|---|---|
| baseline | `morgan` | 512-bit fingerprint → MLP (current model) | ✅ |
| from-scratch graph | `gin`, `gcn` | 2D molecular graph (rdkit + torch-geometric), trained **end-to-end** | ✅ |
| pretrained LM (frozen) | `chemberta` (D=384), `molformer` (D=768) | SMILES language models; embedding pre-extracted, projection trained | ✅ |
| pretrained graph (frozen) | `graphormer` (D=768), `unimol` (D=512) | graph/3D transformers; **Uni-Mol replaces GROVER** | ✅ |

Pretrained encoders use **frozen** pre-extracted embeddings (only a projection
head trains), stored ready-to-train at `data/drug_embeddings/<model>.npy`
(float32 `[241, D]`, drug-index order) with a `<model>.meta.json` provenance
file. Extraction ran in isolated conda envs (`drugemb_lm`, `drugemb_unimol`,
`drugemb_graphormer`) so heavy/pinned deps never touched the training env; the
encoder auto-activates once its `.npy` exists (`is_implemented`).

Checkpoints: ChemBERTa `DeepChem/ChemBERTa-77M-MLM`, MolFormer
`ibm/MoLFormer-XL-both-10pct`, Graphormer `clefourrier/graphormer-base-pcqm4mv2`
(needs `transformers==4.37.2` + `Cython==0.29.37`), Uni-Mol `unimol_tools`
(`mol_pre_all_h_220816.pt`). GIN/GCN need rdkit + torch-geometric in `omicsdrp`.

Note: the 241 GDSC2 drugs contain 10 duplicate-SMILES pairs; these are **merged at
the data level** into 231 unique drugs (IC50 averaged) — see below. The stored
`.npy` remain 241-aligned and are remapped to 231 at load via `_source_row`.

To (re)generate embeddings, the per-model extraction scripts live in the
scratchpad; rerun in the matching `drugemb_*` env writing to
`data/drug_embeddings/<model>.npy`.

## Extending the stubbed drug encoders
Implement `forward(drug_idx) -> [B, embedding_dim]` in a new subclass of
`BaseDrugEncoder` and register it in `build_drug_encoder`. Required deps per
method are listed in `_StubDrugEncoder.REQUIREMENTS`.
```
