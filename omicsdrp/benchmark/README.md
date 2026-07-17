# Stage-2 benchmark: competitor DRP models on OUR GDSC2

Re-trains published DRP models on the **same GDSC2 data, same folds, same metric**
as OmicsDRP, so Table-3 comparisons are apples-to-apples (리뷰3) and parameter/
efficiency numbers are directly comparable (리뷰11). See
`../../plans/benchmark_feasibility.md` for the reproducibility audit.

## Design (what is held identical across all models)

Every model consumes the **frozen export** (`export_data.py`): the identical
`(cell,drug)` pair universe, the identical `build_folds` fold indices
(mixed / unseen_cell / unseen_drug), the identical **ln(IC50)** labels, and OUR
**909-gene** omics source. Only each model's native featurization + architecture
differ. Scoring uses OmicsDRP's own `metrics.regression_metrics`.

**Leakage boundary (same discipline as OmicsDRP nested-CV):** per fold the omics
scaler — and, for TGSA, the gene-gene correlation graph — are fit on **train cells
only**; early stopping watches **inner-val only** (`val_pair_idx`); the outer-test
fold is evaluated exactly once. (Upstream DeepTTA/DeepCDR leaked the test fold into
early stopping; that is corrected here.)

## Models

| Adapter | Model | Cell input (of our 909 genes) | Drug input | Label | Env |
|---|---|---|---|---|---|
| `adapters/deeptta` | DeepTTA/DeepTTC | RNA (909) via MLP | ESPF/BPE transformer | native ln | omicsdrp |
| `adapters/graphdrp` | GraphDRP | 4-omics flatten (3636) via 1D-CNN | SMILES→GIN graph | `sigmoid(0.1·x)` → inverted `10·logit(y)` | omicsdrp |
| `adapters/tgsa` | TGSA/TGDRP | SNP+CNV+RNA on gene-corr graph (GAT) | SMILES→GIN graph | native ln | omicsdrp |

DeepCDR (TF1.x, no 4090 support) and DRPreter (KEGG-pathway collapse on 909 genes)
were audited and excluded — see `vendor/MANIFEST.md`.

## Environment

All three adapters run in the existing **`omicsdrp`** conda env (no separate envs).
Extra packages (see `requirements-extra.txt`):

```bash
conda activate omicsdrp
pip install pyarrow subword_nmt
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install dgllife
export DGLBACKEND=pytorch     # TGSA drug featurizer
```

## Running

```bash
conda activate omicsdrp
cd omicsdrp/benchmark

# functional check, tiny + fast. --smoke shrinks data/epochs; --device picks the
# device (default cpu → set CUDA_VISIBLE_DEVICES="" for ZERO GPU contention, or
# pass --device cuda when the GPU is free).
CUDA_VISIBLE_DEVICES="" python run_benchmark.py --smoke --splits mixed --device cpu

# detached, resumable, auto-restarting full run (the way to run for real).
# GPU MUST be free (check nvidia-smi; the Stage-1 sweep saturates it).
EMAIL_TO=jmj3078@gmail.com DEVICE=cuda ./run_bench_sweep.sh
#   monitor: cat BenchmarkResults/progress.md ; tail -f BenchmarkResults/sweep_logs/sweep_*.log
#   stop:    kill "$(cat BenchmarkResults/sweep.pid)"   (progress kept; rerun to resume)
#   GPU=1 ./run_bench_sweep.sh --models graphdrp tgsa   # pin a spare GPU / subset
```

**Background stability** (mirrors `scripts/run_sweep.sh`): the sweep runs detached
(survives terminal/SSH close), holds a single-instance lock, auto-restarts on
failure, and is **fold-level resumable** — each adapter skips folds whose outputs
already exist, so a crash/reboot loses no completed work (just rerun the same
command). Failed conditions are logged in `progress.md` and skipped, never
aborting the run. `run_benchmark.py` emails a summary on completion via the
Stage-1 msmtp notifier (`EMAIL_TO` / `--email-to`; best-effort, never blocks).
Use `--overwrite` to force recomputation of already-done folds.

Per model × split × fold the adapters save: `fold_k_model.pt` (weights),
`fold_k_scaler.npz` (fold cell scaler; TGSA also stores the gene edge_index),
`fold_k_predictions.parquet` (`sample_idx,drug_idx,true,pred`), `fold_k_meta.json`
(param count, train/infer seconds, hyper-params). `score.py` aggregates into
`BenchmarkResults/benchmark_summary.csv`.

## Three evaluation regimes (per the review plan)

1. **Nested-CV metric** — `--mode nested --split_mode mixed` (리뷰4).
2. **OOD** — `--mode nested --split_mode unseen_cell|unseen_drug` (리뷰3/8).
3. **External-data ensemble** — `--mode ensemble`: train pool = outer train+val,
   early-stop on outer-test, K per-fold models saved for ensembling on external
   data (mirrors `omicsdrp inference_models.py`). No held-out metric is written.

## Layout

```
benchmark/
  export_data.py     # freeze pairs/folds/omics/SMILES/labels (run in omicsdrp env)
  common.py          # dep-light contract (numpy+sklearn): reader, fold scaler, seed, resume, pred schema
  score.py           # per-pair predictions -> regression_metrics -> summary csv
  run_benchmark.py   # resumable orchestrator (all models x splits x modes) + email summary
  run_bench_sweep.sh # detached, auto-restarting, single-instance wrapper
  adapters/{deeptta,graphdrp,tgsa}/{model.py,run.py}
  vendor/          # unmodified upstream clones (gitignored; see MANIFEST.md)
  export/ export_smoke/ BenchmarkResults/   # generated, gitignored
```
