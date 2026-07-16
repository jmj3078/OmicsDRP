#!/usr/bin/env python
"""Freeze the shared benchmark inputs + fold indices to disk.

Run ONCE in the ``omicsdrp`` conda env. It loads our GDSC2 data exactly as the
OmicsDRP nested-CV does, builds the SAME folds via ``build_folds`` for every
split regime, and writes a self-contained export directory that each competitor
adapter reads (from its own isolated env) through ``common.BenchmarkExport``.

This is what guarantees the fair comparison: every model trains and is evaluated
on the identical (cell,drug) pair universe, identical fold membership, identical
ln(IC50) labels, and the identical 909-gene omics source.

    conda activate omicsdrp
    cd omicsdrp/benchmark
    python export_data.py --dataset_path ../../data --out ./export
    python export_data.py --smoke              # tiny: 1 fold per mode, sanity checks
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.config import ExperimentConfig, OMICS_ORDER   # noqa: E402
from omicsdrp.data import load_raw, stack_gene_data          # noqa: E402
from omicsdrp.splits import build_folds                      # noqa: E402

SPLIT_MODES = ("mixed", "unseen_cell", "unseen_drug")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out", default="./export")
    p.add_argument("--split_modes", nargs="+", default=list(SPLIT_MODES),
                   choices=list(SPLIT_MODES))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--n_cluster_cell", type=int, default=6)
    p.add_argument("--n_cluster_drug", type=int, default=8)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--smoke", action="store_true",
                   help="export only fold 0 of each mode + run sanity assertions")
    return p.parse_args()


def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    print(f"[export] loading data from {a.dataset_path} ...")
    raw = load_raw(a.dataset_path, merge_duplicates=True)
    # cell row labels (SANGER ids) straight from the IC50 matrix index; rows are
    # untouched by the duplicate-drug merge (that only collapses columns).
    ic50_index = pd.read_csv(
        f"{a.dataset_path}/IC50_GDSC2.csv", index_col=0).index.astype(str).tolist()
    assert len(ic50_index) == raw.n_cell, "cell label count != n_cell"

    # --- shared, model-agnostic arrays -------------------------------------
    omics = stack_gene_data(raw.gene_data, raw.genes).cpu().numpy().astype("float32")
    assert omics.shape == (raw.n_cell, len(raw.genes), len(OMICS_ORDER)), omics.shape
    labels = raw.ic50.cpu().numpy()[raw.pairs[:, 0], raw.pairs[:, 1]].astype("float32")
    assert np.isfinite(labels).all(), "pair labels contain NaN/inf (should be pre-filtered)"

    smiles = raw.drug_meta["SMILE"].astype(str).tolist()
    drug_ids = raw.drug_meta["DRUG_ID"].astype(str).tolist()
    assert len(smiles) == raw.n_drug == len(drug_ids), "drug_meta not aligned to drug index"

    np.save(os.path.join(out, "pairs.npy"), raw.pairs.astype(np.int64))
    np.save(os.path.join(out, "labels.npy"), labels)
    np.save(os.path.join(out, "omics.npy"), omics)
    _write_json(os.path.join(out, "smiles.json"), smiles)
    _write_json(os.path.join(out, "drug_ids.json"), drug_ids)
    _write_json(os.path.join(out, "cell_labels.json"), ic50_index)

    # --- folds per split mode (reuse build_folds verbatim) -----------------
    fold_summary = {}
    for mode in a.split_modes:
        cfg = ExperimentConfig(
            name=f"benchmark_export_{mode}",
            omics=list(OMICS_ORDER), cell_encoder="attention", drug_encoder="morgan",
            split_mode=mode, outer_folds=a.outer_folds, inner_val_frac=a.inner_val_frac,
            seed=a.seed, n_cluster_cell=a.n_cluster_cell, n_cluster_drug=a.n_cluster_drug,
        )
        folds = build_folds(raw, cfg)
        mode_dir = os.path.join(out, "folds", mode)
        os.makedirs(mode_dir, exist_ok=True)
        kept = folds[:1] if a.smoke else folds
        for fs in kept:
            np.savez(
                os.path.join(mode_dir, f"fold_{fs.fold}.npz"),
                train_pair_idx=fs.train_pair_idx.astype(np.int64),
                val_pair_idx=fs.val_pair_idx.astype(np.int64),
                test_pair_idx=fs.test_pair_idx.astype(np.int64),
                train_sample_indices=fs.train_sample_indices.astype(np.int64),
            )
        fold_summary[mode] = [
            {"fold": fs.fold, "n_train": int(fs.train_pair_idx.size),
             "n_val": int(fs.val_pair_idx.size), "n_test": int(fs.test_pair_idx.size),
             "meta": fs.meta}
            for fs in kept
        ]
        # sanity: outer test folds partition the pair universe with no leakage
        _assert_disjoint(kept)
        print(f"[export] {mode}: {len(kept)} fold(s) written")

    meta = {
        "n_cell": raw.n_cell, "n_drug": raw.n_drug, "n_pairs": int(raw.pairs.shape[0]),
        "genes": list(raw.genes), "omics_order": list(OMICS_ORDER),
        "split_modes": list(a.split_modes), "smoke": bool(a.smoke),
        "config": {"seed": a.seed, "outer_folds": a.outer_folds,
                   "inner_val_frac": a.inner_val_frac,
                   "n_cluster_cell": a.n_cluster_cell, "n_cluster_drug": a.n_cluster_drug},
        "folds": fold_summary,
    }
    _write_json(os.path.join(out, "meta.json"), meta)
    print(f"[export] done -> {out}  "
          f"({raw.n_cell} cells x {raw.n_drug} drugs, {raw.pairs.shape[0]} pairs)")


def _assert_disjoint(folds):
    """train/val/test pair indices must be mutually exclusive within a fold."""
    for fs in folds:
        tr, va, te = set(fs.train_pair_idx.tolist()), set(fs.val_pair_idx.tolist()), \
            set(fs.test_pair_idx.tolist())
        assert not (tr & va), f"fold {fs.fold}: train/val overlap"
        assert not (tr & te), f"fold {fs.fold}: train/test overlap"
        assert not (va & te), f"fold {fs.fold}: val/test overlap"


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


if __name__ == "__main__":
    main()
