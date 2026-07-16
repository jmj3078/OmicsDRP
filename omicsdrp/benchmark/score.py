#!/usr/bin/env python
"""Score every competitor's per-pair predictions with the SAME metric function.

Reads ``BenchmarkResults/<model>/<split_mode>/fold_<k>_predictions.parquet`` (the
unified schema ``sample_idx,drug_idx,true,pred``), scores each fold with
``omicsdrp.metrics.regression_metrics`` -- the identical function OmicsDRP's own
nested-CV uses -- aggregates mean/std across folds, folds in per-fold efficiency
metadata (param count, train/infer seconds) for the 리뷰11 comparison, and writes
``benchmark_summary.csv``.

Run in the ``omicsdrp`` env (only dependency beyond numpy/pandas is the shared
metric function):

    python score.py --results ./BenchmarkResults --out ./BenchmarkResults/benchmark_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from omicsdrp.metrics import regression_metrics   # noqa: E402

_FOLD_RE = re.compile(r"fold_(\d+)_predictions\.(parquet|pkl)$")
_METRICS = ["rmse", "pearson", "r2", "mae", "spearman"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="./BenchmarkResults")
    p.add_argument("--out", default="./BenchmarkResults/benchmark_summary.csv")
    return p.parse_args()


def _read_pred(path: str) -> pd.DataFrame:
    return pd.read_pickle(path) if path.endswith(".pkl") else pd.read_parquet(path)


def _fold_meta(pred_path: str) -> dict:
    """Optional sibling fold_<k>_meta.json with efficiency numbers."""
    m = _FOLD_RE.search(os.path.basename(pred_path))
    meta_path = os.path.join(os.path.dirname(pred_path), f"fold_{m.group(1)}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            return json.load(fh)
    return {}


def main():
    a = parse_args()
    rows = []
    # layout: <results>/<model>/<split_mode>/fold_<k>_predictions.parquet
    pred_paths = sorted(glob.glob(
        os.path.join(a.results, "*", "*", "fold_*_predictions.*")))
    pred_paths = [p for p in pred_paths if _FOLD_RE.search(os.path.basename(p))]
    if not pred_paths:
        print(f"[score] no prediction files under {a.results}")
        return

    # group folds by (model, split_mode)
    groups: dict = {}
    for path in pred_paths:
        split_mode = os.path.basename(os.path.dirname(path))
        model = os.path.basename(os.path.dirname(os.path.dirname(path)))
        groups.setdefault((model, split_mode), []).append(path)

    for (model, split_mode), paths in sorted(groups.items()):
        per_fold, params, train_s, infer_s = [], [], [], []
        for path in sorted(paths):
            df = _read_pred(path)
            m = regression_metrics(df["true"].to_numpy(), df["pred"].to_numpy())
            per_fold.append(m)
            meta = _fold_meta(path)
            if "n_params" in meta:
                params.append(meta["n_params"])
            if "train_sec" in meta:
                train_s.append(meta["train_sec"])
            if "infer_sec" in meta:
                infer_s.append(meta["infer_sec"])

        row = {"model": model, "split_mode": split_mode, "n_folds": len(per_fold)}
        for metric in _METRICS:
            vals = np.array([f[metric] for f in per_fold], dtype=float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
        row["n_params"] = int(np.mean(params)) if params else np.nan
        row["train_sec_mean"] = float(np.mean(train_s)) if train_s else np.nan
        row["infer_sec_mean"] = float(np.mean(infer_s)) if infer_s else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["split_mode", "rmse_mean"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summary.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary.to_string(index=False))
    print(f"\n[score] wrote {a.out}")


if __name__ == "__main__":
    main()
