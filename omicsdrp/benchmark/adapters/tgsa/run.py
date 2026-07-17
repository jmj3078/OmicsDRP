#!/usr/bin/env python
"""TGSA/TGDRP benchmark adapter -- train on OUR GDSC2 pairs/folds, our metric.

Same contract as the DeepTTA / GraphDRP adapters (shared export, identical fold
membership, early-stop on inner-val only, single outer-test eval, weights + cell
scaler + gene graph saved per fold). TGSA-specific points:
  * cell branch = OUR 909-gene x 3-omics (SNP->mu, CNV->cn, RNA->exp) on a gene
    graph built from TRAIN-cell expression correlation (leakage-safe; avoids the
    STRING download that upstream get_STRING_graph needs);
  * drug branch = SMILES->graph (GIN, 77-dim dgllife features);
  * label = native ln(IC50), MSE -- no rescale (identical to the others).

Runs in the omicsdrp env (PyG 2.x + torch_cluster + dgl/dgllife). GPU safety:
default CPU; --smoke forces CPU + tiny data.

    CUDA_VISIBLE_DEVICES="" python run.py --export ../../export_smoke \
        --split_mode mixed --smoke --out ../../BenchmarkResults/TGSA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_HERE, _BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import common                                                        # noqa: E402
from model import (build_drug_graphs, build_gene_graph, build_cluster_predefine,  # noqa: E402
                   build_cell_graphs, build_model, TGSA_OMICS, EXPR_MODALITY)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--split_mode", default="mixed",
                   choices=["mixed", "unseen_cell", "unseen_drug"])
    p.add_argument("--mode", default="nested", choices=["nested", "ensemble"])
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--out", default="./BenchmarkResults/TGSA")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--edge_thresh", type=float, default=0.6,
                   help="|Pearson corr| threshold for the gene-gene graph")
    p.add_argument("--layer", type=int, default=3, help="cell GNN layers (== pool levels)")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute folds even if their outputs already exist")
    return p.parse_args()


class PairDataset(Dataset):
    """One (drug_graph, cell_graph, label) per pair; graphs shared by reference."""

    def __init__(self, pair_idx, pairs, drug_graphs, cell_graphs, labels):
        self.cell_rows = pairs[pair_idx, 0]
        self.drug_cols = pairs[pair_idx, 1]
        self.drug_graphs = drug_graphs
        self.cell_graphs = cell_graphs
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.drug_graphs[int(self.drug_cols[i])],
                self.cell_graphs[int(self.cell_rows[i])],
                self.labels[i])


def _collate(samples):
    drugs, cells, labels = map(list, zip(*samples))
    return (Batch.from_data_list(drugs), Batch.from_data_list(cells),
            torch.tensor(labels, dtype=torch.float))


def _run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    losses, preds, n = [], [], 0
    with torch.set_grad_enabled(train):
        for drug, cell, y in loader:
            drug, cell, y = drug.to(device), cell.to(device), y.to(device)
            out = model(drug, cell).view(-1)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            bs = y.numel()
            losses.append(loss.item() * bs); n += bs
            preds.append(out.detach().cpu().numpy())
    return sum(losses) / n, np.concatenate(preds) if preds else np.array([])


def train_one_fold(model, train_ds, val_ds, device, epochs, patience, lr, batch):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=_collate, drop_last=True)
    va = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=_collate)
    best_mse, best_state, best_epoch, stale = float("inf"), None, -1, 0
    for epoch in range(epochs):
        _run_epoch(model, tr, device, opt)
        val_mse, _ = _run_epoch(model, va, device)
        if val_mse < best_mse - 1e-7:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_mse


def main():
    a = parse_args()
    if a.smoke:   # tiny data/epochs; device is governed by --device (default cpu)
        a.epochs = min(a.epochs, 2)
        a.max_pairs = a.max_pairs or 384
        a.patience = min(a.patience, 2)
        a.batch = min(a.batch, 32)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    device = a.device

    exp = common.BenchmarkExport(a.export)
    drug_graphs = build_drug_graphs(exp.smiles)                     # 231 drug graphs
    omics_sel = common.select_omics_matrix(exp.omics, TGSA_OMICS)   # [N, n_gene, 3]
    n_cell, n_gene, n_om = omics_sel.shape
    # select_omics_matrix returns columns in SORTED omics-index order, so locate
    # the expression column by its sorted position (robust to TGSA_OMICS ordering).
    _sorted_idx = sorted(common.OMICS_TO_INDEX[o] for o in TGSA_OMICS)
    expr_idx = _sorted_idx.index(common.OMICS_TO_INDEX[EXPR_MODALITY])

    folds = a.folds or exp.list_folds(a.split_mode)
    out_dir = os.path.join(a.out, a.split_mode if a.mode == "nested" else "ensemble")
    os.makedirs(out_dir, exist_ok=True)

    for k in folds:
        if not a.overwrite and common.fold_outputs_exist(out_dir, k, a.mode):
            print(f"[TGSA] {a.split_mode}/{a.mode} fold {k}: done, skipping")
            continue
        common.set_seed(a.seed + k)   # per-fold seed, identical to omicsdrp run_fold
        f = exp.load_fold(a.split_mode, k)
        if a.mode == "nested":
            train_idx, val_idx, eval_idx = (
                f["train_pair_idx"], f["val_pair_idx"], f["test_pair_idx"])
        else:
            train_idx = np.concatenate([f["train_pair_idx"], f["val_pair_idx"]])
            val_idx, eval_idx = f["test_pair_idx"], None

        if a.max_pairs:
            rng = np.random.RandomState(a.seed)
            train_idx = rng.permutation(train_idx)[:a.max_pairs]
            val_idx = rng.permutation(val_idx)[:max(64, a.max_pairs // 4)]
            if eval_idx is not None:
                eval_idx = rng.permutation(eval_idx)[:max(64, a.max_pairs // 4)]

        # scaler + gene graph fit on TRAIN cells only (leakage boundary)
        train_cells = np.asarray(sorted(set(exp.pairs[train_idx, 0].tolist())))
        flat_raw = omics_sel.reshape(n_cell, n_gene * n_om)
        scaler = StandardScaler().fit(flat_raw[train_cells])
        omics_scaled = scaler.transform(flat_raw).reshape(n_cell, n_gene, n_om).astype(np.float32)

        edge_index = build_gene_graph(omics_sel[train_cells, :, expr_idx], a.edge_thresh)
        cluster_predefine = build_cluster_predefine(edge_index, n_gene, a.layer, device)
        cell_graphs = build_cell_graphs(omics_scaled, edge_index)

        model = build_model(num_feature=n_om, cluster_predefine=cluster_predefine,
                            layer=a.layer, batch_size=a.batch)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        def ds(idx):
            return PairDataset(idx, exp.pairs, drug_graphs, cell_graphs, exp.labels[idx])

        t0 = time.time()
        best_epoch, best_val = train_one_fold(
            model, ds(train_idx), ds(val_idx), device,
            a.epochs, a.patience, a.lr, a.batch)
        train_sec = time.time() - t0

        torch.save(model.state_dict(), os.path.join(out_dir, f"fold_{k}_model.pt"))
        np.savez(os.path.join(out_dir, f"fold_{k}_scaler.npz"),
                 mean=scaler.mean_, scale=scaler.scale_, genes=np.array(exp.genes),
                 modality="+".join(TGSA_OMICS), edge_index=edge_index,
                 edge_thresh=a.edge_thresh)

        infer_sec = np.nan
        if eval_idx is not None:
            t1 = time.time()
            _, preds = _run_epoch(
                model, DataLoader(ds(eval_idx), batch_size=a.batch, shuffle=False,
                                  collate_fn=_collate), device)
            infer_sec = time.time() - t1
            df = pd.DataFrame({
                "sample_idx": exp.pairs[eval_idx, 0],
                "drug_idx": exp.pairs[eval_idx, 1],
                "true": exp.labels[eval_idx].astype(np.float64),
                "pred": preds.astype(np.float64)})
            common.write_predictions(df, os.path.join(out_dir, f"fold_{k}_predictions.parquet"))

        meta = {"model": "TGSA", "split_mode": a.split_mode, "mode": a.mode,
                "fold": int(k), "n_params": n_params, "best_epoch": int(best_epoch),
                "best_val_mse": float(best_val), "train_sec": float(train_sec),
                "infer_sec": float(infer_sec) if infer_sec == infer_sec else None,
                "device": device, "epochs": a.epochs, "patience": a.patience,
                "lr": a.lr, "batch": a.batch, "edge_thresh": a.edge_thresh,
                "n_gene": int(n_gene), "n_omics": int(n_om), "n_edges": int(edge_index.shape[1]),
                "smoke": a.smoke, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                "n_eval": int(len(eval_idx)) if eval_idx is not None else 0}
        with open(os.path.join(out_dir, f"fold_{k}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[TGSA] {a.split_mode}/{a.mode} fold {k}: params={n_params:,} "
              f"n_edges={edge_index.shape[1]} best_epoch={best_epoch} "
              f"val_mse={best_val:.4f} train={train_sec:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
