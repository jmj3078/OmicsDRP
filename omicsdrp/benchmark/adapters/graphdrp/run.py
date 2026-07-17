#!/usr/bin/env python
"""GraphDRP benchmark adapter -- train on OUR GDSC2 pairs/folds, our metric.

Same contract as the DeepTTA adapter (shared export, identical fold membership,
early-stop on inner-val only, single outer-test eval, weights + cell scaler saved
per fold for all three regimes). GraphDRP-specific points:
  * cell branch = OUR 909-gene x 4-omics vector (3636-dim), scaled train-only;
  * drug branch = SMILES->graph (GIN);
  * label is trained in GraphDRP's squashed space sigmoid(0.1*lnIC50) and predictions
    are inverted with 10*logit(y) before scoring, so metrics are in ln(IC50) space
    -- identical to OmicsDRP / the other adapters.

Runs in the omicsdrp env (PyG 2.x). GPU safety: default CPU; --smoke forces CPU.

    CUDA_VISIBLE_DEVICES="" python run.py --export ../../export_smoke \
        --split_mode mixed --smoke --out ../../BenchmarkResults/GraphDRP
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import common                                              # noqa: E402
from model import GINConvNet, smile_to_graph, ln_to_norm, norm_to_ln   # noqa: E402

OMICS = ["SNP", "MET", "CNV", "RNA"]   # feed the full 4-omics stack (== OmicsDRP)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--split_mode", default="mixed",
                   choices=["mixed", "unseen_cell", "unseen_drug"])
    p.add_argument("--mode", default="nested", choices=["nested", "ensemble"])
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--out", default="./BenchmarkResults/GraphDRP")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute folds even if their outputs already exist")
    return p.parse_args()


def _build_graphs(smiles_list):
    """SMILES (drug-index order) -> list of (n_atom, feats, edge_index)."""
    return [smile_to_graph(s) for s in smiles_list]


def _make_dataset(pair_idx, pairs, graphs, cell_mat, norm_labels):
    """One PyG Data per pair: drug graph + cell target + squashed label."""
    data_list = []
    for j, pi in enumerate(pair_idx):
        cell, drug = int(pairs[pi, 0]), int(pairs[pi, 1])
        n_atom, feats, edge_index = graphs[drug]
        data_list.append(Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            target=torch.tensor(cell_mat[cell], dtype=torch.float).unsqueeze(0),
            y=torch.tensor([norm_labels[j]], dtype=torch.float)))
    return data_list


def _run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    losses, preds, n = [], [], 0
    with torch.set_grad_enabled(train):
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze(1)
            loss = crit(out, data.y.view(-1))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item() * data.num_graphs)
            preds.append(out.detach().cpu().numpy())
            n += data.num_graphs
    return sum(losses) / n, np.concatenate(preds) if preds else np.array([])


def train_one_fold(model, train_list, val_list, device, epochs, patience, lr, batch):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = DataLoader(train_list, batch_size=batch, shuffle=True, drop_last=True)
    va = DataLoader(val_list, batch_size=batch, shuffle=False)
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
        a.max_pairs = a.max_pairs or 512
        a.patience = min(a.patience, 2)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    device = a.device

    exp = common.BenchmarkExport(a.export)
    graphs = _build_graphs(exp.smiles)           # 231 drug graphs, once
    omics_sel = common.select_omics_matrix(exp.omics, OMICS)          # [N, n_gene, 4]
    n_cell, n_gene, n_om = omics_sel.shape
    flat_raw = omics_sel.reshape(n_cell, n_gene * n_om)               # [N, 3636]
    cell_len = flat_raw.shape[1]

    folds = a.folds or exp.list_folds(a.split_mode)
    out_dir = os.path.join(a.out, a.split_mode if a.mode == "nested" else "ensemble")
    os.makedirs(out_dir, exist_ok=True)

    for k in folds:
        if not a.overwrite and common.fold_outputs_exist(out_dir, k, a.mode):
            print(f"[GraphDRP] {a.split_mode}/{a.mode} fold {k}: done, skipping")
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

        # cell scaler fit on this fold's TRAIN cells only (leakage boundary)
        train_cells = np.asarray(sorted(set(exp.pairs[train_idx, 0].tolist())))
        scaler = StandardScaler().fit(flat_raw[train_cells])
        cell_mat = scaler.transform(flat_raw).astype(np.float32)

        def ds(idx):
            return _make_dataset(idx, exp.pairs, graphs, cell_mat,
                                 ln_to_norm(exp.labels[idx]))

        model = GINConvNet(cell_len=cell_len)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        t0 = time.time()
        best_epoch, best_val = train_one_fold(
            model, ds(train_idx), ds(val_idx), device,
            a.epochs, a.patience, a.lr, a.batch)
        train_sec = time.time() - t0

        torch.save(model.state_dict(), os.path.join(out_dir, f"fold_{k}_model.pt"))
        np.savez(os.path.join(out_dir, f"fold_{k}_scaler.npz"),
                 mean=scaler.mean_, scale=scaler.scale_,
                 genes=np.array(exp.genes), modality="+".join(OMICS), cell_len=cell_len)

        infer_sec = np.nan
        if eval_idx is not None:
            t1 = time.time()
            _, norm_pred = _run_epoch(
                model, DataLoader(ds(eval_idx), batch_size=a.batch, shuffle=False), device)
            infer_sec = time.time() - t1
            df = pd.DataFrame({
                "sample_idx": exp.pairs[eval_idx, 0],
                "drug_idx": exp.pairs[eval_idx, 1],
                "true": exp.labels[eval_idx].astype(np.float64),   # ln IC50 (native)
                "pred": norm_to_ln(norm_pred)})                    # invert squash -> ln
            common.write_predictions(df, os.path.join(out_dir, f"fold_{k}_predictions.parquet"))

        meta = {"model": "GraphDRP", "split_mode": a.split_mode, "mode": a.mode,
                "fold": int(k), "n_params": n_params, "best_epoch": int(best_epoch),
                "best_val_norm_mse": float(best_val), "train_sec": float(train_sec),
                "infer_sec": float(infer_sec) if infer_sec == infer_sec else None,
                "device": device, "epochs": a.epochs, "patience": a.patience,
                "lr": a.lr, "batch": a.batch, "smoke": a.smoke, "cell_len": int(cell_len),
                "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                "n_eval": int(len(eval_idx)) if eval_idx is not None else 0}
        with open(os.path.join(out_dir, f"fold_{k}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[GraphDRP] {a.split_mode}/{a.mode} fold {k}: params={n_params:,} "
              f"best_epoch={best_epoch} val_norm_mse={best_val:.5f} "
              f"train={train_sec:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
