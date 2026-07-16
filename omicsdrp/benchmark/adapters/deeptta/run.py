#!/usr/bin/env python
"""DeepTTA benchmark adapter -- train on OUR GDSC2 pairs/folds, our metric.

Consumes the frozen export (``export_data.py``) so DeepTTA sees the IDENTICAL
(cell,drug) pair universe, fold membership, ln(IC50) labels and 909-gene RNA
source as OmicsDRP. Early stopping uses ONLY the fold's inner-val pairs
(upstream leaked the test fold here); the outer-test fold is evaluated once.

Three regimes (all save weights + the fold RNA scaler, for reproducibility and
for external-data inference):
  * ``--mode nested`` with ``--split_mode mixed``        -> Nested-CV metric (리뷰4)
  * ``--mode nested`` with ``--split_mode unseen_cell|unseen_drug`` -> OOD (리뷰3/8)
  * ``--mode ensemble``  -> train pool = outer train+val, early-stop on outer-test;
                            K per-fold models saved for external-data ensembling
                            (mirrors omicsdrp inference_models.py).

GPU safety: default device is CPU. Pass ``--device cuda`` only when the GPU is
free (the Stage-1 sweep runs at ~full GPU). ``--smoke`` forces CPU + tiny data.

    # safe functional smoke (CPU, tiny), zero GPU contention:
    CUDA_VISIBLE_DEVICES="" python run.py --export ../../export_smoke \
        --split_mode mixed --smoke --out /tmp/bench_smoke/DeepTTA
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import common                                    # noqa: E402
from model import DeepTTAModel, ESPFTokenizer    # noqa: E402

RNA = "RNA"   # DeepTTA is expression-only; feed our 909-gene RNA modality


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True, help="dir written by export_data.py")
    p.add_argument("--split_mode", default="mixed",
                   choices=["mixed", "unseen_cell", "unseen_drug"])
    p.add_argument("--mode", default="nested", choices=["nested", "ensemble"])
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="fold ids to run (default: all present in the export)")
    p.add_argument("--out", default="./BenchmarkResults/DeepTTA")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="cap pairs per split (smoke sets 512)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
class PairDataset(Dataset):
    """One row per (cell,drug) pair; RNA held once as [N_cell, n_gene] (no dup)."""

    def __init__(self, cell_rows, tokens, masks, rna_matrix, labels):
        self.cell_rows = np.asarray(cell_rows)
        self.tokens = tokens        # [n_pairs, 50] int64
        self.masks = masks          # [n_pairs, 50] int64
        self.rna = rna_matrix       # [N_cell, n_gene] float32
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.cell_rows)

    def __getitem__(self, i):
        return (self.tokens[i], self.masks[i],
                self.rna[self.cell_rows[i]], self.labels[i])


def _loader(ds, batch, shuffle, device):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(device == "cuda"), drop_last=False)


def _run_epoch(model, loader, device, opt=None):
    """Train (opt given) or eval one epoch; returns (mean_loss, preds, trues)."""
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    losses, preds, trues = [], [], []
    with torch.set_grad_enabled(train):
        for tokens, masks, gene, y in loader:
            tokens, masks = tokens.to(device), masks.to(device)
            gene, y = gene.to(device), y.float().to(device)
            out = model(tokens, masks, gene).squeeze(1)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item() * len(y))
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    n = sum(len(t) for t in trues)
    return (sum(losses) / n, np.concatenate(preds), np.concatenate(trues))


def train_one_fold(model, train_ds, val_ds, device, epochs, patience, lr, batch):
    """Early stop on val MSE (val only!); restore best weights. Returns best_epoch."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse, best_state, best_epoch, stale = float("inf"), None, -1, 0
    tr_loader = _loader(train_ds, batch, True, device)
    va_loader = _loader(val_ds, batch, False, device)
    for epoch in range(epochs):
        _run_epoch(model, tr_loader, device, opt)
        val_mse, _, _ = _run_epoch(model, va_loader, device)
        if val_mse < best_mse - 1e-6:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_mse


# --------------------------------------------------------------------------- #
def main():
    a = parse_args()
    if a.smoke:
        a.device = "cpu"
        a.epochs = min(a.epochs, 2)
        a.max_pairs = a.max_pairs or 512
        a.patience = min(a.patience, 2)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    exp = common.BenchmarkExport(a.export)
    genes = exp.genes
    n_gene = len(genes)

    # ESPF-encode all 231 drugs once (drug-index order)
    tok = ESPFTokenizer()
    assert tok.vocab_size == 2586, tok.vocab_size
    drug_tokens = np.zeros((exp.n_drug, 50), dtype=np.int64)
    drug_masks = np.zeros((exp.n_drug, 50), dtype=np.int64)
    for d, smi in enumerate(exp.smiles):
        ids, m = tok.encode(smi)
        drug_tokens[d], drug_masks[d] = ids, m

    folds = a.folds or exp.list_folds(a.split_mode)
    out_dir = os.path.join(a.out, a.split_mode if a.mode == "nested" else "ensemble")
    os.makedirs(out_dir, exist_ok=True)

    for k in folds:
        common.set_seed(a.seed + k)   # per-fold seed, identical to omicsdrp run_fold
        f = exp.load_fold(a.split_mode, k)
        if a.mode == "nested":
            train_idx, val_idx, eval_idx = (
                f["train_pair_idx"], f["val_pair_idx"], f["test_pair_idx"])
        else:  # ensemble: train pool = train+val, early-stop on test, no held-out eval
            train_idx = np.concatenate([f["train_pair_idx"], f["val_pair_idx"]])
            val_idx, eval_idx = f["test_pair_idx"], None

        if a.max_pairs:
            rng = np.random.RandomState(a.seed)
            train_idx = rng.permutation(train_idx)[:a.max_pairs]
            val_idx = rng.permutation(val_idx)[:max(64, a.max_pairs // 4)]
            if eval_idx is not None:
                eval_idx = rng.permutation(eval_idx)[:max(64, a.max_pairs // 4)]

        # RNA scaler fit on the fold's TRAIN cells only (leakage boundary), saved.
        train_cells = np.asarray(sorted(set(exp.pairs[train_idx, 0].tolist())))
        rna_raw = common.select_omics_matrix(exp.omics, [RNA])[:, :, 0]  # [N_cell, n_gene]
        scaler = StandardScaler().fit(rna_raw[train_cells])
        rna = scaler.transform(rna_raw).astype(np.float32)

        def make_ds(pair_idx):
            cells = exp.pairs[pair_idx, 0]
            drugs = exp.pairs[pair_idx, 1]
            return PairDataset(cells, drug_tokens[drugs], drug_masks[drugs],
                               rna, exp.labels[pair_idx])

        model = DeepTTAModel(input_dim_gene=n_gene, vocab_size=tok.vocab_size)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        t0 = time.time()
        best_epoch, best_val = train_one_fold(
            model, make_ds(train_idx), make_ds(val_idx),
            device, a.epochs, a.patience, a.lr, a.batch)
        train_sec = time.time() - t0

        # save weights + scaler (needed for all 3 regimes / external inference)
        torch.save(model.state_dict(), os.path.join(out_dir, f"fold_{k}_model.pt"))
        np.savez(os.path.join(out_dir, f"fold_{k}_scaler.npz"),
                 mean=scaler.mean_, scale=scaler.scale_, genes=np.array(genes),
                 modality=RNA)

        infer_sec = np.nan
        if eval_idx is not None:
            t1 = time.time()
            _, preds, trues = _run_epoch(model, _loader(make_ds(eval_idx), a.batch, False, device), device)
            infer_sec = time.time() - t1
            df = pd.DataFrame({
                "sample_idx": exp.pairs[eval_idx, 0],
                "drug_idx": exp.pairs[eval_idx, 1],
                "true": trues, "pred": preds})
            common.write_predictions(df, os.path.join(out_dir, f"fold_{k}_predictions.parquet"))

        meta = {"model": "DeepTTA", "split_mode": a.split_mode, "mode": a.mode,
                "fold": int(k), "n_params": n_params, "best_epoch": int(best_epoch),
                "best_val_mse": float(best_val), "train_sec": float(train_sec),
                "infer_sec": float(infer_sec) if infer_sec == infer_sec else None,
                "device": device, "epochs": a.epochs, "patience": a.patience,
                "lr": a.lr, "batch": a.batch, "smoke": a.smoke,
                "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                "n_eval": int(len(eval_idx)) if eval_idx is not None else 0}
        with open(os.path.join(out_dir, f"fold_{k}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[DeepTTA] {a.split_mode}/{a.mode} fold {k}: "
              f"params={n_params:,} best_epoch={best_epoch} val_mse={best_val:.4f} "
              f"train={train_sec:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
