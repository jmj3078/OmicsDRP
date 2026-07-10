"""Clustering & split diagnostics for the unseen-cell / unseen-drug regimes.

Purpose: make the *threshold* (the number of KMeans clusters used to stratify
groups across folds) an informed, user-set choice rather than a hidden default.
Everything is written to disk as figures + CSV so the user can look and decide
``n_cluster_cell`` / ``n_cluster_drug`` themselves.

Produces, for a target ("cell" | "drug"):

  * ``<target>_k_sweep.csv/.png``   -- inertia (elbow) + silhouette vs. #clusters,
                                       silhouette computed in the target's natural
                                       geometry (euclidean for cells, Jaccard for
                                       drugs). This is the plot to pick k from.
  * ``<target>_embedding_k{K}.png`` -- 2D PCA (and t-SNE) scatter coloured by the
                                       chosen clustering, to eyeball separation.
  * ``<target>_ood_distance_k{K}.png`` -- per outer fold, distribution of each
                                       test group's distance to its NEAREST train
                                       group. Small distances ⇒ conservative
                                       (test not fully OOD); a long right tail ⇒
                                       some test groups are far from anything in
                                       train. This directly shows how OOD the
                                       split is at the chosen k.
  * ``<target>_cluster_fold_table.csv`` -- how clusters spread across outer folds.
"""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .splits import group_features, cluster_labels, build_folds


def _reduce(features: np.ndarray, metric: str, seed: int, pca_dim: int = 50):
    """Standardise + PCA the features into the space KMeans actually sees."""
    x = StandardScaler().fit_transform(features)
    if x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    return x


def k_sweep(raw, config, target: str, k_values: List[int],
            out_dir: str, seed: Optional[int] = None) -> pd.DataFrame:
    seed = config.seed if seed is None else seed
    features, metric = group_features(raw, config, target)
    x = _reduce(features, metric, seed)

    # silhouette in the natural metric (euclidean cells / jaccard drugs) on the
    # ORIGINAL features, using the KMeans labels from the reduced space.
    sil_features = features.astype(bool) if metric == "jaccard" else features
    rows = []
    for k in k_values:
        if k < 2 or k >= features.shape[0]:
            continue
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
        km_inertia = float(KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x).inertia_)
        try:
            sil = float(silhouette_score(sil_features, labels, metric=metric))
        except Exception:
            sil = float("nan")
        sizes = np.bincount(labels)
        rows.append({"k": k, "inertia": km_inertia, "silhouette": sil,
                     "min_cluster_size": int(sizes.min()),
                     "max_cluster_size": int(sizes.max())})
    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{target}_k_sweep.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["k"], df["inertia"], "o-", color="tab:blue", label="inertia (elbow)")
    ax1.set_xlabel("number of clusters (k)")
    ax1.set_ylabel("KMeans inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["silhouette"], "s-", color="tab:red",
             label=f"silhouette ({metric})")
    ax2.set_ylabel(f"silhouette ({metric})", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    best = df.loc[df["silhouette"].idxmax()] if df["silhouette"].notna().any() else None
    if best is not None:
        ax1.axvline(best["k"], ls="--", color="grey", alpha=0.7)
        ax1.set_title(f"{target}: pick k — best silhouette at k={int(best['k'])} "
                      f"({best['silhouette']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{target}_k_sweep.png"), dpi=140)
    plt.close(fig)
    return df


def embedding_plot(raw, config, target: str, k: int, out_dir: str,
                   seed: Optional[int] = None, do_tsne: bool = True) -> None:
    seed = config.seed if seed is None else seed
    features, metric = group_features(raw, config, target)
    labels = cluster_labels(features, k, seed)

    x = _reduce(features, metric, seed)
    coords = {"pca": PCA(n_components=2, random_state=seed).fit_transform(x)}
    if do_tsne and features.shape[0] > 10:
        perp = min(30, max(5, features.shape[0] // 10))
        coords["tsne"] = TSNE(n_components=2, random_state=seed,
                              perplexity=perp, init="pca").fit_transform(x)

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(coords), figsize=(6 * len(coords), 5), squeeze=False)
    for ax, (name, xy) in zip(axes[0], coords.items()):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=14, alpha=0.8)
        ax.set_title(f"{target} {name.upper()} — k={k}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{target} clustering (k={k}), coloured by cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{target}_embedding_k{k}.png"), dpi=140)
    plt.close(fig)


def ood_distance_report(raw, config, target: str, k: int, out_dir: str,
                        seed: Optional[int] = None) -> pd.DataFrame:
    """For the ACTUAL split at cluster count k, measure how OOD each test group is.

    For every outer fold, compute each held-out test group's distance to its
    nearest training group (in the natural metric) and summarise the
    distribution. This is the concrete number to judge whether the split is
    "conservative enough".
    """
    seed = config.seed if seed is None else seed
    from dataclasses import replace
    cfg = replace(config, split_mode=f"unseen_{target}",
                  **({"n_cluster_cell": k} if target == "cell" else {"n_cluster_drug": k}))
    features, metric = group_features(raw, cfg, target)
    feat = features.astype(bool) if metric == "jaccard" else features

    folds = build_folds(raw, cfg)
    group_col = 0 if target == "cell" else 1  # column in pairs
    rows = []
    per_fold_dists = []
    for f in folds:
        train_groups = np.unique(raw.pairs[f.train_pair_idx][:, group_col])
        test_groups = np.unique(raw.pairs[f.test_pair_idx][:, group_col])
        D = pairwise_distances(feat[test_groups], feat[train_groups], metric=metric)
        nn = D.min(axis=1)  # nearest train group per test group
        per_fold_dists.append((f.fold, nn))
        rows.append({"fold": f.fold, "n_test_groups": len(test_groups),
                     "nn_dist_mean": float(nn.mean()), "nn_dist_median": float(np.median(nn)),
                     "nn_dist_p90": float(np.percentile(nn, 90)),
                     "nn_dist_max": float(nn.max())})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{target}_ood_distance_k{k}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for fold, nn in per_fold_dists:
        ax.hist(nn, bins=25, alpha=0.5, label=f"fold {fold}")
    ax.set_xlabel(f"test group → nearest TRAIN group distance ({metric})")
    ax.set_ylabel("count")
    ax.set_title(f"{target}: OOD-ness of unseen split at k={k}\n"
                 f"(left = conservative / similar neighbour in train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{target}_ood_distance_k{k}.png"), dpi=140)
    plt.close(fig)
    return df


def cluster_fold_table(raw, config, target: str, k: int, out_dir: str,
                       seed: Optional[int] = None) -> pd.DataFrame:
    """Cross-tab of cluster vs. outer test fold — shows the stratification worked
    (each cluster spread across folds, not dumped into one)."""
    seed = config.seed if seed is None else seed
    features, _ = group_features(raw, config, target)
    labels = cluster_labels(features, k, seed)
    n_groups = features.shape[0]
    skf = StratifiedKFold(n_splits=config.outer_folds, shuffle=True, random_state=42)
    fold_of = np.empty(n_groups, dtype=int)
    for fold, (_, test_g) in enumerate(skf.split(np.arange(n_groups), labels), 1):
        fold_of[test_g] = fold
    tab = pd.crosstab(pd.Series(labels, name="cluster"),
                      pd.Series(fold_of, name="test_fold"))
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, f"{target}_cluster_fold_table.csv"))
    return tab


def run_full_diagnostics(raw, config, target: str, k_values: List[int],
                         chosen_k: int, out_dir: str) -> None:
    """Everything for one target, written under ``out_dir``."""
    print(f"[diag:{target}] k-sweep over {k_values} ...")
    k_sweep(raw, config, target, k_values, out_dir)
    print(f"[diag:{target}] embedding + OOD + fold table at chosen k={chosen_k} ...")
    embedding_plot(raw, config, target, chosen_k, out_dir)
    ood_distance_report(raw, config, target, chosen_k, out_dir)
    cluster_fold_table(raw, config, target, chosen_k, out_dir)
    print(f"[diag:{target}] done -> {out_dir}")
