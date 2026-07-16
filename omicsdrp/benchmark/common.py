"""Dependency-light contract shared by every competitor adapter.

An adapter runs inside its OWN conda env (DeepTTA -> omicsdrp env, GraphDRP ->
PyG env, DeepCDR -> TF1.x env) and therefore CANNOT import the ``omicsdrp``
package (torch/torch-geometric version clashes). So this module deliberately
depends on **numpy + sklearn only** and is imported by every adapter to:

  * read the frozen benchmark export produced by ``export_data.py``
    (the identical (cell,drug) pair universe + our ``build_folds`` fold indices),
  * rebuild fold-scaled 909-gene cell features with the SAME leakage boundary
    OmicsDRP uses (StandardScaler fit on train cells only), and
  * emit / validate the unified per-pair prediction schema so ``score.py`` can
    score every model with the identical metric function.

The export is written ONCE (in the omicsdrp env); adapters only read it.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    """Replicates omicsdrp.engine.set_seed EXACTLY, so competitor training shares
    OmicsDRP's determinism protocol. Call per fold with ``base_seed + fold`` (as
    nested_cv.run_fold does) so a fold is reproducible whether run alone or in a
    sequence. Imports torch lazily (common.py stays importable in torch-free envs)."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

# Fixed omics column order inside the exported [N_cell, n_gene, 4] tensor,
# mirroring omicsdrp.config.OMICS_ORDER (preprocessing-fixed).
OMICS_ORDER = ["SNP", "MET", "CNV", "RNA"]
OMICS_TO_INDEX = {name: i for i, name in enumerate(OMICS_ORDER)}

# The prediction file every adapter must emit (one row per outer-test pair).
PREDICTION_COLUMNS = ["sample_idx", "drug_idx", "true", "pred"]

SPLIT_MODES = ("mixed", "unseen_cell", "unseen_drug")


# --------------------------------------------------------------------------- #
# reading the frozen export
# --------------------------------------------------------------------------- #
class BenchmarkExport:
    """Loader over the directory written by ``export_data.py``.

    Attributes
    ----------
    pairs        : [M, 2] int   -> (cell_row, drug_col); the shared pair universe.
    labels       : [M]  float   -> ln(IC50) for each pair (native log, no scaling).
    omics        : [N_cell, n_gene, 4] float32  -> UNSCALED per-gene omics.
    genes        : list[str]    -> gene order along axis 1 of ``omics``.
    smiles       : list[str]    -> SMILES in drug-index order (len == n_drug).
    drug_ids     : list[str]    -> external GDSC DRUG_ID in drug-index order.
    cell_labels  : list[str]    -> external cell id (SANGER) in cell-row order.
    meta         : dict         -> full manifest (config, counts, ...).
    """

    def __init__(self, export_dir: str):
        self.dir = export_dir
        with open(os.path.join(export_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.pairs = np.load(os.path.join(export_dir, "pairs.npy"))
        self.labels = np.load(os.path.join(export_dir, "labels.npy"))
        self.omics = np.load(os.path.join(export_dir, "omics.npy"))
        self.genes = self.meta["genes"]
        self.smiles = _read_json(os.path.join(export_dir, "smiles.json"))
        self.drug_ids = _read_json(os.path.join(export_dir, "drug_ids.json"))
        self.cell_labels = _read_json(os.path.join(export_dir, "cell_labels.json"))

    @property
    def n_cell(self) -> int:
        return int(self.omics.shape[0])

    @property
    def n_drug(self) -> int:
        return len(self.smiles)

    def load_fold(self, split_mode: str, fold: int) -> Dict[str, np.ndarray]:
        """Return the fold index arrays for one (split_mode, fold).

        Keys: train_pair_idx, val_pair_idx, test_pair_idx, train_sample_indices
        -- all index into ``self.pairs`` (except train_sample_indices, which are
        cell rows for fitting the scaler). Identical to omicsdrp FoldSpec.
        """
        path = os.path.join(self.dir, "folds", split_mode, f"fold_{fold}.npz")
        z = np.load(path)
        return {k: z[k] for k in z.files}

    def list_folds(self, split_mode: str) -> List[int]:
        d = os.path.join(self.dir, "folds", split_mode)
        return sorted(
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(d) if f.startswith("fold_")
        )


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# fold-scaled cell features (leakage boundary preserved)
# --------------------------------------------------------------------------- #
def scale_omics_train(omics: np.ndarray, train_cell_rows: np.ndarray) -> np.ndarray:
    """StandardScaler fit on *train cell rows only*, applied to all rows.

    Mirrors omicsdrp.data.scale_gene_data: for every gene, each of its omics
    columns is standardised using statistics from the train cells alone. This is
    the nested-CV leakage boundary; adapters MUST call this per fold.

    Parameters
    ----------
    omics : [N_cell, n_gene, n_omics] float
    train_cell_rows : 1-D int array of cell rows in the fold's training pairs.

    Returns
    -------
    [N_cell, n_gene, n_omics] float32, scaled.
    """
    train_idx = np.asarray(sorted(set(int(i) for i in train_cell_rows)))
    n_cell, n_gene, n_omics = omics.shape
    out = np.empty_like(omics, dtype=np.float32)
    for g in range(n_gene):
        scaler = StandardScaler()
        scaler.fit(omics[train_idx, g, :])
        out[:, g, :] = scaler.transform(omics[:, g, :])
    return out


def select_omics_matrix(omics: np.ndarray, omics_names: List[str]) -> np.ndarray:
    """Slice the [N, n_gene, 4] tensor to the chosen modalities, in OMICS_ORDER.

    Returns [N, n_gene, len(omics_names)] with the modality axis kept in the
    canonical SNP,MET,CNV,RNA order so layout is deterministic across adapters.
    """
    idx = sorted(OMICS_TO_INDEX[o] for o in omics_names)
    return omics[:, :, idx]


def build_pair_frame(export: "BenchmarkExport", pair_idx: np.ndarray):
    """Materialise a fold partition as parallel arrays the adapters consume.

    Returns dict with cell_row, drug_col, smiles, ln_ic50, and the original
    pair_idx (so predictions can be written back with sample_idx/drug_idx).
    """
    pi = np.asarray(pair_idx)
    cells = export.pairs[pi, 0]
    drugs = export.pairs[pi, 1]
    return {
        "pair_idx": pi,
        "cell_row": cells,
        "drug_col": drugs,
        "smiles": [export.smiles[d] for d in drugs],
        "ln_ic50": export.labels[pi],
    }


# --------------------------------------------------------------------------- #
# prediction I/O
# --------------------------------------------------------------------------- #
def validate_predictions(df) -> None:
    """Raise if a per-pair prediction frame is not in the unified schema."""
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"prediction frame missing columns {missing}; "
            f"required schema = {PREDICTION_COLUMNS}")
    if df[["sample_idx", "drug_idx"]].isnull().any().any():
        raise ValueError("sample_idx/drug_idx contain nulls")


def write_predictions(df, path: str) -> None:
    """Write per-pair predictions (parquet, .pkl fallback), schema-validated."""
    validate_predictions(df)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df[PREDICTION_COLUMNS]
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # fallback: swap the .parquet suffix for .pkl (never append, so the
        # scorer's fold_<k>_predictions.<ext> pattern still matches).
        alt = path[:-len(".parquet")] + ".pkl" if path.endswith(".parquet") else path + ".pkl"
        df.to_pickle(alt)
