"""Data loading with omics-subset support.

Loads the three raw artefacts (per-gene omics tensors, drug metadata, IC50
matrix) and exposes helpers to:

  * select an omics subset (a column slice of every per-gene [N, 4] tensor),
  * fit feature scaling **only on training samples** (no leakage across the
    outer-test / inner-val boundary),
  * build an encoder-agnostic ``Dataset`` that yields *indices*; the drug
    representation itself is owned by the drug encoder (so a GNN can hold graph
    objects while Morgan holds a fingerprint table, without changing the loop).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


@dataclass
class RawData:
    gene_data: Dict[str, torch.Tensor]   # gene -> [N, 4]  (SNP, MET, CNV, RNA)
    genes: List[str]
    drug_meta: pd.DataFrame              # columns incl DRUG_ID, SMILE, Morgan_Fingerprint
    ic50: torch.Tensor                   # [N_cell, N_drug]
    pairs: np.ndarray                    # [M, 2] -> (sample_idx, drug_idx), IC50 non-NaN
    n_cell: int
    n_drug: int


def _canonical_smiles(smiles: List[str]) -> List[str]:
    """RDKit-canonicalise SMILES (so same molecule/different string collapses);
    fall back to the raw string if rdkit is unavailable or parsing fails."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        out = []
        for s in smiles:
            m = Chem.MolFromSmiles(str(s))
            out.append(Chem.MolToSmiles(m) if m is not None else str(s))
        return out
    except Exception:
        return [str(s) for s in smiles]


def merge_duplicate_drugs(ic50_df: pd.DataFrame, drug_meta: pd.DataFrame):
    """Collapse drugs that are the SAME molecule (identical canonical SMILES) into
    one, at the DATA level.

    GDSC2 registers 10 molecules twice under different DRUG_IDs, with slightly
    different IC50s per screening batch. Structure-based encoders can't tell them
    apart, so we merge them: IC50 becomes the (NaN-aware) mean across the duplicate
    columns, the merged label is ``name1 (id1)/name2 (id2)``, and a ``_source_row``
    column records the representative original row (used to remap pretrained
    embedding tables, whose rows are in the original 241-drug order).

    Returns (merged_ic50_df [n_cell x n_unique], merged_drug_meta [n_unique]).
    """
    keys = _canonical_smiles(drug_meta["SMILE"].astype(str).tolist())
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    ic50_vals = ic50_df.values  # [n_cell, n_drug], column order == drug_meta rows
    merged_rows, merged_cols = [], []
    import warnings
    with warnings.catch_warnings():
        # a cell that is NaN in BOTH duplicate columns -> nanmean of all-NaN ->
        # NaN (correct; that pair is simply absent). Silence the noisy warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for idxs in groups.values():
            rep = idxs[0]
            row = drug_meta.iloc[rep].copy()
            if len(idxs) > 1:
                ids = [str(drug_meta.iloc[j]["DRUG_ID"]) for j in idxs]
                names = [f"{drug_meta.iloc[j]['DRUG_NAME']} ({drug_meta.iloc[j]['DRUG_ID']})"
                         for j in idxs]
                row["DRUG_ID"] = "/".join(ids)
                row["DRUG_NAME"] = "/".join(names)
                merged_cols.append(np.nanmean(ic50_vals[:, idxs], axis=1))
            else:
                merged_cols.append(ic50_vals[:, rep])
            row["_source_row"] = rep
            merged_rows.append(row)

    merged_drug_meta = pd.DataFrame(merged_rows).reset_index(drop=True)
    merged_ic50 = np.column_stack(merged_cols).astype(np.float32)
    merged_ic50_df = pd.DataFrame(merged_ic50, index=ic50_df.index,
                                  columns=list(merged_drug_meta["DRUG_ID"]))
    return merged_ic50_df, merged_drug_meta


def load_raw(dataset_path: str, merge_duplicates: bool = True) -> RawData:
    gene_data = torch.load(f"{dataset_path}/PGKB_Gene_data_dict.pth")
    genes = list(gene_data.keys())

    ic50_df = pd.read_csv(f"{dataset_path}/IC50_GDSC2.csv", index_col=0)
    drug_meta = pd.read_csv(
        f"{dataset_path}/TargetDrugs_with_MorganFingerprint_GDSC2_512.txt", sep="\t")

    if merge_duplicates:
        ic50_df, drug_meta = merge_duplicate_drugs(ic50_df, drug_meta)

    ic50 = torch.from_numpy(ic50_df.values).type(torch.FloatTensor)
    pairs = np.argwhere(~np.isnan(ic50_df.values))  # [M, 2]

    return RawData(
        gene_data=gene_data,
        genes=genes,
        drug_meta=drug_meta,
        ic50=ic50,
        pairs=pairs,
        n_cell=ic50.shape[0],
        n_drug=ic50.shape[1],
    )


def select_omics(gene_data: Dict[str, torch.Tensor],
                 omics_indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Column-slice every per-gene tensor down to the chosen modalities."""
    idx = torch.as_tensor(list(omics_indices), dtype=torch.long)
    return {g: t.index_select(1, idx).contiguous() for g, t in gene_data.items()}


def scale_gene_data(gene_data: Dict[str, torch.Tensor],
                    train_sample_indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    """StandardScaler fit on *train* cell rows only, applied to all rows.

    Fitting on the union of train samples (never val/test) is what keeps the
    nested-CV estimate honest.
    """
    train_idx = np.asarray(sorted(set(int(i) for i in train_sample_indices)))
    scaled: Dict[str, torch.Tensor] = {}
    for gene, mat in gene_data.items():
        mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)
        scaler = StandardScaler()
        scaler.fit(mat_np[train_idx])
        scaled[gene] = torch.tensor(scaler.transform(mat_np), dtype=torch.float32)
    return scaled


class OmicsDrugDataset(Dataset):
    """Yields (sample_features_dict, drug_idx, ic50, (sample_idx, drug_idx)).

    Drug features are *not* materialised here -- the model's drug encoder gathers
    them by ``drug_idx`` from its own precomputed table/graph list.
    """

    def __init__(self, gene_data: Dict[str, torch.Tensor], ic50: torch.Tensor,
                 pairs: Sequence[Sequence[int]]):
        self.gene_data = gene_data
        self.ic50 = ic50
        self.pairs = [(int(s), int(d)) for s, d in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sample_idx, drug_idx = self.pairs[idx]
        sample_features = {g: self.gene_data[g][sample_idx] for g in self.gene_data}
        ic50_value = self.ic50[sample_idx, drug_idx]
        return sample_features, drug_idx, ic50_value, (sample_idx, drug_idx)
