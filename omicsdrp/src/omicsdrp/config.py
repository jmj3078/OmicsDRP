"""Experiment configuration for the OmicsDRP ablation skeleton.

A single ``ExperimentConfig`` fully specifies one run along every ablation axis
of Stage 1:

  1. which omics modalities feed the cell branch,
  2. the cell encoder architecture (attention vs. plain MLP),
  3. the drug representation method,
  4. the data-splitting regime (mixed / unseen-cell / unseen-drug),
  5. the nested-CV / training hyper-parameters.

The orchestrator (:mod:`run_ablations`) materialises a grid of these configs
and executes them sequentially, so adding a new ablated variant is just adding
an entry to a list -- no changes to the training engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import hashlib
import json


# Column order inside every per-gene tensor is fixed by preprocessing
# (see data/raw_data/Data_Preprocessing.ipynb, cell 9):
#     np.column_stack((snp, met, cnv, rna))
OMICS_ORDER = ["SNP", "MET", "CNV", "RNA"]
OMICS_TO_INDEX = {name: i for i, name in enumerate(OMICS_ORDER)}


def omics_to_indices(omics: List[str]) -> List[int]:
    """Map modality names (e.g. ``["RNA", "MET"]``) to column indices, kept in
    canonical ``OMICS_ORDER`` so the model input layout is deterministic."""
    unknown = [o for o in omics if o not in OMICS_TO_INDEX]
    if unknown:
        raise ValueError(f"Unknown omics {unknown}; valid = {OMICS_ORDER}")
    return sorted(OMICS_TO_INDEX[o] for o in omics)


# Recognised choices -- the registries validate against these.
CELL_ENCODERS = ("attention", "mlp")
# Drug-representation candidates (Stage-1 ablation axis 3). Three families:
#   baseline     : morgan                 -- 512-bit fingerprint -> MLP (current model)
#   from-scratch : gin, gcn               -- 2D molecular graph, trained end-to-end
#   pretrained   : chemberta, molformer   -- SMILES language models (frozen embedding)
#                  graphormer, unimol     -- graph/3D transformers (frozen embedding)
# Pretrained encoders use FROZEN, pre-extracted embeddings as fixed input; only a
# projection head is trained. (Uni-Mol replaces GROVER: better maintained via
# `unimol_tools`, stronger drug-discovery context. See notes in drug_encoders.py.)
DRUG_ENCODERS = ("morgan", "gin", "gcn", "chemberta", "molformer", "graphormer", "unimol")
DRUG_ENCODER_FAMILY = {
    "morgan": "baseline",
    "gin": "scratch_graph", "gcn": "scratch_graph",
    "chemberta": "pretrained_lm", "molformer": "pretrained_lm",
    "graphormer": "pretrained_graph", "unimol": "pretrained_graph",
}
SPLIT_MODES = ("mixed", "unseen_cell", "unseen_drug")


@dataclass
class ExperimentConfig:
    # --- identity ---
    name: str = "default"

    # --- ablation axis 1: omics combination ---
    # Single omics collapses the cell branch (input_dim == 1), so the smallest
    # meaningful baselines are 2-modality, e.g. ["RNA", "MET"] or ["RNA", "CNV"].
    omics: List[str] = field(default_factory=lambda: ["SNP", "MET", "CNV", "RNA"])

    # --- ablation axis 2: cell encoder ---
    cell_encoder: str = "attention"          # one of CELL_ENCODERS

    # --- ablation axis 3: drug representation ---
    drug_encoder: str = "morgan"             # one of DRUG_ENCODERS

    # --- ablation axis 4: split regime ---
    split_mode: str = "mixed"                # one of SPLIT_MODES
    # For unseen splits: number of clusters used to stratify groups across folds
    # so a whole similarity-cluster never lands entirely in the test fold.
    # Defaults kept low so every cluster has >= outer_folds members (valid
    # stratification); raise only after inspecting scripts/inspect_clusters.py
    # output -- the k-sweep flags where singleton clusters start appearing.
    n_cluster_cell: int = 6
    n_cluster_drug: int = 8

    # --- nested CV (outer-test / inner-val) ---
    outer_folds: int = 5
    inner_val_frac: float = 0.15             # fraction of outer-train held for early stopping
    seed: int = 2024

    # --- model dims ---
    embedding_dim: int = 128
    gene_embed_dim: int = 8
    num_heads: int = 2
    dropout: float = 0.1

    # --- optimisation ---
    batch_size: int = 128
    num_epochs: int = 100
    lr: float = 0.01
    weight_decay: float = 1e-4
    patience: int = 7

    # --- io ---
    dataset_path: str = "../data"
    out_root: str = "./Results"

    def omics_indices(self) -> List[int]:
        return omics_to_indices(self.omics)

    @property
    def n_omics(self) -> int:
        return len(self.omics)

    def tag(self) -> str:
        """Short, filesystem-safe, collision-resistant experiment tag."""
        omics_tag = "+".join(o for o in OMICS_ORDER if o in self.omics)
        base = f"{self.name}__{omics_tag}__{self.cell_encoder}__{self.drug_encoder}__{self.split_mode}"
        h = hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:6]
        return f"{base}__{h}"

    def to_dict(self) -> dict:
        return asdict(self)
