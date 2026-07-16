"""TGSA/TGDRP wiring for the benchmark (native PyTorch -- no framework port).

Imports the UPSTREAM model verbatim (models.TGDRP + GNN_drug/GNN_cell) and the
upstream OGB/dgllife drug featurizer (smiles2graph); the adapter only supplies
data plumbing so nothing upstream is edited. Two small helpers replace the parts
of ``preprocess_gene.py`` that assume the fixed 706-gene CGC panel and a STRING
links file we don't ship:

  * ``build_gene_graph`` -- gene-gene edges from EXPRESSION CORRELATION over OUR
    909 genes (TGSA's own ``get_genes_graph`` method='pearson'), computed on
    TRAIN cells only (leakage boundary). This avoids the STRING download that
    ``get_STRING_graph`` needs (``9606.protein.links.detailed`` is not shipped).
  * ``build_cluster_predefine`` -- graclus pooling clusters for an arbitrary
    node count (upstream ``get_predefine_cluster`` hardcodes 706).

The cell branch is dimension-flexible (``num_feature`` inferred), so we feed our
909-gene multi-omics: SNP->mutation, CNV->copy-number, RNA->expression (TGSA has
no methylation branch, so MET is dropped). Label is native ln(IC50), MSE -- no
rescale, identical to the other adapters.

Provenance: violet-sto/TGSA @ cdd9903 (see vendor/MANIFEST.md).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("DGLBACKEND", "pytorch")   # smiles2graph -> dgllife needs this

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool

_VENDOR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "TGSA"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from models.TGDRP import TGDRP                    # noqa: E402  (upstream, unmodified)
from smiles2graph import smiles2graph             # noqa: E402  (upstream featurizer)
from rdkit import Chem                            # noqa: E402

# TGSA cell omics = mutation, copy-number, expression (no methylation branch).
# Map from our OMICS_ORDER [SNP,MET,CNV,RNA]:
TGSA_OMICS = ["SNP", "CNV", "RNA"]                # -> mu, cn, exp ; MET dropped
EXPR_MODALITY = "RNA"                             # correlation graph is expression-based


def build_drug_graphs(smiles_list):
    """SMILES (drug-index order) -> list of PyG Data (77-dim atom features)."""
    graphs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smi}")
        graphs.append(smiles2graph(mol))
    return graphs


def build_gene_graph(expr_train: np.ndarray, thresh: float) -> np.ndarray:
    """Gene-gene edge_index from |Pearson correlation| > thresh (TRAIN cells only).

    expr_train : [n_train_cell, n_gene] expression. Correlation is scale-invariant,
    so raw or standardised expression gives the same edges. Returns [2, E] int64.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(expr_train.T)          # zero-variance genes -> NaN row
    corr = np.abs(np.nan_to_num(corr, nan=0.0))   # NaN -> 0 => no edges for those
    adj = (corr > thresh).astype(np.int64)
    np.fill_diagonal(adj, 0)
    edge_index = np.array(np.nonzero(adj), dtype=np.int64)   # [2, E]
    return edge_index


def build_cluster_predefine(edge_index: np.ndarray, n_gene: int, n_levels: int, device):
    """graclus pooling clusters for an n_gene graph (generalises the 706 hardcode)."""
    g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long),
             x=torch.zeros(n_gene, 1))
    g = Batch.from_data_list([g])
    cluster_predefine = {}
    for i in range(n_levels):
        cluster = graclus(g.edge_index, None, g.x.size(0))
        g = max_pool(cluster, g, transform=None)
        cluster_predefine[i] = cluster
    return {i: j.to(device) for i, j in cluster_predefine.items()}


def build_cell_graphs(omics_scaled: np.ndarray, edge_index: np.ndarray):
    """Per-cell PyG Data: x = [n_gene, n_omics] scaled omics, shared gene edges.

    omics_scaled : [N_cell, n_gene, n_omics]. Returns list of Data (cell-row order).
    """
    ei = torch.tensor(edge_index, dtype=torch.long)
    cells = []
    for c in range(omics_scaled.shape[0]):
        cells.append(Data(x=torch.tensor(omics_scaled[c], dtype=torch.float),
                          edge_index=ei))
    return cells


class _Args:
    """Minimal namespace TGDRP expects (matches upstream main.py defaults)."""
    def __init__(self, num_feature, batch_size=128, layer_drug=3, dim_drug=128,
                 layer=3, hidden_dim=8, dropout_ratio=0.2):
        self.num_feature = num_feature
        self.batch_size = batch_size
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio


def build_model(num_feature: int, cluster_predefine, **kw) -> TGDRP:
    return TGDRP(cluster_predefine, _Args(num_feature, **kw))
