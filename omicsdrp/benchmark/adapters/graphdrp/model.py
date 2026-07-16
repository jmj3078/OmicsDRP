"""GraphDRP (GINConvNet) architecture + SMILES graph featurizer, isolated.

Upstream ``preprocess.py`` imports pubchempy/h5py/matplotlib at module top and
hardcodes the cell-vector width via ``fc1_xt = Linear(2944, ...)`` (tuned for its
735-feature cell vector). We copy the tiny featurizer + model here so the cell
branch adapts to OUR cell-vector length (909 genes x 4 omics = 3636) and nothing
upstream is touched. Only rdkit + networkx + torch_geometric are used (all present
in the omicsdrp env, so no separate env is needed -- the audit's PyG-2.x path).

Label convention (kept identical to GraphDRP): the model regresses the squashed
target ``y = sigmoid(0.1 * lnIC50)`` in [0,1] and ends in Sigmoid; predictions are
inverted back to ln(IC50) via ``10 * logit(y)`` for scoring (round-trip verified).

Provenance: hauldhut/GraphDRP @ ad25065 (see vendor/MANIFEST.md).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import networkx as nx
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

ATOM_FDIM = 78   # matches GraphDRP num_features_xd


# --------------------------------------------------------------------------- #
# label transform (GraphDRP's normalised IC50 <-> our ln IC50)
# --------------------------------------------------------------------------- #
def ln_to_norm(x: np.ndarray) -> np.ndarray:
    """ln(IC50) -> sigmoid(0.1 * x) in (0,1); GraphDRP's training target."""
    return 1.0 / (1.0 + np.exp(-0.1 * np.asarray(x, dtype=np.float64)))


def norm_to_ln(y: np.ndarray) -> np.ndarray:
    """Invert: y in (0,1) -> 10 * logit(y) = ln(IC50). Clipped for numerical safety."""
    y = np.clip(np.asarray(y, dtype=np.float64), 1e-7, 1 - 1e-7)
    return 10.0 * np.log(y / (1.0 - y))


# --------------------------------------------------------------------------- #
# SMILES -> graph (copied from GraphDRP preprocess.py)
# --------------------------------------------------------------------------- #
def _one_of_k(x, allowable, unk=False):
    if x not in allowable:
        if not unk:
            raise ValueError(f"{x} not in {allowable}")
        x = allowable[-1]
    return [x == s for s in allowable]


def _atom_features(atom):
    return np.array(
        _one_of_k(atom.GetSymbol(),
                  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                   'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                   'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                   'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'], unk=True)
        + _one_of_k(atom.GetDegree(), list(range(11)))
        + _one_of_k(atom.GetTotalNumHs(), list(range(11)), unk=True)
        + _one_of_k(atom.GetImplicitValence(), list(range(11)), unk=True)
        + [atom.GetIsAromatic()])


def smile_to_graph(smile: str):
    """Return (n_atom, features[n_atom,78], edge_index[2,E]); matches GraphDRP."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smile}")
    feats = []
    for atom in mol.GetAtoms():
        f = _atom_features(atom).astype(np.float64)
        feats.append(f / f.sum())               # row-normalised, as upstream
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    if edges:
        g = nx.Graph(edges).to_directed()
        edge_index = np.array([[e1, e2] for e1, e2 in g.edges], dtype=np.int64).T
    else:  # single-atom / bond-less molecule
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return mol.GetNumAtoms(), np.asarray(feats, dtype=np.float32), edge_index


# --------------------------------------------------------------------------- #
# cell-branch conv arithmetic -> fc1_xt input width
# --------------------------------------------------------------------------- #
def cell_fc_in(cell_len: int, n_filters: int = 32) -> int:
    """Replicate GraphDRP's 3x (Conv1d k8, MaxPool1d 3) to size fc1_xt.

    Verified: cell_len=735 -> 2944 (the upstream hardcoded value)."""
    def conv(x, k=8):
        return x - k + 1

    def pool(x, k=3):
        return (x - k) // k + 1

    L = cell_len
    for _ in range(3):
        L = pool(conv(L))
    if L <= 0:
        raise ValueError(f"cell vector length {cell_len} too small for the conv stack")
    return n_filters * 4 * L


# --------------------------------------------------------------------------- #
# model (copied from models/ginconv.py, cell width made configurable)
# --------------------------------------------------------------------------- #
class GINConvNet(nn.Module):
    def __init__(self, cell_len: int, num_features_xd: int = ATOM_FDIM,
                 n_filters: int = 32, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        dim = 32
        self.n_output = 1

        def gin(in_dim):
            return GINConv(Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim)))

        self.conv1, self.bn1 = gin(num_features_xd), nn.BatchNorm1d(dim)
        self.conv2, self.bn2 = gin(dim), nn.BatchNorm1d(dim)
        self.conv3, self.bn3 = gin(dim), nn.BatchNorm1d(dim)
        self.conv4, self.bn4 = gin(dim), nn.BatchNorm1d(dim)
        self.conv5, self.bn5 = gin(dim), nn.BatchNorm1d(dim)
        self.fc1_xd = Linear(dim, output_dim)

        # cell-line 1D-CNN branch (width-agnostic; fc1_xt sized to cell_len)
        self.conv_xt_1 = nn.Conv1d(1, n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(cell_fc_in(cell_len, n_filters), output_dim)

        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(F.relu(self.conv3(x, edge_index)))
        x = self.bn4(F.relu(self.conv4(x, edge_index)))
        x = self.bn5(F.relu(self.conv5(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.dropout(F.relu(self.fc1_xd(x)), p=0.2, training=self.training)

        target = data.target[:, None, :]
        c = self.pool_xt_1(F.relu(self.conv_xt_1(target)))
        c = self.pool_xt_2(F.relu(self.conv_xt_2(c)))
        c = self.pool_xt_3(F.relu(self.conv_xt_3(c)))
        xt = self.fc1_xt(c.view(c.size(0), -1))

        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return torch.sigmoid(self.out(xc))
