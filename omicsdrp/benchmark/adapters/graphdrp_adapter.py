"""GraphDRP adapter — native inputs, our folds.

Native cell input : 735 binary genomic features (mutation + CNV) from
                    ``PANCANCER_Genetic_feature.csv``, fed to a 1-D CNN.
Native drug input : SMILES -> molecular graph (78-dim atom features) via
                    GraphDRP's own ``preprocess.smile_to_graph``, encoded by GIN.
Label             : ln(IC50). GraphDRP's paper squashes the target with
                    sigmoid(0.1*x); we keep the shared ln target instead so the
                    reported metric is on the same scale as every other model
                    here. Only the label transform differs from upstream — the
                    inputs and architecture are untouched.

Run inside the ``benchmark_graphdrp`` env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402
import common  # noqa: E402

UPSTREAM = common.VENDOR_DIR / "GraphDRP"
sys.path.insert(0, str(UPSTREAM))

from models.ginconv import GINConvNet  # noqa: E402
from preprocess import smile_to_graph  # noqa: E402


class GraphDRPAdapter(_base.BaseAdapter):
    name = "graphdrp"
    # upstream training.py defaults: --lr 1e-4, --train_batch 1024
    default_lr = 1e-4
    default_batch_size = 1024

    def load_native(self, export):
        feat = pd.read_csv(UPSTREAM / "data" / "PANCANCER_Genetic_feature.csv")
        # long -> wide binary matrix, exactly how GraphDRP builds it
        wide = (feat.pivot_table(index="cosmic_sample_id", columns="genetic_feature",
                                 values="is_mutated", aggfunc="max")
                    .fillna(0).astype(np.float32))
        self.n_cell_features = wide.shape[1]

        cosmic = export.cell_cosmic_ids
        self.cell_ok = np.array([c in wide.index for c in cosmic])
        mat = np.zeros((len(cosmic), self.n_cell_features), dtype=np.float32)
        present = np.where(self.cell_ok)[0]
        mat[present] = wide.loc[cosmic[present]].values
        self.cell_feat = mat

        # Molecular graphs from our SMILES, using GraphDRP's own featuriser.
        self.graphs = []
        ok = []
        for s in export.drug_smiles:
            try:
                c_size, features, edge_index = smile_to_graph(str(s))
                if c_size == 0 or len(edge_index) == 0:
                    raise ValueError("empty graph")
                self.graphs.append((
                    torch.tensor(np.asarray(features), dtype=torch.float),
                    torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                ))
                ok.append(True)
            except Exception:
                self.graphs.append(None)
                ok.append(False)
        self.drug_ok = np.asarray(ok)
        print(f"  cells with genomic features: {self.cell_ok.sum()}/{len(cosmic)} "
              f"({self.n_cell_features} features) | graphs: {self.drug_ok.sum()}"
              f"/{len(export.drug_smiles)}")

    def usable_pairs(self, export, pair_idx):
        p = export.pairs[pair_idx]
        return pair_idx[self.cell_ok[p[:, 0]] & self.drug_ok[p[:, 1]]]

    def fit_scaler(self, export, fit_idx):
        # Binary indicators — GraphDRP feeds them raw, so no scaling is applied.
        return None

    def make_loader(self, export, pair_idx, scaler, *, batch_size, shuffle):
        items = []
        for pi in pair_idx:
            c, d = export.pairs[pi]
            x, edge_index = self.graphs[d]
            items.append(Data(
                x=x, edge_index=edge_index,
                y=torch.tensor([float(export.y[pi])]),
                target=torch.tensor(self.cell_feat[c]).unsqueeze(0),
            ))
        # upstream training.py builds its loaders with defaults (drop_last=False)
        return DataLoader(items, batch_size=batch_size, shuffle=shuffle)

    def build_model(self):
        return GINConvNet()

    def forward(self, model, batch):
        batch = batch.to(self.device)
        pred, _ = model(batch)
        return pred.squeeze(-1), batch.y.view(-1)


if __name__ == "__main__":
    _base.run(GraphDRPAdapter())
