"""DeepTTA (DeepTTC) architecture + ESPF tokenizer, isolated for the benchmark.

The upstream ``Step3_model.py`` hardwires ``input_dim_gene = 17737``, imports
heavy top-level deps (lifelines, prettytable) and a fixed ``cuda:0`` device, so
we re-declare the three tiny nn modules here (the ONLY vendored import is
``model_helper`` -- pure torch) and expose the expression MLP's input width as a
constructor argument. The ESPF/BPE tokenizer is copied verbatim from
``Step2_DataEncoding._drug2emb_encoder`` (drug-set-agnostic; reads the vendored
ESPF vocab). This keeps every change isolated in the adapter, upstream untouched.

Provenance: jianglikun/DeepTTC @ a657d56 (see vendor/MANIFEST.md).
"""
from __future__ import annotations

import codecs
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

# vendored pure-torch layer library
_VENDOR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "DeepTTC"))
import sys
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from model_helper import Encoder_MultipleLayers, Embeddings   # noqa: E402


# --------------------------------------------------------------------------- #
# ESPF / BPE drug tokenizer (copied from Step2_DataEncoding._drug2emb_encoder)
# --------------------------------------------------------------------------- #
_MAX_D = 50
_ESPF_DIR = os.path.join(_VENDOR, "ESPF")


class ESPFTokenizer:
    """Pretrained ChEMBL ESPF/BPE tokenizer; identical to upstream, drug-agnostic."""

    def __init__(self, espf_dir: str = _ESPF_DIR):
        from subword_nmt.apply_bpe import BPE
        codes = os.path.join(espf_dir, "drug_codes_chembl_freq_1500.txt")
        sub_csv = pd.read_csv(os.path.join(espf_dir, "subword_units_map_chembl_freq_1500.csv"))
        self.bpe = BPE(codecs.open(codes), merges=-1, separator="")
        idx2word = sub_csv["index"].values
        self.words2idx = dict(zip(idx2word, range(len(idx2word))))
        self.vocab_size = len(idx2word)   # == 2586, matches transformer input_dim_drug

    def encode(self, smile: str):
        """SMILES -> (token_ids[50] int64, mask[50] int64). OOV subwords -> [0]."""
        t1 = self.bpe.process_line(smile).split()
        try:
            i1 = np.asarray([self.words2idx[i] for i in t1])
        except KeyError:
            i1 = np.array([0])
        l = len(i1)
        if l < _MAX_D:
            ids = np.pad(i1, (0, _MAX_D - l), "constant", constant_values=0)
            mask = np.array([1] * l + [0] * (_MAX_D - l))
        else:
            ids = i1[:_MAX_D]
            mask = np.array([1] * _MAX_D)
        return ids.astype(np.int64), mask.astype(np.int64)


# --------------------------------------------------------------------------- #
# architecture (copied from Step3_model, input_dim_gene made configurable)
# --------------------------------------------------------------------------- #
class DrugTransformer(nn.Module):
    def __init__(self, vocab_size: int = 2586):
        super().__init__()
        emb_size, dropout = 128, 0.1
        self.emb = Embeddings(vocab_size, emb_size, _MAX_D, dropout)
        self.encoder = Encoder_MultipleLayers(8, emb_size, 512, 8, 0.1, 0.1)

    def forward(self, tokens, mask):
        ex_mask = mask.unsqueeze(1).unsqueeze(2)
        ex_mask = (1.0 - ex_mask) * -10000.0
        emb = self.emb(tokens.long())
        encoded = self.encoder(emb.float(), ex_mask.float())
        return encoded[:, 0]                     # 128-dim CLS-like token


class GeneMLP(nn.Module):
    """Expression MLP; ``input_dim_gene`` = number of genes we feed (909 for us)."""

    def __init__(self, input_dim_gene: int):
        super().__init__()
        hidden_dim_gene = 256
        dims = [input_dim_gene, 1024, 256, 64, hidden_dim_gene]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out_dim = hidden_dim_gene

    def forward(self, v):
        for l in self.predictor:
            v = F.relu(l(v.float()))
        return v


class DeepTTAModel(nn.Module):
    """Drug transformer + gene MLP -> concat -> regression head (predicts ln IC50)."""

    def __init__(self, input_dim_gene: int, vocab_size: int = 2586):
        super().__init__()
        self.model_drug = DrugTransformer(vocab_size)
        self.model_gene = GeneMLP(input_dim_gene)
        self.dropout = nn.Dropout(0.1)
        dims = [128 + self.model_gene.out_dim, 1024, 1024, 512, 1]
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, tokens, mask, gene):
        vD = self.model_drug(tokens, mask)
        vP = self.model_gene(gene)
        vf = torch.cat((vD, vP), 1)
        for i, l in enumerate(self.predictor):
            vf = l(vf) if i == len(self.predictor) - 1 else F.relu(self.dropout(l(vf)))
        return vf
