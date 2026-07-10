"""Cell-branch encoders (ablation axis 2: attention vs. plain MLP).

Both encoders share the same per-gene embedding front-end (each gene's omics
vector -> ``gene_embed_dim``) and produce a fixed ``embedding_dim`` cell
representation, so they are drop-in interchangeable in :class:`DRPModel`.

  * :class:`AttentionCellEncoder` -- the original hierarchical self-attention
    over the ~909 gene tokens.
  * :class:`MLPCellEncoder` -- a parameter-comparable baseline that flattens the
    gene embeddings and runs an MLP, i.e. no cross-gene attention. This isolates
    the contribution of the attention mechanism.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneEmbeddingLayer(nn.Module):
    """Per-gene omics vector -> gene_embed_dim. ``input_dim`` == n selected omics."""

    def __init__(self, gene_embed_dim: int, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, gene_embed_dim)
        self.batchnorm = nn.BatchNorm1d(gene_embed_dim)
        self.fc2 = nn.Linear(gene_embed_dim, gene_embed_dim)
        self.layernorm = nn.LayerNorm(gene_embed_dim)

    def forward(self, x):
        x = F.relu(self.batchnorm(self.fc1(x)))
        x = self.layernorm(self.fc2(x))
        return x


class _GeneFrontEnd(nn.Module):
    """Shared: turns the omics dict into a [B, n_gene, gene_embed_dim] tensor."""

    def __init__(self, genes, gene_embed_dim: int, input_dim: int):
        super().__init__()
        self.genes = list(genes)
        self.gene_embedding_layers = nn.ModuleDict(
            {g: GeneEmbeddingLayer(gene_embed_dim, input_dim) for g in self.genes})

    def forward(self, gene_data: dict) -> torch.Tensor:
        embs = [self.gene_embedding_layers[g](gene_data[g]) for g in self.genes]
        return torch.stack(embs, dim=1)  # [B, n_gene, gene_embed_dim]


class AttentionCellEncoder(nn.Module):
    def __init__(self, genes, n_gene, gene_embed_dim, num_heads, embedding_dim,
                 dropout, input_dim):
        super().__init__()
        self.front = _GeneFrontEnd(genes, gene_embed_dim, input_dim)
        self.attention = nn.MultiheadAttention(gene_embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(gene_embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(gene_embed_dim, gene_embed_dim), nn.ReLU(),
            nn.Linear(gene_embed_dim, int(0.5 * embedding_dim)))
        self.norm2 = nn.LayerNorm(int(0.5 * embedding_dim))
        self.fc = nn.Linear(int(0.5 * embedding_dim) * n_gene, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # cached for interpretability (Stage 3)

    def forward(self, gene_data):
        g = self.front(gene_data)
        attn_out, attn_w = self.attention(g, g, g)
        x = self.norm1(g + attn_out)
        x = self.norm2(self.ff(x))
        x = x.flatten(start_dim=1)
        x = self.dropout(self.ln(self.fc(x)))
        self.last_attn = attn_w
        return x


class MLPCellEncoder(nn.Module):
    """Attention-free baseline: flatten gene embeddings -> MLP -> embedding_dim."""

    def __init__(self, genes, n_gene, gene_embed_dim, embedding_dim, dropout, input_dim):
        super().__init__()
        self.front = _GeneFrontEnd(genes, gene_embed_dim, input_dim)
        self.net = nn.Sequential(
            nn.Linear(gene_embed_dim * n_gene, embedding_dim), nn.ReLU(),
            nn.LayerNorm(embedding_dim), nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout))
        self.last_attn = None

    def forward(self, gene_data):
        g = self.front(gene_data)          # [B, n_gene, gene_embed_dim]
        x = g.flatten(start_dim=1)
        return self.net(x)


def build_cell_encoder(kind: str, genes, n_gene, gene_embed_dim, num_heads,
                       embedding_dim, dropout, input_dim: int):
    """Factory. ``input_dim`` = number of selected omics modalities."""
    if kind == "attention":
        return AttentionCellEncoder(genes, n_gene, gene_embed_dim, num_heads,
                                    embedding_dim, dropout, input_dim)
    if kind == "mlp":
        return MLPCellEncoder(genes, n_gene, gene_embed_dim, embedding_dim,
                              dropout, input_dim)
    raise ValueError(f"Unknown cell_encoder {kind!r}")
