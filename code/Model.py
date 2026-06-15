import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout):
        super(DrugEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.batchnorm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.batchnorm(self.fc1(x))))
        x = self.dropout(self.layernorm(self.fc2(x)))
        return x


class GeneEmbeddingLayer(nn.Module):
    def __init__(self, gene_embed_dim, input_dim=4):
        super(GeneEmbeddingLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, gene_embed_dim)
        self.batchnorm = nn.BatchNorm1d(gene_embed_dim)
        self.fc2 = nn.Linear(gene_embed_dim, gene_embed_dim)
        self.layernorm = nn.LayerNorm(gene_embed_dim)
        
    def forward(self, x):
        x = F.relu(self.batchnorm(self.fc1(x)))
        x = self.layernorm(self.fc2(x))
        return x


class CellularAttentionLayer(nn.Module):
    def __init__(self, gene_embed_dim, num_heads, flatten_dim, embedding_dim, dropout):
        super(CellularAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=gene_embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(gene_embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(gene_embed_dim, gene_embed_dim),
            nn.ReLU(),
            nn.Linear(gene_embed_dim, int(0.5*embedding_dim))
            )
        self.norm2 = nn.LayerNorm(int(0.5*embedding_dim))

        self.fc = nn.Linear(flatten_dim, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_embeddings):
        attn_output, attn_weights = self.attention(gene_embeddings, gene_embeddings, gene_embeddings)
        x = self.norm1(gene_embeddings + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(ff_output)

        x = x.flatten(start_dim=1)
        x = self.dropout(self.ln(self.fc(x)))
        return x, attn_weights
    
    
class ResponsePredictionLayer(nn.Module):
    def __init__(self, embedding_dim, dropout, output_dim=1):
        super(ResponsePredictionLayer, self).__init__()
        self.fc1 = nn.Linear(2*embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)        
        self.fc4 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, cell_embedding, drug_embedding):
        x = torch.cat((cell_embedding, drug_embedding), dim=1)
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class HiarachialAttentionModel(nn.Module):
    def __init__(self, unique_genes, embedding_dim, drug_input_dim, n_gene, gene_embed_dim, num_heads, dropout):
        super(HiarachialAttentionModel, self).__init__()

        # Drug Embedding Layer
        self.drug_embedding = DrugEmbedding(drug_input_dim, embedding_dim, dropout)

        # Gene Embedding Layers
        self.gene_embedding_layers = nn.ModuleDict({
            gene: GeneEmbeddingLayer(gene_embed_dim) for gene in unique_genes
        })

        # Cell-line Attention Layer
        self.cellular_attention_layers = CellularAttentionLayer(gene_embed_dim, num_heads, int(0.5*embedding_dim)*n_gene, embedding_dim, dropout)

        # Prediction Layer
        self.response_prediction = ResponsePredictionLayer(embedding_dim, dropout)
    
    def forward(self, gene_data, drug_data):
        drug_embedding = self.drug_embedding(drug_data)
        gene_embeddings = []
        for gene in gene_data:
            gene_embedding = self.gene_embedding_layers[gene](gene_data[gene])
            gene_embeddings.append(gene_embedding)

        gene_embeddings = torch.stack(gene_embeddings, dim=1)
        cell_embedding, _ = self.cellular_attention_layers(gene_embeddings)
        output = self.response_prediction(cell_embedding, drug_embedding)
        return output
