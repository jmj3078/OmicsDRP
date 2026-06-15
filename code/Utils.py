import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import torch.nn as nn


class Omics_Drug_Combination_Dataset(Dataset):
    def __init__(self, gene_data, drug_data, IC50, sample_drug_indices):
        self.gene_data = gene_data
        self.drug_data = drug_data
        self.IC50 = IC50
        self.sample_drug_indices = sample_drug_indices
    def __len__(self) : 
        return len(self.sample_drug_indices)
    
    def __getitem__(self, idx):
        sample_idx, drug_idx = self.sample_drug_indices[idx]
        sample_features = {gene: self.gene_data[gene][sample_idx] for gene in self.gene_data}
        drug_features = self.drug_data[drug_idx]
        ic50_value = self.IC50[sample_idx, drug_idx]

        return sample_features, drug_features, ic50_value, (sample_idx, drug_idx)



def load_drug_info(dataset_path):
    IC50 = pd.read_csv(f"{dataset_path}/IC50_GDSC2.csv", index_col=0)
    Drug_meta = pd.read_csv(f"{dataset_path}/TargetDrugs_with_MorganFingerprint_GDSC2_512.txt", sep='\t')
    Drug_IDs = []
    Drug_info = []
    for _, row in Drug_meta.iterrows():
        drug_id = row['DRUG_ID']
        Drug_IDs.append(drug_id)
        morgan_fingerprint = row['Morgan_Fingerprint'].split(',')
        Drug_info.append(torch.tensor([float(i) for i in morgan_fingerprint]))
    IC50_not_nan_args = np.argwhere(~np.isnan(IC50.values))
    IC50_tensor = torch.from_numpy(IC50.values).type(torch.FloatTensor)
    return Drug_IDs, Drug_info, IC50_tensor, IC50_not_nan_args



def load_Gene_Pathway_data(dataset_path):
    gene_data = torch.load(f"{dataset_path}/PGKB_Gene_data_dict.pth")
    return gene_data



def make_fold_datasets(
    raw_gene_data,
    raw_drug_data,
    raw_IC50,
    raw_indices,
    train_index,
    val_index,
):
    train_pairs = [raw_indices[i] for i in train_index]
    val_pairs   = [raw_indices[i] for i in val_index]
    train_sample_indices = sorted({s for (s, d) in train_pairs})
    
    gene_data_scaled = {}
    for gene, mat in raw_gene_data.items():
        mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else mat
        scaler = StandardScaler()
        scaler.fit(mat_np[train_sample_indices])
        mat_scaled = scaler.transform(mat_np)
        gene_data_scaled[gene] = torch.tensor(mat_scaled, dtype=torch.float32)

    drug_data_final = raw_drug_data

    train_dataset = Omics_Drug_Combination_Dataset(
        gene_data=gene_data_scaled,
        drug_data=drug_data_final,
        IC50=raw_IC50,
        sample_drug_indices=train_pairs,
    )
    val_dataset = Omics_Drug_Combination_Dataset(
        gene_data=gene_data_scaled,
        drug_data=drug_data_final,
        IC50=raw_IC50,
        sample_drug_indices=val_pairs,
    )
    return train_dataset, val_dataset



class EarlyStopping:
    def __init__(self, patience, path, delta=0, verbose=False):
        self.patience = patience
        self.path = path
        self.delta = delta
        self.verbose = verbose
        self.best_val_loss = None
        self.count = 0
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def build_adamw_optimizer(model, lr, weight_decay):
    no_decay_kw = ("bias", "bn", "batchnorm", "layernorm", "ln", "norm")
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in n.lower() for k in no_decay_kw) else decay).append(p)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, 
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)



def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    for sample_features, drug_features, labels, _ in dataloader:
        sample_features = {k: v.to(device) for k, v in sample_features.items()}
        drug_features = drug_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sample_features, drug_features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_labels.append(labels.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    rmse = root_mean_squared_error(all_labels, all_outputs)
    return running_loss / len(dataloader), rmse



def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for sample_features, drug_features, labels, _ in dataloader:
            sample_features = {k: v.to(device) for k, v in sample_features.items()}
            drug_features = drug_features.to(device)
            labels = labels.to(device)

            outputs = model(sample_features, drug_features)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    rmse = root_mean_squared_error(all_labels, all_outputs)
    return running_loss / len(dataloader), rmse



def evaluate_with_embeddings(model, dataloader, val_subset, device):
    model.eval()
    all_labels = []
    all_outputs = []
    all_sample_indices = []
    all_drug_indices = []

    with torch.no_grad():
        for sample_features, drug_features, labels, (sample_idx, drug_idx) in dataloader:
            sample_features = {k: v.to(device) for k, v in sample_features.items()}
            drug_features = drug_features.to(device)
            labels = labels.to(device)
            sample_idx = sample_idx.to(device)
            drug_idx = drug_idx.to(device)

            outputs = model(sample_features, drug_features)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.squeeze(-1).cpu().numpy())
            all_sample_indices.append(sample_idx.cpu().numpy())
            all_drug_indices.append(drug_idx.cpu().numpy())
            
    all_labels = np.concatenate(all_labels).tolist()
    all_outputs = np.concatenate(all_outputs).tolist()
    all_sample_indices = np.concatenate(all_sample_indices).tolist()
    all_drug_indices = np.concatenate(all_drug_indices).tolist()

    embeddings = {'drug_embedding': [], 'cell_embedding': [], 'gene_attention_score': []}
    gene_data = val_subset.gene_data
    drug_data = val_subset.drug_data
    drug_data_tensor = torch.stack(drug_data).to(device)
    with torch.no_grad():
        drug_embedding = model.drug_embedding(drug_data_tensor)
        gene_embeddings = []
        for gene, features in gene_data.items():
            features = features.to(device)
            gene_emb = model.gene_embedding_layers[gene](features)
            gene_embeddings.append(gene_emb)
            
        gene_embeddings = torch.stack(gene_embeddings, dim=1)
        cell_embedding, gene_attention_score = model.cellular_attention_layers(gene_embeddings)

        embeddings = {'drug_embedding': drug_embedding.cpu().numpy(),
                      'cell_embedding': cell_embedding.cpu().numpy(),
                      'gene_attention_score': gene_attention_score.cpu().numpy()}

    return all_labels, all_outputs, all_sample_indices, all_drug_indices, embeddings



def plot_with_loss_curve(metric1, metric2, ylabel, num_folds=5):
    all_fold_results = []
    for fold in range(1, num_folds + 1):
        results_df = pd.read_pickle(f"./Training_History/Training_History_Early_Stopping_Fold_{fold}.pkl")
        all_fold_results.append(results_df)

    data = pd.concat(all_fold_results, ignore_index=True)

    mean1 = data.groupby('Epoch')[metric1].mean()
    std1 = data.groupby('Epoch')[metric1].std()
    mean2 = data.groupby('Epoch')[metric2].mean()
    std2 = data.groupby('Epoch')[metric2].std()
    epochs = mean1.index

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mean1, label=f'Mean {metric1}')
    plt.fill_between(epochs, mean1 - std1, mean1 + std1, alpha=0.3)
    plt.plot(epochs, mean2, label=f'Mean {metric2}')
    plt.fill_between(epochs, mean2 - std2, mean2 + std2, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{metric1} and {metric2} per Epoch (5-Fold CV)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Figures/{ylabel}_5_fold_cv.png')
    plt.close()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            if fan_in != 0 and fan_out != 0:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)



def print_settings(path_dir, batch_size, num_epochs, embedding_dim, gene_embed_dim, num_heads, lr, weight_decay, dropout):
    keys = ["Data Path", "Batch Size", "Num Epochs", 
            "Embedding Dimension", "Embedding Dim of Gene", "Num Heads", 
            "Learning Rate", "Weight Decay", "Dropout"]
    values = [path_dir, batch_size, num_epochs, 
              embedding_dim, gene_embed_dim, num_heads, 
              lr, weight_decay, dropout]
    
    print("=" * 47)
    print(f"{'Hyperparameter':<21} | {'Value':<21}")
    print("-" * 47)
    for key, value in zip(keys, values):
        print(f"{key:<21} | {str(value):<21}")
    print("=" * 47 + "\n")

