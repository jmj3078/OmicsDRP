import pandas as pdos
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import argparse
import os
from Model import *
from Utils import *

set_seed(2024)

parser = argparse.ArgumentParser()
# parser.add_argument('--target_fold', type=int, required=True)       # using job schedular --array
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--drug_input_dim', type=int, default=512)
parser.add_argument('--n_gene', type=int, default=909)
parser.add_argument('--gene_embed_dim', type=int, default=8)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=7)
parser.add_argument('--dataset_path', type=str)
args = parser.parse_args()

# target_fold = args.target_fold        # using job schedular --array
batch_size = args.batch
num_epochs = args.num_epochs
embedding_dim = args.embedding_dim
drug_input_dim = args.drug_input_dim
n_gene = args.n_gene
gene_embed_dim = args.gene_embed_dim
num_heads = args.num_heads
lr = args.lr
weight_decay = args.weight_decay
dropout = args.dropout
dataset_path = args.dataset_path
patience = args.patience
path_dir = os.path.basename(dataset_path)


gene_data = load_Gene_Pathway_data(dataset_path)
unique_genes = list(gene_data.keys())
Drug_IDs, Drug_info, IC50_tensor, IC50_not_nan_args = load_drug_info(dataset_path)
dataset = Omics_Drug_Combination_Dataset(gene_data=gene_data, 
                                         drug_data=Drug_info, 
                                         IC50=IC50_tensor, 
                                         sample_drug_indices=IC50_not_nan_args)

os.makedirs(f"./Training_History/", exist_ok=True)
os.makedirs(f"./Models/", exist_ok=True)

# Training Result
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
results = {
    'Fold': [],
    'Epoch': [],
    'Train Loss': [],
    'Val Loss': [],
    'Train RMSE': [],
    'Val RMSE': []
}

print_settings(path_dir, batch_size, num_epochs, embedding_dim, 
               gene_embed_dim, num_heads, lr, weight_decay, dropout)

for train_index, val_index in kf.split(dataset):
    fold += 1
    # if fold != target_fold:     # using job schedular --array
    #     continue
    best_ckpt_path = f"./Models/Model_Fold_{fold}.pth"
    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

    print(f"Fold {fold}")
    print("--------------------------------")
    train_subset, val_subset = make_fold_datasets(gene_data, Drug_info, IC50_tensor, 
                                                  IC50_not_nan_args, train_index, val_index)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiarachialAttentionModel(unique_genes, 
                                     embedding_dim, 
                                     drug_input_dim, 
                                     n_gene, 
                                     gene_embed_dim, 
                                     num_heads, 
                                     dropout)
    model.apply(initialize_weights)
    criterion = nn.MSELoss()
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    early_stopper = EarlyStopping(
        patience=patience,
        path=best_ckpt_path,
        delta=0,
        verbose=True
    )

    for epoch in range(num_epochs):
        train_loss, train_rmse = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_rmse = evaluate(model, val_dataloader, criterion, device)
        
        results['Fold'].append(fold)
        results['Epoch'].append(epoch+1)
        results['Train Loss'].append(train_loss)
        results['Val Loss'].append(val_loss)
        results['Train RMSE'].append(train_rmse)
        results['Val RMSE'].append(val_rmse)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')

        early_stopper(val_rmse, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    results_df = pd.DataFrame(results)
    results_df.to_pickle(f"./Training_History/Training_History_Fold_{fold}.pkl")