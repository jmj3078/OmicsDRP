import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import argparse
import os
from Model import *
from Utils import *
import gc


parser = argparse.ArgumentParser()
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
parser.add_argument('--dataset_path', type=str)
args = parser.parse_args()


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
path_dir = os.path.basename(dataset_path)

gene_data = load_Gene_Pathway_data(dataset_path)
unique_genes = list(gene_data.keys())
Drug_IDs, Drug_info, IC50_tensor, IC50_not_nan_args = load_drug_info(dataset_path)
dataset = Omics_Drug_Combination_Dataset(gene_data=gene_data, 
                                         drug_data=Drug_info, 
                                         IC50=IC50_tensor, 
                                         sample_drug_indices=IC50_not_nan_args)


os.makedirs(f"./Embeddings/", exist_ok=True)
os.makedirs(f"./Predictions/", exist_ok=True)
os.makedirs(f"./Figures/", exist_ok=True)


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

for train_index, val_index in kf.split(dataset):
    fold += 1
    best_ckpt_path = f"./Models/model_{fold}.pth"

    _, val_subset = make_fold_datasets(gene_data, Drug_info, IC50_tensor, 
                                       IC50_not_nan_args, train_index, val_index)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    model = HiarachialAttentionModel(unique_genes, 
                                     embedding_dim, 
                                     drug_input_dim, 
                                     n_gene, 
                                     gene_embed_dim, 
                                     num_heads, 
                                     dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_ckpt_path = f"./Models/Model_Fold_{fold}.pth"
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()
    criterion = nn.MSELoss()

    val_labels, val_predictions, val_sample_idices, val_drug_indices, embeddings = evaluate_with_embeddings(model, val_dataloader, val_subset, device)

    prediction_df = pd.DataFrame([val_sample_idices, val_drug_indices, val_labels, val_predictions]).T
    prediction_df.columns = ['Sample_index', 'Drug_index', 'True', 'Pred']
    prediction_df.to_pickle(f"./Predictions/Predictions_Fold_{fold}.pkl")
    print(f"Fold {fold} prediction_df saved at: ./Predictions/Predictions_Fold_{fold}.pkl")
    
    torch.save(embeddings, f"./Embeddings/Embeddings_Fold_{fold}.pth")
    print(f"Fold {fold} embeddings saved at: ./Embeddings/Embeddings_Fold_{fold}.pth")
    
    del model, val_dataloader, val_labels, val_predictions, val_sample_idices, val_drug_indices, prediction_df, embeddings
    gc.collect()
    torch.cuda.empty_cache()


results_df = pd.read_pickle(f"./Training_History/Training_History_Fold_{fold}.pkl")
plot_with_loss_curve('Train Loss', 'Val Loss', 'Loss', f'./Figures/Loss_5_fold_cv.png')
plot_with_loss_curve('Train RMSE', 'Val RMSE', 'RMSE', f'./Figures/RMSE_5_fold_cv.png')