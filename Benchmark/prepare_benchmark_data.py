import os
import pandas as pd
import numpy as np
import pickle
import json

def main():
    print("Preparing benchmark datasets...")
    our_data_dir = "/project/OmicsDRP_Review/Benchmark/our_data"
    
    # Load our datasets
    df_ic50 = pd.read_csv(os.path.join(our_data_dir, "IC50_GDSC2.csv"))
    df_meta = pd.read_csv(os.path.join(our_data_dir, "Cell_line_meta.csv"))
    
    # Load drug info
    df_drug = pd.read_csv(os.path.join(our_data_dir, "TargetDrugs_with_MorganFingerprint_GDSC2_512.txt"), sep="\t")
    # Clean up column names in drug metadata
    df_drug.columns = df_drug.columns.str.strip()
    
    # Create mappings
    model_to_cosmic = dict(zip(df_meta['Model_ID'], df_meta['COSMIC_ID']))
    model_to_name = dict(zip(df_meta['Model_ID'], df_meta['Cell_line_Name']))
    model_to_tcga = dict(zip(df_meta['Model_ID'], df_meta['TCGA_Classfication']))
    model_to_tissue = dict(zip(df_meta['Model_ID'], df_meta['Tissue']))
    model_to_subtype = dict(zip(df_meta['Model_ID'], df_meta['Tissue_subtype']))
    
    drug_id_to_name = dict(zip(df_drug['DRUG_ID'].astype(str), df_drug['DRUG_NAME']))
    drug_id_to_smiles = dict(zip(df_drug['DRUG_ID'].astype(str), df_drug['SMILE']))
    drug_id_to_pubchem = dict(zip(df_drug['DRUG_ID'].astype(str), df_drug['PubChem_ID']))
    
    # Meltdown our IC50 matrix into pairwise format
    rows = []
    for idx, row in df_ic50.iterrows():
        model_id = row['SANGER_MODEL_ID']
        cosmic_id = model_to_cosmic.get(model_id)
        cell_name = model_to_name.get(model_id)
        tcga = model_to_tcga.get(model_id, 'UNCLASSIFIED')
        tissue = model_to_tissue.get(model_id, 'unknown')
        subtype = model_to_subtype.get(model_id, 'unknown')
        
        if not cosmic_id:
            continue
            
        for col in df_ic50.columns:
            if col == 'SANGER_MODEL_ID':
                continue
            ic50_val = row[col]
            if pd.isna(ic50_val):
                continue
                
            drug_name = drug_id_to_name.get(str(col))
            if not drug_name:
                continue
                
            rows.append({
                'SANGER_MODEL_ID': model_id,
                'COSMIC_ID': int(cosmic_id),
                'Cell_Line_Name': cell_name,
                'TCGA_DESC': tcga,
                'Tissue': tissue,
                'Tissue_Subtype': subtype,
                'DRUG_ID': int(col),
                'Drug_Name': drug_name,
                'LN_IC50': ic50_val
            })
            
    df_pairs = pd.DataFrame(rows)
    print(f"Constructed {len(df_pairs)} pairwise drug-cell response entries.")

    # Generate random train/val/test splits (80/10/10)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df_pairs))
    train_size = int(0.8 * len(df_pairs))
    val_size = int(0.1 * len(df_pairs))
    
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:train_size+val_size]
    test_idx = shuffled_indices[train_size+val_size:]
    
    df_train = df_pairs.iloc[train_idx]
    df_val = df_pairs.iloc[val_idx]
    df_test = df_pairs.iloc[test_idx]
    
    # ----------------------------------------------------
    # 1. Prepare GraphDRP Data
    # ----------------------------------------------------
    graphdrp_data_dir = "/project/OmicsDRP_Review/Benchmark/GraphDRP/data"
    
    # GraphDRP PANCANCER_IC.csv columns:
    # Drug name,Drug Id,Cell line name,Cosmic sample Id,TCGA classification,Tissue,Tissue sub-type,IC Result ID,IC50,AUC,Max conc,RMSE,Z score,Dataset version
    graphdrp_rows = []
    for _, item in df_pairs.iterrows():
        graphdrp_rows.append([
            item['Drug_Name'],
            item['DRUG_ID'],
            item['Cell_Line_Name'],
            item['COSMIC_ID'],
            item['TCGA_DESC'],
            item['Tissue'],
            item['Tissue_Subtype'],
            12345,  # Dummy Result ID
            item['LN_IC50'],
            0.98,   # Dummy AUC
            2.0,    # Dummy Max conc
            0.02,   # Dummy RMSE
            0.0,    # Dummy Z score
            17      # Dummy Dataset version
        ])
    df_graphdrp_ic = pd.DataFrame(graphdrp_rows, columns=[
        'Drug name','Drug Id','Cell line name','Cosmic sample Id','TCGA classification','Tissue','Tissue sub-type','IC Result ID','IC50','AUC','Max conc','RMSE','Z score','Dataset version'
    ])
    df_graphdrp_ic.to_csv(os.path.join(graphdrp_data_dir, "PANCANCER_IC.csv"), index=False)
    
    # GraphDRP drug_smiles.csv columns: name,CID,CanonicalSMILES,IsomericSMILES
    graphdrp_smiles = []
    for _, item in df_drug.iterrows():
        drug_name = item['DRUG_NAME']
        pubchem_id = item['PubChem_ID']
        smile = item['SMILE']
        if pd.isna(pubchem_id):
            pubchem_id = 999999
        graphdrp_smiles.append([drug_name, int(pubchem_id), smile, smile])
    df_graphdrp_smiles = pd.DataFrame(graphdrp_smiles, columns=['name','CID','CanonicalSMILES','IsomericSMILES'])
    df_graphdrp_smiles.to_csv(os.path.join(graphdrp_data_dir, "drug_smiles.csv"), index=False)
    
    # ----------------------------------------------------
    # 2. Prepare DeepTTA Data
    # ----------------------------------------------------
    deeptta_data_dir = "/project/OmicsDRP_Review/Benchmark/DeepTTA/GDSC_data"
    
    # smile_inchi.csv columns: drug_id,smiles,inchi
    deeptta_smiles = []
    for _, item in df_drug.iterrows():
        deeptta_smiles.append([item['DRUG_ID'], item['SMILE'], 'None'])
    df_deeptta_smiles = pd.DataFrame(deeptta_smiles, columns=['drug_id','smiles','inchi'])
    df_deeptta_smiles.to_csv(os.path.join(deeptta_data_dir, "smile_inchi.csv"), index=False)
    
    # Drug_listTue_Aug10_2021.csv columns: drug_id,Name,Synonyms,Targets,Target pathway,PubCHEM,Sample Size,Count,jlk_X,jlk_XX
    deeptta_drug_list = []
    for _, item in df_drug.iterrows():
        pubchem = item['PubChem_ID']
        if pd.isna(pubchem):
            pubchem = '12345'
        else:
            pubchem = str(int(pubchem))
        deeptta_drug_list.append([
            item['DRUG_ID'],
            item['DRUG_NAME'],
            item['SYNONYMS'],
            item['TARGET'],
            item['TARGET_PATHWAY'],
            pubchem,
            'GDSC2',
            800,
            'SANGER',
            565
        ])
    df_deeptta_dl = pd.DataFrame(deeptta_drug_list, columns=[
        'drug_id','Name','Synonyms','Targets','Target pathway','PubCHEM','Sample Size','Count','jlk_X','jlk_XX'
    ])
    df_deeptta_dl.to_csv(os.path.join(deeptta_data_dir, "Drug_listTue_Aug10_2021.csv"), index=False)
    
    # GDSC2_fitted_dose_response_25Feb20.xlsx
    # We will save this as an excel sheet matching pairfile expected columns
    # expected columns: ['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']
    df_pairs_deeptta = df_pairs[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']].copy()
    df_pairs_deeptta.to_excel(os.path.join(deeptta_data_dir, "GDSC2_fitted_dose_response_25Feb20.xlsx"), index=False)
    
    # Cell_line_RMA_proc_basalExp.txt: Rows = 909 genes, Columns = DATA.COSMIC_ID
    # Load our RNA expression data
    df_rna_raw = pd.read_csv("/project/OmicsDRP_Review/data/raw_data/RNA_PGKB.csv", index_col=0)
    # Map index to DATA.COSMIC_ID
    cosmic_ids = df_meta['COSMIC_ID'].astype(int).tolist()
    column_names = ['DATA.' + str(cid) for cid in cosmic_ids]
    
    # We need to write this with genes as index/rows and cell lines as columns
    df_rna_mapped = df_rna_raw.T
    df_rna_mapped.columns = column_names
    # Reset index to make gene name a column
    df_rna_mapped = df_rna_mapped.reset_index()
    df_rna_mapped.rename(columns={'index': 'Gene'}, inplace=True)
    df_rna_mapped.to_csv(os.path.join(deeptta_data_dir, "Cell_line_RMA_proc_basalExp.txt"), sep="\t", index=False)
    
    # Modify Step3_model.py to set input_dim_gene = 909
    step3_path = "/project/OmicsDRP_Review/Benchmark/DeepTTA/Step3_model.py"
    with open(step3_path, 'r') as f:
        content = f.read()
    content = content.replace("input_dim_gene = 17737", "input_dim_gene = 909")
    with open(step3_path, 'w') as f:
        f.write(content)
    print("Modified DeepTTA Step3_model.py input_dim_gene to 909.")
    
    # ----------------------------------------------------
    # 3. Prepare PaccMann Data
    # ----------------------------------------------------
    paccmann_ic50_dir = "/project/OmicsDRP_Review/Benchmark/PaccMann/examples/IC50"
    os.makedirs(os.path.join(paccmann_ic50_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(paccmann_ic50_dir, "splitted_data"), exist_ok=True)
    
    # Train sensitivity CSV: drug, cell_line, IC50
    df_train_pm = df_train[['SMILES' if False else 'Drug_Name', 'COSMIC_ID', 'LN_IC50']].copy()
    df_train_pm.columns = ['drug', 'cell_line', 'IC50']
    df_train_pm.to_csv(os.path.join(paccmann_ic50_dir, "splitted_data", "train_sensitivity.csv"), index=False)
    
    # Test sensitivity CSV: drug, cell_line, IC50
    df_test_pm = df_test[['SMILES' if False else 'Drug_Name', 'COSMIC_ID', 'LN_IC50']].copy()
    df_test_pm.columns = ['drug', 'cell_line', 'IC50']
    df_test_pm.to_csv(os.path.join(paccmann_ic50_dir, "splitted_data", "test_sensitivity.csv"), index=False)
    
    # GEP CSV: index is cell line ID, columns are genes
    df_gep = df_rna_raw.copy()
    df_gep.index = cosmic_ids
    df_gep.to_csv(os.path.join(paccmann_ic50_dir, "data", "gene_expression.csv"))
    
    # SMI CSV: drug name \t SMILES (with no header)
    df_smi = df_drug[['DRUG_NAME', 'SMILE']].copy()
    df_smi.to_csv(os.path.join(paccmann_ic50_dir, "data", "drugs.smi"), sep="\t", header=False, index=False)
    
    # Gene list pickle: list of 909 genes
    genes_list = df_rna_raw.columns.tolist()
    with open(os.path.join(paccmann_ic50_dir, "data", "genes_list.pkl"), "wb") as f:
        pickle.dump(genes_list, f)
        
    # Generate smiles_language folder with vocabulary
    # Since PaccMann uses a smiles language folder, we can initialize a simple SMILESTokenizer in python and save it there
    # We will write a small python snippet to run under benchmark_paccmann env to save vocabulary
    print("Benchmark data preparation complete.")

if __name__ == "__main__":
    main()
