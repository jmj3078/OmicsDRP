# **OmicsDRP**
Precision medicine aims to identify optimal treatments based on the molecular profiles of individual patients. However, predicting drug response remains a major challenge due to intratumoral heterogeneity. To fully capture this biological complexity, integrating comprehensive multi-omics profiles has become essential, driving the necessity for methodologies that can leverage complex biological and chemical data modalities. In this study, we propose OmicsDRP, a dual-branch deep learning model that predicts drug response by integrating cellular multi-omics profiles and drug structures. This architecture enables each branch to learn informative representations from distinct data modalities, thereby effectively capturing cross-modality interactions governing drug sensitivity. Through cross-validation on the Genomics of Drug Sensitivity in Cancer (GDSC) dataset, OmicsDRP demonstrates consistent improvements in predictive performance over conventional machine learning methods and exhibits competitive predictive accuracy compared to existing drug response prediction models. Furthermore, comprehensive interpretability analyses—including latent embedding visualizations and feature-level SHAP analysis—reveal biologically meaningful insights aligned with known drug targets and mechanisms of action, validating the model's reliability at the molecular level. In summary, OmicsDRP presents a reliable framework for precision oncology by combining robust predictive performance with biological interpretability.
<img width="3284" height="1753" alt="Image" src="https://github.com/user-attachments/assets/54a075e6-8b3c-4354-bea8-7018ef0d4656" />

## Running Step
###### 1. Training 5-fold cross-validation
```text
python Train_5fold_CV.py \
    --batch 128 \
    --num_epochs 100 \
    --embedding_dim 128 \
    --gene_embed_dim 8 \
    --num_heads 2 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --dataset_path ${dataset_path}
```

###### 2. Save results
```text
python Save_Results.py \
    --batch 128 \
    --num_epochs 100 \
    --embedding_dim 128 \
    --gene_embed_dim 8 \
    --num_heads 2 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --dataset_path ${dataset_path}
```


## Dependencies
* `python` = 3.9.19
* `pytorch` = 2.3.0
* `pandas` = 1.5.3
* `numpy` = 1.12.6
* `scipy` = 1.13.1
* `matplotlib` = 3.9.2
