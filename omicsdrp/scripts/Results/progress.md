# OmicsDRP Stage-1 sweep progress

`[#####################-------] 9/12`  elapsed 5h30m, ETA ~1h49m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_cell *(baseline)* | done | 1.3086±0.0280 | 0.7857 | 0.8871 | 5 | 41m03s |
| 2 | 1_feature | omics=SNP+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3359±0.0095 | 0.7766 | 0.8814 | 5 | 36m46s |
| 3 | 1_feature | omics=MET+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3179±0.0088 | 0.7826 | 0.8855 | 5 | 38m21s |
| 4 | 1_feature | omics=CNV+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3369±0.0277 | 0.7761 | 0.8821 | 5 | 38m35s |
| 5 | 1_feature | omics=SNP+MET+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3233±0.0190 | 0.7808 | 0.8842 | 5 | 44m45s |
| 6 | 1_feature | omics=SNP+CNV+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3260±0.0083 | 0.7799 | 0.8835 | 5 | 35m42s |
| 7 | 1_feature | omics=MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3265±0.0193 | 0.7798 | 0.8837 | 5 | 44m21s |
| 8 | 2_attention | omics=SNP+MET+CNV+RNA · cell=mlp · drug=morgan · split=unseen_cell | done | 1.3080±0.0315 | 0.7859 | 0.8871 | 5 | 11m41s |
| 9 | 3_drug | omics=SNP+MET+CNV+RNA · cell=attention · drug=chemberta · split=unseen_cell | done | 1.3177±0.0232 | 0.7828 | 0.8858 | 5 | 38m36s |
