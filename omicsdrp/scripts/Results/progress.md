# OmicsDRP Stage-1 sweep progress

`[############################] 16/16`  elapsed 8h36m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=molformer · split=unseen_cell *(baseline)* | cached | 1.3090±0.0269 | 0.7856 | 0.8868 | 5 | - |
| 2 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP · cell=attention · drug=molformer · split=unseen_cell | done | 1.3223±0.0305 | 0.7812 | 0.8851 | 5 | 34m06s |
| 3 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET · cell=attention · drug=molformer · split=unseen_cell | done | 1.3293±0.0263 | 0.7789 | 0.8834 | 5 | 39m26s |
| 4 | 1_feature | omics=SNP+MET+CNV+RNA · noise=CNV · cell=attention · drug=molformer · split=unseen_cell | done | 1.3163±0.0287 | 0.7832 | 0.8854 | 5 | 45m10s |
| 5 | 1_feature | omics=SNP+MET+CNV+RNA · noise=RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.3762±0.0134 | 0.7630 | 0.8748 | 5 | 28m03s |
| 6 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET · cell=attention · drug=molformer · split=unseen_cell | done | 1.3236±0.0227 | 0.7808 | 0.8841 | 5 | 39m24s |
| 7 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+CNV · cell=attention · drug=molformer · split=unseen_cell | done | 1.3214±0.0190 | 0.7815 | 0.8848 | 5 | 39m03s |
| 8 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.3564±0.0281 | 0.7698 | 0.8781 | 5 | 35m49s |
| 9 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+CNV · cell=attention · drug=molformer · split=unseen_cell | done | 1.3285±0.0221 | 0.7792 | 0.8840 | 5 | 44m46s |
| 10 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.4756±0.0442 | 0.7275 | 0.8533 | 5 | 22m33s |
| 11 | 1_feature | omics=SNP+MET+CNV+RNA · noise=CNV+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.3630±0.0333 | 0.7675 | 0.8767 | 5 | 42m59s |
| 12 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+CNV · cell=attention · drug=molformer · split=unseen_cell | done | 1.3260±0.0237 | 0.7800 | 0.8840 | 5 | 44m25s |
| 13 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.4718±0.0408 | 0.7289 | 0.8542 | 5 | 25m05s |
| 14 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+CNV+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.3684±0.0355 | 0.7657 | 0.8761 | 5 | 35m29s |
| 15 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+CNV+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.5527±0.0423 | 0.6983 | 0.8379 | 5 | 21m51s |
| 16 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+CNV+RNA · cell=attention · drug=molformer · split=unseen_cell | done | 1.5620±0.0485 | 0.6947 | 0.8358 | 5 | 18m16s |
