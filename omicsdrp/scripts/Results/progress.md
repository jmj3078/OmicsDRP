# OmicsDRP Stage-1 sweep progress

`[###########-----------------] 5/12`  elapsed 3h19m, ETA ~4h39m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_cell *(baseline)* | done | 1.3086±0.0280 | 0.7857 | 0.8871 | 5 | 41m03s |
| 2 | 1_feature | omics=SNP+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3359±0.0095 | 0.7766 | 0.8814 | 5 | 36m46s |
| 3 | 1_feature | omics=MET+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3179±0.0088 | 0.7826 | 0.8855 | 5 | 38m21s |
| 4 | 1_feature | omics=CNV+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3369±0.0277 | 0.7761 | 0.8821 | 5 | 38m35s |
| 5 | 1_feature | omics=SNP+MET+RNA · cell=attention · drug=morgan · split=unseen_cell | done | 1.3233±0.0190 | 0.7808 | 0.8842 | 5 | 44m45s |
