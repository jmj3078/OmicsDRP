# OmicsDRP Stage-1 sweep progress

`[##############--------------] 6/12`  elapsed 11h11m, ETA ~11h11m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · cell=attention · drug=morgan · split=mixed *(baseline)* | done | 0.9652±0.0211 | 0.8834 | 0.9401 | 5 | 2h06m |
| 2 | 1_feature | omics=SNP+RNA · cell=attention · drug=morgan · split=mixed | done | 0.9709±0.0090 | 0.8820 | 0.9393 | 5 | 1h48m |
| 3 | 1_feature | omics=MET+RNA · cell=attention · drug=morgan · split=mixed | done | 0.9700±0.0109 | 0.8822 | 0.9395 | 5 | 1h43m |
| 4 | 1_feature | omics=CNV+RNA · cell=attention · drug=morgan · split=mixed | done | 0.9662±0.0104 | 0.8832 | 0.9399 | 5 | 2h05m |
| 5 | 1_feature | omics=SNP+MET+RNA · cell=attention · drug=morgan · split=mixed | done | 0.9701±0.0043 | 0.8823 | 0.9394 | 5 | 1h46m |
| 6 | 1_feature | omics=SNP+CNV+RNA · cell=attention · drug=morgan · split=mixed | done | 0.9739±0.0128 | 0.8813 | 0.9390 | 5 | 1h40m |
