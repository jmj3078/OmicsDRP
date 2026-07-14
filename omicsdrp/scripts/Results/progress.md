# OmicsDRP Stage-1 sweep progress

`[############################] 12/12`  elapsed 4h55m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug *(baseline)* | done | 2.6530±0.2203 | 0.1024 | 0.4226 | 5 | 25m17s |
| 2 | 1_feature | omics=SNP+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6489±0.2624 | 0.1040 | 0.4258 | 5 | 29m13s |
| 3 | 1_feature | omics=MET+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6478±0.3045 | 0.1044 | 0.4302 | 5 | 26m00s |
| 4 | 1_feature | omics=CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6913±0.2106 | 0.0775 | 0.4183 | 5 | 24m32s |
| 5 | 1_feature | omics=SNP+MET+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6605±0.2622 | 0.0975 | 0.4245 | 5 | 22m04s |
| 6 | 1_feature | omics=SNP+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6734±0.1994 | 0.0888 | 0.4247 | 5 | 29m32s |
| 7 | 1_feature | omics=MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6802±0.2465 | 0.0825 | 0.4174 | 5 | 26m21s |
| 8 | 2_attention | omics=SNP+MET+CNV+RNA · cell=mlp · drug=morgan · split=unseen_drug | done | 2.6157±0.2211 | 0.1275 | 0.4449 | 5 | 5m11s |
| 9 | 3_drug | omics=SNP+MET+CNV+RNA · cell=attention · drug=chemberta · split=unseen_drug | done | 2.5355±0.1261 | 0.1797 | 0.5051 | 5 | 32m07s |
| 10 | 3_drug | omics=SNP+MET+CNV+RNA · cell=attention · drug=molformer · split=unseen_drug | done | 2.5363±0.1764 | 0.1799 | 0.4954 | 5 | 19m54s |
| 11 | 3_drug | omics=SNP+MET+CNV+RNA · cell=attention · drug=graphormer · split=unseen_drug | done | 2.6965±0.1422 | 0.0693 | 0.4326 | 5 | 29m38s |
| 12 | 3_drug | omics=SNP+MET+CNV+RNA · cell=attention · drug=unimol · split=unseen_drug | done | 2.5723±0.2244 | 0.1594 | 0.4473 | 5 | 25m42s |
