# OmicsDRP Stage-1 sweep progress

`[############################] 21/21`  elapsed 5h36m

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=morgan · split=unseen_drug *(baseline)* | cached | 2.6530±0.2203 | 0.1024 | 0.4226 | 5 | - |
| 2 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP · cell=attention · drug=morgan · split=unseen_drug | done | 2.6066±0.2636 | 0.1328 | 0.4342 | 5 | 19m41s |
| 3 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET · cell=attention · drug=morgan · split=unseen_drug | done | 2.6250±0.2511 | 0.1215 | 0.4342 | 5 | 19m19s |
| 4 | 1_feature | omics=SNP+MET+CNV+RNA · noise=CNV · cell=attention · drug=morgan · split=unseen_drug | done | 2.6726±0.2166 | 0.0897 | 0.4093 | 5 | 21m04s |
| 5 | 1_feature | omics=SNP+MET+CNV+RNA · noise=RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6175±0.2447 | 0.1261 | 0.4327 | 5 | 21m24s |
| 6 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET · cell=attention · drug=morgan · split=unseen_drug | done | 2.6682±0.2145 | 0.0924 | 0.4111 | 5 | 16m00s |
| 7 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+CNV · cell=attention · drug=morgan · split=unseen_drug | done | 2.6488±0.2104 | 0.1065 | 0.4314 | 5 | 25m47s |
| 8 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6617±0.2117 | 0.0965 | 0.4217 | 5 | 22m51s |
| 9 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+CNV · cell=attention · drug=morgan · split=unseen_drug | done | 2.6605±0.2462 | 0.0976 | 0.4293 | 5 | 19m38s |
| 10 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6570±0.1956 | 0.1013 | 0.4356 | 5 | 26m32s |
| 11 | 1_feature | omics=SNP+MET+CNV+RNA · noise=CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.7048±0.2104 | 0.0666 | 0.3993 | 5 | 28m00s |
| 12 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+CNV · cell=attention · drug=morgan · split=unseen_drug | done | 2.6447±0.2355 | 0.1082 | 0.4359 | 5 | 25m27s |
| 13 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6878±0.2189 | 0.0794 | 0.4224 | 5 | 22m11s |
| 14 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6055±0.2320 | 0.1342 | 0.4478 | 5 | 22m09s |
| 15 | 1_feature | omics=SNP+MET+CNV+RNA · noise=MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6684±0.2097 | 0.0919 | 0.4254 | 5 | 21m48s |
| 16 | 1_feature | omics=SNP+MET+CNV+RNA · noise=SNP+MET+CNV+RNA · cell=attention · drug=morgan · split=unseen_drug | done | 2.6281±0.2357 | 0.1199 | 0.4322 | 5 | 24m18s |
| 17 | 2_attention | omics=SNP+MET+CNV+RNA · noise=none · cell=mlp · drug=morgan · split=unseen_drug | cached | 2.6157±0.2211 | 0.1275 | 0.4449 | 5 | - |
| 18 | 3_drug | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=chemberta · split=unseen_drug | cached | 2.5355±0.1261 | 0.1797 | 0.5051 | 5 | - |
| 19 | 3_drug | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=molformer · split=unseen_drug | cached | 2.5363±0.1764 | 0.1799 | 0.4954 | 5 | - |
| 20 | 3_drug | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=graphormer · split=unseen_drug | cached | 2.6965±0.1422 | 0.0693 | 0.4326 | 5 | - |
| 21 | 3_drug | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=unimol · split=unseen_drug | cached | 2.5723±0.2244 | 0.1594 | 0.4473 | 5 | - |
