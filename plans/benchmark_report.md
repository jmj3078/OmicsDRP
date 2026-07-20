# Stage-2 외부 DRP 모델 벤치마크 후보 조사 결과 보고서 (Benchmark Report)

본 보고서는 [/project/OmicsDRP_Review/plans/Benchmark_plan.md](file:///project/OmicsDRP_Review/plans/Benchmark_plan.md) 계획에 따라, 23종의 기존 DRP(Drug Response Prediction) 모델 및 2종의 신규 누락 후보(TGSA, TransEDRP) 등 총 25종 모델에 대해 논문 전문 분석(Stage A)과 GitHub 코드 감사(Stage B)를 마친 최종 리포트입니다.

---

## 1. 벤치마크 쇼트리스트 (Shortlist) 추천 및 종합 요약

### 1.1 벤치마크 규칙 및 선정 기준
1. **Native 입력 보존**: 각 외부 모델은 고유의 입력 형식(SMILES, Graph, 1D 오믹스 등)을 그대로 사용합니다. (이전 909-유전자 공용 입력 제한 폐기)
2. **평가 방식 고정**: 우리 데이터의 Nested CV 폴드 스플릿, Unseen Split 로직, regression 평가 metric을 적용하여 엄밀하게 교차 검증합니다.
3. **복잡도 비교**: 성능뿐만 아니라 모델의 파라미터 수 및 학습 속도를 기록하여 비교합니다.

### 1.2 추천 쇼트리스트 (High Fit 모델 7종)
다음 7개 모델은 **(1) 구현 소스 코드가 온전히 존재하고**, **(2) SMILES 및 Bulk 오믹스를 입력으로 취하며**, **(3) 최신 GPU(RTX 4090/CUDA 12) 환경에 포팅이 수월**하여 벤치마크 대상 모델로 최우선 추천합니다.

| 추천 모델 | 출판년도 | 주요 입력 형태 | 코드 상태 | 최신 하드웨어 호환성 | 추천 사유 |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **GraphDRP** | 2022 | SMILES (Graph) + 1D Genomic vector | Full | **Yes** (PyTorch / PyG) | 코드 및 전처리 완성도가 최상이며, GNN 구조 분석에 필수적임. |
| **DeepTTA** | 2022 | SMILES (Subword) + Gene Expression | Full | **Yes** (PyTorch) | Transformer 기반의 직관적인 아키텍처로 범용 데이터셋 연동이 매우 용이함. |
| **PaccMann** | 2019 | SMILES (Text) + Gene Expression | Full | **Yes** (PyTorch / pytoda) | PyTorch 모듈화 및 전처리 패키지(`pytoda`)가 완비되어 커스텀 데이터 이식이 매우 쉬움. |
| **TGSA** | 2022 | SMILES (Graph) + PPI Network Graph | Full | **Yes** (PyTorch / PyG) | GNN 및 단백질 상호작용(PPI) 네트워크 기반의 우수한 baseline 모델. |
| **DeepDRA** | 2024 | Drug descriptors + Multi-omics | Full | **Yes** (PyTorch/TF) | Autoencoder를 활용한 차원 축소와 MLP 기반의 정석적인 multi-omics 통합 모델. |
| **CSG2A** | 2024 | SMILES + PPI Prior + Expression | Full | **Yes** (PyTorch) | 유전자 상호작용 어텐션 및 전이학습(LINCS -> GDSC) 메커니즘을 벤치마크하기 좋음. |
| **DrugCell** | 2020 | Morgan FP + Gene Mutation | Full | **Yes/Hard** (PyTorch / networkx) | Gene Ontology (GO) 계층 구조를 반영한 VNN 구조로, 해석 가능성 비교군으로 가치가 높음. |

---

## 2. DRP 리뷰 논문 6편 배경 요약

1. **Caponigro et al. (Nature Reviews Drug Discovery 2011)**
   - 암 치료 가설의 전임상 테스트 발전 과정을 다룸. 고전적 약물 반응성 평가 방식과 약물 작용 메커니즘 검증을 위한 세포주 활용 가이드라인 제시.
2. **Costello et al. (Nature Biotechnology 2014)**
   - 드림 챌린지(DREAM Challenge)를 통해 약물 감수성 예측 알고리즘 공동 평가. 유전자 발현 데이터가 가장 높은 예측력을 가짐을 확인하고 baseline ML의 표준을 설정.
3. **Adam et al. (npj Precision Oncology 2020)**
   - DRP에 머신러닝을 적용할 때 발생하는 한계(데이터 이질성, 생물학적 해석 불가능성 등)와 최신 발전을 정리.
4. **Baptista et al. (Briefings in Bioinformatics 2021)**
   - 암 약물 반응 예측을 위한 딥러닝 기법의 현황을 정리하며 multi-omics 데이터 late integration 기법 및 CNN/Autoencoder 기반의 주요 모델을 비교.
5. **Partin et al. (Frontiers in Medicine 2023)**
   - DRP 딥러닝 연구의 주요 트렌드 분석. 데이터 전처리 불일치성과 데이터 누수(Data Leakage) 문제를 주요 과제로 지목함.
6. **Zhang et al. (Briefings in Bioinformatics 2024)**
   - 약물 발견 프로세스에서 어텐션(Attention) 및 트랜스포머(Transformer) 구조가 어떻게 활용되고 있는지 요약.

---

## 3. 개별 DRP 모델 25종 감사 상세 보고서

*모든 감사 레포지토리는 로컬 작업 폴더 [benchmark_sources/repos/](file:///project/OmicsDRP_Review/benchmark_sources/repos/)에 `--depth 1`로 클론되어 저장되었습니다.*

---

### [1] Menden et al. (PLoS One 2013)
- **Paper Link**: [https://doi.org/10.1371/journal.pone.0061318](https://doi.org/10.1371/journal.pone.0061318)
- **GitHub URL**: None (공개 코드 없음)
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 별도 딥러닝 인코더 없음. 77개 암유전자의 Mutation 및 CNA 바이너리 인코딩.
  - Drug Encoder: PubChem 1D/2D 화학 구조 디스크립터.
  - Fusion: 두 피처를 Concatenation 후 Random Forest 및 얕은 MLP 연결.
- **Native Inputs**: CCLE/GDSC Mutation + Drug Descriptors -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: No (코드가 존재하지 않음)
- **적합성 평가 요약**: 고전적 ML 기법이며 공식 구현체가 없어 벤치마크 편입 불가.

---

### [2] CDRScan (Scientific Reports 2018)
- **Paper Link**: [https://doi.org/10.1038/s41598-018-27214-6](https://doi.org/10.1038/s41598-018-27214-6)
- **GitHub URL**: None
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: Mutational Fingerprint를 입력으로 하는 1D CNN.
  - Drug Encoder: Molecular Fingerprint를 입력으로 하는 1D CNN.
  - Fusion: 두 CNN의 출력을 Concat하여 FC Layer 통과.
- **Native Inputs**: GDSC Binary mutation + PubChem drug fingerprints -> IC50 viability
- **RTX 4090 / CUDA 12 구동 여부**: No (코드 부재)
- **적합성 평가 요약**: 공식 퍼블릭 코드가 존재하지 않아 재현 불가능.

---

### [3] DeepDR (BMC Medical Genomics 2019)
- **Paper Link**: [https://doi.org/10.1186/s12920-018-0460-9](https://doi.org/10.1186/s12920-018-0460-9)
- **GitHub URL**: None
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 사전 학습된 다중 오믹스 Autoencoder.
  - Drug Encoder: 약물 화학 구조의 MLP 인코딩.
  - Fusion: 두 인코딩을 병합 후 Fully Connected Layer 통과.
- **Native Inputs**: Mutation, Expression + Chemical descriptors -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: No (코드 부재)
- **적합성 평가 요약**: 공식 구현 코드가 없어 활용 불가.

---

### [4] MOLI (Bioinformatics 2019)
- **Paper Link**: [https://doi.org/10.1093/bioinformatics/btz318](https://doi.org/10.1093/bioinformatics/btz318)
- **GitHub URL**: [https://github.com/hosseinshn/MOLI](https://github.com/hosseinshn/MOLI)
- **Local Path**: [benchmark_sources/repos/MOLI](file:///project/OmicsDRP_Review/benchmark_sources/repos/MOLI)
- **Repro Grade**: B
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: Mutation, CNA, Expression 각각에 대해 독립적인 Feed-forward sub-networks 적용.
  - Drug Encoder: **없음** (약물 구조 피처를 학습하지 않고, 각 약물별로 독립된 모델을 개별 학습함).
  - Fusion: late integration concat 후 Triplet Loss + BCE 최적화.
- **Native Inputs**: CCLE Multi-omics (약물 피처 없음) -> Binary target (Responder vs Non-responder)
- **RTX 4090 / CUDA 12 구동 여부**: Yes (구버전 PyTorch 구조이나 최신 환경에서 정상 작동 가능)
- **적합성 평가 요약**: 약물 피처를 모델 내부에서 처리하지 않고 약물별 모델을 따로 만드는 형태라 대규모 Unseen Drug 성능을 측정하는 우리 벤치마크 평가 방식에 부적합함.

---

### [5] tCNN (BMC Bioinformatics 2019)
- **Paper Link**: [https://doi.org/10.1186/s12859-019-2910-6](https://doi.org/10.1186/s12859-019-2910-6)
- **GitHub URL**: [https://github.com/Lowpassfilter/tCNNS-Project](https://github.com/Lowpassfilter/tCNNS-Project)
- **Local Path**: [benchmark_sources/repos/tCNN](file:///project/OmicsDRP_Review/benchmark_sources/repos/tCNN)
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 735개 유전자 변이 바이너리 벡터를 입력으로 하는 1D-CNN.
  - Drug Encoder: SMILES one-hot encoding 기반 1D-CNN.
  - Fusion: Concat -> FC layers (1024x3).
- **Native Inputs**: SMILES + 735 Mutation/CNA -> Normalized ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: No (GitHub 레포지토리에 소스 코드가 없이 비어 있음)
- **적합성 평가 요약**: 공식 저장소에 README 1줄만 존재하고 실제 소스 코드가 전혀 없어 복제 불가.

---

### [6] PaccMann (Molecular Pharmaceutics 2019)
- **Paper Link**: [https://doi.org/10.1021/acs.molpharmaceut.9b00520](https://doi.org/10.1021/acs.molpharmaceut.9b00520)
- **GitHub URL**: [https://github.com/PaccMann/paccmann_predictor](https://github.com/PaccMann/paccmann_predictor)
- **Local Path**: [benchmark_sources/repos/PaccMann](file:///project/OmicsDRP_Review/benchmark_sources/repos/PaccMann)
- **Repro Grade**: A
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: Gene Expression에 대한 Multimodal Attention-based CNN.
  - Drug Encoder: SMILES에 대한 Multiscale CNN + Contextual Attention.
  - Fusion: Inter-modality Attention을 적용한 고도화된 퓨전 기법.
- **Native Inputs**: SMILES string + Cell line transcriptomics -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: Yes (표준 PyTorch 구조)
- **적합성 평가 요약**: 코드의 패키징 완성도 및 데이터 파이프라인(`pytoda`)이 훌륭하여 벤치마크 편입 강추.

---

### [7] PathDNN (JCIM 2020)
- **Paper Link**: [https://doi.org/10.1021/acs.jcim.0c00331](https://doi.org/10.1021/acs.jcim.0c00331)
- **GitHub URL**: [https://github.com/Charrick/drug_sensitivity_pred](https://github.com/Charrick/drug_sensitivity_pred) (제3자 백업)
- **Local Path**: [benchmark_sources/repos/PathDNN](file:///project/OmicsDRP_Review/benchmark_sources/repos/PathDNN)
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: KEGG 패스웨이 지식을 사용하여 Gene-Pathway 가중치를 마스킹한 Linear layer.
  - Drug Encoder: Morgan Fingerprint.
  - Fusion: Concat -> MLP.
- **Native Inputs**: Gene Expression + Drug Features -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: Hard (환경 설정이 파편화되어 있고 원본 패스웨이 파일 유실)
- **적합성 평가 요약**: 제3자 저장소도 핵심 스크립트만 남아 있고, 학습용 raw 데이터 전처리 파이프라인이 완전하지 않아 벤치마크 편입이 곤란함.

---

### [8] DeepCDR (Bioinformatics 2020)
- **Paper Link**: [https://doi.org/10.1093/bioinformatics/btaa278](https://doi.org/10.1093/bioinformatics/btaa278)
- **GitHub URL**: [https://github.com/kimmo1019/DeepCDR](https://github.com/kimmo1019/DeepCDR)
- **Local Path**: [benchmark_sources/repos/DeepCDR](file:///project/OmicsDRP_Review/benchmark_sources/repos/DeepCDR)
- **Repro Grade**: B
- **Benchmark Fit**: Med
- **Architecture**:
  - Cell Encoder: 다중 오믹스(Mutation, Expr, Methyl)에 대한 1D CNN.
  - Drug Encoder: Graph Convolutional Network (GCN).
  - Fusion: Concat -> FC layers.
- **Native Inputs**: Multi-omics (697 genes expression, 34673 mutation, 808 methylation) + SMILES graph -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: Hard (Keras 2.1.4, TensorFlow 1.13.1 의존성으로 인해 CUDA 12/RTX 4090 단독 구동 불가. Docker 및 TF2 포팅 작업 필수)
- **적합성 평가 요약**: 하이브리드 GCN 구조는 훌륭하나 프레임워크 한계(TF1)로 포팅 노력이 상당수 필요함.

---

### [9] DrugCell (Cancer Cell 2020)
- **Paper Link**: [https://doi.org/10.1016/j.ccell.2020.11.001](https://doi.org/10.1016/j.ccell.2020.11.001)
- **GitHub URL**: [https://github.com/idekerlab/DrugCell](https://github.com/idekerlab/DrugCell)
- **Local Path**: [benchmark_sources/repos/DrugCell](file:///project/OmicsDRP_Review/benchmark_sources/repos/DrugCell)
- **Repro Grade**: B
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: Gene Ontology 계층 구조를 맵핑한 Visible Neural Network (VNN).
  - Drug Encoder: Fully Connected Layer (MLP).
  - Fusion: Concat -> MLP.
- **Native Inputs**: 3008개 유전자 Mutation (Binary) + 2048-dim Morgan FP -> AUC
- **RTX 4090 / CUDA 12 구동 여부**: Yes/Hard (PyTorch 기반이나 레거시 구버전 API가 섞여 있어 최신 PyTorch로 포팅 필요)
- **적합성 평가 요약**: 해석 가능성을 강조하는 독특한 VNN 구조로, GO 계층 구조 정렬만 수행하면 벤치마크 가치가 큼.

---

### [10] DeepDSC (IEEE/ACM TCBB 2021)
- **Paper Link**: [https://doi.org/10.1109/TCBB.2019.2961895](https://doi.org/10.1109/TCBB.2019.2961895)
- **GitHub URL**: None
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: Stacked Deep Autoencoder를 이용한 유전자 발현 정보 차원축소.
  - Drug Encoder: Morgan Fingerprints.
  - Fusion: Concat -> MLP.
- **Native Inputs**: Gene Expression + Morgan Fingerprint -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: No (공식 코드 공개 부재)
- **적합성 평가 요약**: 공식 배포 코드가 전혀 존재하지 않아 벤치마크 제외.

---

### [11] HiDRA (JCIM 2021)
- **Paper Link**: [https://doi.org/10.1021/acs.jcim.1c00706](https://doi.org/10.1021/acs.jcim.1c00706)
- **GitHub URL**: None (Supplementary SI zip으로만 제공)
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 유전자 및 KEGG 패스웨이 기반의 Hierarchical Attention.
  - Drug Encoder: Morgan Fingerprint (512-dim).
  - Fusion: Attention에 Drug Embedding을 concat하여 쿼리로 쓰는 Fusion -> MLP.
- **Native Inputs**: Gene Expression + Morgan Fingerprint -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: No (관리되는 형상관리 저장소 없음)
- **적합성 평가 요약**: 저널 첨부파일로만 코드를 제공하여 접근성 및 유지보수성이 떨어짐.

---

### [12] DeepDRK (Briefings in Bioinformatics 2021)
- **Paper Link**: [https://doi.org/10.1093/bib/bbab048](https://doi.org/10.1093/bib/bbab048)
- **GitHub URL**: [https://github.com/wangyc82/DeepDRK](https://github.com/wangyc82/DeepDRK)
- **Local Path**: None (R 기반 프로젝트)
- **Repro Grade**: C
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell/Drug: 커널(Kernel) 데이터 통합 기법을 사용하여 다중 오믹스와 화합물 DTI 특징 통합.
- **Native Inputs**: Mutation, CNV, Epigenomics, Chemical features, DTI -> Response
- **RTX 4090 / CUDA 12 구동 여부**: No (R 및 H2O Java 프레임워크 기반이므로 Python/PyTorch 파이프라인에 이식 불가능)
- **적합성 평가 요약**: 모델의 개발 생태계가 R/Java로 완전히 분리되어 있어 통합 파이프라인 구축에 부적합함.

---

### [13] SWnet (BMC Bioinformatics 2021)
- **Paper Link**: [https://doi.org/10.1186/s12859-021-04352-9](https://doi.org/10.1186/s12859-021-04352-9)
- **GitHub URL**: [https://github.com/zuozhaorui/SWnet](https://github.com/zuozhaorui/SWnet)
- **Local Path**: [benchmark_sources/repos/SWnet](file:///project/OmicsDRP_Review/benchmark_sources/repos/SWnet)
- **Repro Grade**: B
- **Benchmark Fit**: Med
- **Architecture**:
  - Cell Encoder: Multi-omics (Expression, Mutation) MLP -> Self-attention.
  - Drug Encoder: Graph Convolutional Network (GCN).
  - Fusion: Self-attention 기반 퓨전 후 FC Layer 통과.
- **Native Inputs**: SMILES Graph + CCLE Multi-omics -> AUC / ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch 및 PyG 기반)
- **적합성 평가 요약**: 아키텍처는 유효하나 입력 오믹스 가공 코드가 파편화되어 있어 연동 시 약간의 공수가 요구됨.

---

### [14] GraphDRP (IEEE/ACM TCBB 2022)
- **Paper Link**: [https://doi.org/10.1109/TCBB.2021.3090422](https://doi.org/10.1109/TCBB.2021.3090422)
- **GitHub URL**: [https://github.com/hauldhut/GraphDRP](https://github.com/hauldhut/GraphDRP)
- **Local Path**: [benchmark_sources/repos/GraphDRP](file:///project/OmicsDRP_Review/benchmark_sources/repos/GraphDRP)
- **Repro Grade**: A
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: Genomic aberration 바이너리 벡터에 대한 1D CNN.
  - Drug Encoder: GIN, GAT, GCN 등 다양한 분자 그래프 GNN 지원.
  - Fusion: Concat -> FC layers.
- **Native Inputs**: SMILES graph + Cell line aberration vector -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: Yes (최신 PyTorch 및 PyG 지원)
- **적합성 평가 요약**: 코드 구조가 모듈형으로 깔끔하고, 전처리 파이프라인이 정교하게 관리되어 있어 벤치마크 모델 1순위 후보.

---

### [15] DRPreter (IJMS 2022)
- **Paper Link**: [https://doi.org/10.3390/ijms232415516](https://doi.org/10.3390/ijms232415516)
- **GitHub URL**: [https://github.com/babaling/DRPreter](https://github.com/babaling/DRPreter)
- **Local Path**: [benchmark_sources/repos/DRPreter](file:///project/OmicsDRP_Review/benchmark_sources/repos/DRPreter)
- **Repro Grade**: B
- **Benchmark Fit**: Med
- **Architecture**:
  - Cell Encoder: PPI 패스웨이 서브그래프 지식을 통합한 GNN.
  - Drug Encoder: 분자 그래프 GNN.
  - Fusion: Type-aware Transformer 융합.
- **Native Inputs**: SMILES graph + Biological Pathway network mapping -> AUC / IC50
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch 기반)
- **적합성 평가 요약**: 세포주 피처를 고정된 외부 PPI 네트워크 그래프 구조로 변환해야 하는 전처리 제약이 있어 확장성이 중간 수준임.

---

### [16] Precily (Nature Communications 2022)
- **Paper Link**: [https://doi.org/10.1038/s41467-022-33291-z](https://doi.org/10.1038/s41467-022-33291-z)
- **GitHub URL**: [https://github.com/SmritiChawla/Precily](https://github.com/SmritiChawla/Precily)
- **Local Path**: [benchmark_sources/repos/Precily](file:///project/OmicsDRP_Review/benchmark_sources/repos/Precily)
- **Repro Grade**: C
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: R 기반 GSVA 패스웨이 풍부도 점수 처리 MLP.
  - Drug Encoder: SMILESVec (Word2Vec 임베딩).
  - Fusion: Concat -> DNN.
- **Native Inputs**: GSVA Scores + SMILESVec -> Sensitivity
- **RTX 4090 / CUDA 12 구동 여부**: Hard (R과 Python 환경의 혼재 및 구형 Keras 의존성)
- **적합성 평가 요약**: R을 통한 GSVA 전처리 자동화 및 구버전 의존성으로 인해 파이프라인 통합 비용이 큼.

---

### [17] DeepTTA (Briefings in Bioinformatics 2022)
- **Paper Link**: [https://doi.org/10.1093/bib/bbac100](https://doi.org/10.1093/bib/bbac100)
- **GitHub URL**: [https://github.com/jianglikun/DeepTTC](https://github.com/jianglikun/DeepTTC) (DeepTTC와 동일)
- **Local Path**: [benchmark_sources/repos/DeepTTA](file:///project/OmicsDRP_Review/benchmark_sources/repos/DeepTTA)
- **Repro Grade**: A
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: Gene Expression에 대한 Transformer Encoder.
  - Drug Encoder: SMILES sub-word 토큰에 대한 Transformer Encoder.
  - Fusion: Concat -> MLP.
- **Native Inputs**: SMILES subword sequence + Gene expression vector -> ln(IC50)
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch 기반, 호환성 최상)
- **적합성 평가 요약**: 트랜스포머 기반의 입력 표현 및 전처리 코드가 완비되어 있어 벤치마크 편입 적극 권장.

---

### [18] GPDRP (BMC Bioinformatics 2023)
- **Paper Link**: [https://doi.org/10.1186/s12859-023-05618-0](https://doi.org/10.1186/s12859-023-05618-0)
- **GitHub URL**: [https://github.com/yyk124/GPDRP](https://github.com/yyk124/GPDRP)
- **Local Path**: [benchmark_sources/repos/GPDRP](file:///project/OmicsDRP_Review/benchmark_sources/repos/GPDRP)
- **Repro Grade**: B
- **Benchmark Fit**: Med
- **Architecture**:
  - Cell Encoder: RNA-seq 발현량 기반 pathway activity score MLP.
  - Drug Encoder: Graph GNN + Graph Transformer.
  - Fusion: Concat -> MLP.
- **Native Inputs**: SMILES graph + Pathway activity matrix -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch)
- **적합성 평가 요약**: 약물 그래프 트랜스포머는 훌륭하나 세포주 pathway activity 계산 제약이 존재함.

---

### [19] DeepCoVDR (Bioinformatics 2023)
- **Paper Link**: [https://doi.org/10.1093/bioinformatics/btad469](https://doi.org/10.1093/bioinformatics/btad469)
- **GitHub URL**: [https://github.com/Hhhzj-7/DeepCoVDR](https://github.com/Hhhzj-7/DeepCoVDR)
- **Local Path**: None (학습용 text/npy 링크 수동 관리 필요)
- **Repro Grade**: C
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 3-layer DNN.
  - Drug Encoder: Chemprop DMPNN + Graph Transformer.
  - Fusion: Cross-attention -> MLP.
- **Native Inputs**: SMILES graph + 11,794 Gene Expression -> COVID-19 drug response
- **RTX 4090 / CUDA 12 구동 여부**: Hard (구버전 Python 3.8 및 Chemprop 환경 고정)
- **적합성 평가 요약**: COVID-19 약물반응에 전이학습(Fine-tuning)을 적용하는 특수 모델로, 일반 DRP 벤치마크용으로는 설계가 다소 한정적임.

---

### [20] DeepDRA (PLoS One 2024)
- **Paper Link**: [https://doi.org/10.1371/journal.pone.0307649](https://doi.org/10.1371/journal.pone.0307649)
- **GitHub URL**: [https://github.com/bcb-sut/DeepDRA](https://github.com/bcb-sut/DeepDRA)
- **Local Path**: [benchmark_sources/repos/DeepDRA](file:///project/OmicsDRP_Review/benchmark_sources/repos/DeepDRA)
- **Repro Grade**: B
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: 다중 오믹스 특징을 축소하기 위한 modality-specific Autoencoder.
  - Drug Encoder: 약물 descriptor/fingerprint Autoencoder.
  - Fusion: 압축된 특징 Concat -> MLP Supervised Predictor.
- **Native Inputs**: Multi-omics + Drug descriptors -> ln(IC50) / Response
- **RTX 4090 / CUDA 12 구동 여부**: Yes (일반적인 PyTorch Autoencoder 구조)
- **적합성 평가 요약**: 차원 축소 Autoencoder와 MLP 구조를 사용하는 대형 다중 오믹스 DRP 벤치마크에 매우 적합.

---

### [21] CSG2A (Bioinformatics 2024)
- **Paper Link**: [https://doi.org/10.1093/bioinformatics/btae342](https://doi.org/10.1093/bioinformatics/btae342)
- **GitHub URL**: [https://github.com/eugenebang/CSG2A](https://github.com/eugenebang/CSG2A)
- **Local Path**: [benchmark_sources/repos/CSG2A](file:///project/OmicsDRP_Review/benchmark_sources/repos/CSG2A)
- **Repro Grade**: B
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: PPI 지식 기반 Gene-Gene Attention (CSG2A) 및 RNA 발현 인코더.
  - Drug Encoder: SMILES, Dosage, Time을 입력으로 하는 Chemical Condition Encoder.
  - Fusion: Perturbation 어텐션 기반 융합 (LINCS -> GDSC 전이학습).
- **Native Inputs**: LINCS L1000 Gene expression + SMILES, Dosage -> Cell viability / IC50
- **RTX 4090 / CUDA 12 구동 여부**: Yes (최신 PyTorch 친화적 어텐션 메커니즘)
- **적합성 평가 요약**: 전이학습 메커니즘을 벤치마크하기 좋고 코드 완성도가 우수하여 편입 강추.

---

### [22] scDrug+ (Biomedicine & Pharmacotherapy 2024)
- **Paper Link**: [https://doi.org/10.1016/j.biopha.2024.117070](https://doi.org/10.1016/j.biopha.2024.117070)
- **GitHub URL**: [https://github.com/ailabstw/scDrugplus](https://github.com/ailabstw/scDrugplus)
- **Local Path**: None (Docker 의존형 파이프라인)
- **Repro Grade**: B
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: single-cell RNA-seq 임베딩 추출기.
  - Drug Encoder: 분자 구조 Graph 인코더.
  - Fusion: Multi-modal joint projection.
- **Native Inputs**: scRNA-seq Expression matrix + SMILES -> Response/Viability
- **RTX 4090 / CUDA 12 구동 여부**: Yes (Docker 환경 구동)
- **적합성 평가 요약**: 세포 단위 감수성을 평가하기 위해 single-cell RNA-seq 데이터를 입력을 요구하므로, bulk 오믹스 기반의 GDSC/DepMap 벤치마크에는 입력 구조 불일치로 부적합함.

---

### [23] DTLCDR (Journal of Pharmaceutical Analysis 2025)
- **Paper Link**: [https://doi.org/10.1016/j.jpha.2025.001327](https://doi.org/10.1016/j.jpha.2025.001327)
- **GitHub URL**: [https://github.com/yujie0317/DTLCDR](https://github.com/yujie0317/DTLCDR)
- **Local Path**: None (DTI 네트워크 매핑 필요)
- **Repro Grade**: C
- **Benchmark Fit**: Med
- **Architecture**:
  - Cell Encoder: HuggingFace LLM 기반의 전사체 언어 표현 임베딩.
  - Drug Encoder: Drug-Target Interaction (DTI) 그래프 네트워크 인코더.
  - Fusion: target-based multimodal fusion.
- **Native Inputs**: LLM transcriptomics embedding + Drug-Target interaction map -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch 기반 트랜스포머 생태계)
- **적합성 평가 요약**: DTI(약물-타겟 정보) 네트워크가 사전 구성되어 있어야 약물 특징 생성이 가능하여 커스텀 신약(SMILES만 있는 경우) 평가 시 한계가 있음.

---

### [24] TGSA (Bioinformatics 2022) — 신규 추가
- **Paper Link**: [https://doi.org/10.1093/bioinformatics/btab562](https://doi.org/10.1093/bioinformatics/btab562)
- **GitHub URL**: [https://github.com/violet-sto/TGSA](https://github.com/violet-sto/TGSA)
- **Local Path**: [benchmark_sources/repos/TGSA](file:///project/OmicsDRP_Review/benchmark_sources/repos/TGSA)
- **Repro Grade**: A
- **Benchmark Fit**: High
- **Architecture**:
  - Cell Encoder: STRING PPI 네트워크 기반의 GNN.
  - Drug Encoder: 분자 그래프 기반 GNN.
  - Fusion: Twin Graph Neural Network (TGDRP) + Similarity Augmentation (SA) 퓨전.
- **Native Inputs**: SMILES Graph + STRING PPI mapped Gene expression -> GDSC2 IC50
- **RTX 4090 / CUDA 12 구동 여부**: Yes (PyTorch & PyG 기반)
- **적합성 평가 요약**: 유사도 기반의 SA 모듈과 GNN을 유기적으로 연동한 훌륭한 모델로, 벤치마크 최고 수준의 적합도 보유.

---

### [25] TransEDRP (Briefings in Bioinformatics 2022) — 신규 추가
- **Paper Link**: [https://doi.org/10.1093/bib/bbac467](https://doi.org/10.1093/bib/bbac467)
- **GitHub URL**: None
- **Local Path**: None
- **Repro Grade**: D
- **Benchmark Fit**: Low
- **Architecture**:
  - Cell Encoder: 전사체 시퀀스에 대한 Multi-head attention.
  - Drug Encoder: Chirality와 Aromaticity를 반영한 Edge-embedding Graph Transformer.
  - Fusion: Dual-branch feature Transformer fusion.
- **Native Inputs**: SMILES Graph + Transcriptomics sequence -> IC50
- **RTX 4090 / CUDA 12 구동 여부**: No (공식 코드 공개 부재)
- **적합성 평가 요약**: 공식 퍼블릭 소스 코드가 유실되었거나 공개되지 않아 벤치마크 편입이 불가능함.
