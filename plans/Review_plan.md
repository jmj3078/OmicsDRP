### 1단계
1. CV Metric 보고 수정 -> Nested CV를 통한 Early stopping 편향 위험 수정
2. Ablation Study를 위한 코드 대폭 수정
   Ablation study를 수행할 부분 : 
   1. Omics data조합, 단 단일 오믹스 사용시 모델 구조 붕괴 발생함. 따라서 Baseline이 필요한데 RNA + Met 혹은 RNA + CNV 조합으로 하는게 좋을 듯 함.
   2. Attention 구조와 단순 MLP 구조의 성능 차이
   3. Drug Representation Methods 간의 차이
    - 후보군 : GIN, GCN : SMILES -> 분자그래프 -> 처음부터 모델 학습
    - ChemBERTa, MolFormer, GROVER, Graphormer : Pretrained Model 활용, Embedding만 넣어주기
3. Unseen Cell line, Unseen Drug에 대한 split 로직 수정
    - Unseen Cell Line, Unseen Drug는 현재 Random 5 fold split으로 수행중 -> Clustering 수행, Stratify하게 Split하여 지나친 편향이 발생하지 않도록 (Cell line간 유사도, Drug 간 유사도에 따라 우연히 "완전히 OOD인" 데이터가 Test에 들어가지 않도록 보다 보수적으로 접근하기)하기
  
### 2단계
1. 비교를 위한 외부 테스트 모델 가져오기, 동일 데이터와 동일 학습 조건 하에서 CV + Unseen Drug + Unseen Cell line 성능 평가 하기
2. 파라미터 수 기반의 모델 간 복잡도 비교, 이 분야에 정확한 "SOTA"는 없으므로, 추후 논의를 통해 어떤 모델을 비교용으로 선정할지 제공할 예정
3. External Test Dataset 기반 평가 수행 -> 5 fold CV 데이터를 Ensemble하는게 가장 쉬운 방법일 듯 함. 본 모델과 비교용 외부 테스트 모델 가져와서 Inference로 결과 정리

### 3단계
1. Negative Control로 모델의 해석파트 보강 + 통계적 검정 기반의 엄밀성 반영
2. UMAP Clustering에 지표 추가하기
3. Visualization 개선하기
4. 이외 기타 등등 작업 (가장 난이도가 낮은 작업)수행