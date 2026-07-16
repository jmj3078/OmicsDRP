# Stage-2 벤치마크 재현성 심층감사 — 공정 재학습 실현가능성

**작성일:** 2026-07-16
**목적:** 리뷰3(직접 재구현 비교)·리뷰11(효율성/파라미터 비교)을 위해, 선정 후보 4개 모델을
**우리 GDSC2 데이터 · 동일 fold(`build_folds`) · 동일 metric** 조건에서 재학습해 공정 비교가 가능한지
각 모델의 실제 코드를 clone·정독하여 판정. 병렬 서브에이전트 4개로 코드레벨 감사 수행.

**우리 데이터 좌표:** 멀티오믹스(SNP/MET/CNV/RNA, **PharmGKB 909유전자로 제한**) cell branch +
Morgan FP drug branch + **ln(IC50)** 회귀. GDSC2, 873 cell × 231 drug(241→중복 SMILES 10쌍 병합).
라벨 = `IC50_GDSC2.csv`(이미 자연로그). 오믹스 raw = `raw_data/{SNP,MET,CNV,RNA}_PGKB.csv`.

---

## 종합 판정 요약

| 항목 | **GraphDRP** | **DeepCDR** | **DeepTTA** | **DRPreter** |
|---|---|---|---|---|
| **판정** | ✅ FEASIBLE | ✅ FEASIBLE | ✅ FEASIBLE | ✅ FEASIBLE (단, 공정성 캐비엇) |
| **난이도** | Medium | Medium | **Low–Medium (최저)** | Medium |
| **연도/방향성** | 2022 / 다름(경량 graph) | 2020 / 유사(멀티오믹스) | 2022 / 다름(transformer drug) | 2022 / 중간(pathway+graph) |
| **Cell branch** | width-agnostic 1D-CNN | dim은 CSV 너비서 추론(패널 하드코딩 없음) | MLP, `input_dim_gene` 상수 1개 | KEGG 34-pathway GAT 서브그래프 |
| **909유전자 공정투입** | ✅ 깔끔 (`fc1_xt` Linear 1줄 수정) | ✅ 깔끔 (RNA→gexpr, MET→methy, SNP→mut; CNV 브랜치 없음→드롭) | ✅ 깔끔 (`17737→909` 1줄) | ⚠️ **374/909만 KEGG 교집합, pathway 2개 단일유전자 붕괴** |
| **Drug 입력** | SMILES→graph, 하드코딩 없음 | SMILES→graph(DeepChem 75d) | ESPF/BPE 토큰, **231개 OOV 0** 실측 | SMILES→graph(77d), 하드코딩 없음 |
| **라벨 처리** | ⚠️ `sigmoid(0.1·x)` [0,1] → **역변환 `10·logit(y)`** (검증됨) | ✅ ln(IC50) 직접 | ✅ ln(IC50) 직접 | ✅ ln(IC50) 직접 |
| **fold 주입** | preprocessing 분리 → `.pt` 재생성으로 주입 | in-code split → 교체 용이 | runtime split → DataFrame 주입 | `utils.load_data` ~20줄 교체 (3-split 모두 지원) |
| **early-stopping 누수** | ✅ val로 정상 | ⚠️ **test=val 누수 → 수정 필요** | ⚠️ **test=val 누수 → 수정 필요** | ✅ val 분리 정상 |
| **환경** | PyG 1.x → **2.x 포팅 2줄 권장** | TF1.13/py2 → **RTX4090 미지원(CPU 또는 nvidia-tf), 격리 env** | **기존 `omicsdrp` env 재사용(pandas 2줄 수정)** | PyG1.7/cu111 **4090 불가 → PyG2.x 포팅 필수** |
| **metric 외부채점** | ✅ per-pair 예측 반환 | ✅ (예측 덤프 3줄 추가) | ✅ per-pair 반환 | ✅ per-pair df 덤프 |
| **omics 폭** | mut+CNV(우리 것 대체) | 3-omics(SNP/RNA/MET); CNV 추가시 소규모 수술 | **RNA 단일** | **RNA 단일(default)** |
| **라이선스** | Apache-2.0 | MIT | none | MIT |

**핵심 결론: 4개 모두 하드 블로커 없이 공정 재학습 가능.** 어느 모델도 gene panel이 아키텍처에
하드와이어돼 있지 않아(모두 런타임에 차원 추론) 우리 909유전자를 정당하게 투입 가능. 유일한
과학적 캐비엇은 DRPreter의 KEGG-pathway 의존성(909→374).

---

## 반복되는 공통 쟁점 (모든 모델 어댑터 공통 설계)

1. **909유전자 공정투입** — GraphDRP/DeepCDR/DeepTTA는 cell branch가 차원 유연 → 우리 909유전자를
   그대로 먹이는 것이 *가장 공정한* 투입(각 모델이 동일한 909-유전자 뷰를 받음). DRPreter만
   KEGG-pathway 주석이 필요해 374/909로 축소 → 별도 서술 필요.
2. **early-stopping 누수 정정** — DeepCDR·DeepTTA는 원 코드가 **test fold를 early-stopping val로
   사용**(optimistic bias). 우리 nested-CV 불변식과 충돌 → outer-train 내부에 inner-val을 별도로
   carve해서 교정해야 함(우리 Stage-1과 동일 원칙).
3. **라벨 스케일** — GraphDRP만 [0,1] 정규화 → `ln_IC50 = 10·logit(y)`로 역변환 후 채점.
   나머지 3개는 native ln(IC50)이라 변환 불필요.
4. **fold 주입** — 4개 모두 split이 코드/전처리 단계에 있고 파일에 baked되지 않아(또는 재생성
   가능) `build_folds`의 mixed/unseen_cell/unseen_drug 인덱스를 직접 주입 가능.
5. **외부 채점** — 4개 모두 per-pair 예측을 반환/덤프 → 각 모델의 내장 metric 무시하고 우리
   RMSE/PCC/R² 파이프라인으로 통일 채점.
6. **격리 환경** — DeepTTA는 기존 `omicsdrp` env 재사용(예외적). GraphDRP/DRPreter는 PyG 2.x로
   소폭 포팅(API 표면 작음). DeepCDR만 TF1.x라 별도 격리 env + RTX4090 GPU 미지원(CPU 학습 또는
   nvidia-tensorflow fork). **`omicsdrp`/`pytorch_env` 오염 금지 — `drugemb_*` 패턴대로 격리.**

---

## 모델별 상세

### GraphDRP — ✅ FEASIBLE / Medium
- **Cell:** `preprocess.py:186-222`가 binary [n_cell×735](310 mutation + 425 CNV)를 동적 생성, gene
  리스트 하드코딩 없음. 모델은 length-735를 1D conv 3블록 → `fc1_xt=Linear(2944,128)`. **우리
  909유전자 벡터로 교체 시 conv 산술로 입력크기(예 909→`Linear(3712,128)`) 한 줄만 수정.**
  우리 SNP/CNV는 정수(0–4)라 binary화하거나 연속값 투입(conv은 float 허용).
- **Drug:** `smile_to_graph` (78d atom feat), 하드코딩 없음. 231 SMILES 직접. `atom_features`가 degree>10/
  희귀원소서 예외 → try/except 권장.
- **라벨:** `1/(1+exp(x)^-0.1)=sigmoid(0.1·x)`, Sigmoid 출력. **역변환 `10·logit(y)` 수치 왕복검증 완료.**
- **fold:** split은 preprocessing에서 셔플/슬라이스, training은 split-agnostic. `TestbedDataset` 직접
  호출로 fold별 `.pt` 재생성(~30줄 어댑터). 3-split 모두 매핑.
- **환경:** PyG 1.x(`utils.py:5` `DataLoader` import). torch1.13/cu117+PyG wheel 핀 또는 **PyG 2.x로
  import 2줄 포팅(권장)** — conv 레이어 불변.
- **어댑터 수술:** ① fold별 데이터빌더(SMILES, 우리 omics 벡터, `sigmoid(0.1·ln)` 라벨) → `.pt` 방출
  ② `fc1_xt` 입력크기 수정 ③ PyG 2.x env ④ `10·logit(y)` 역변환 후 우리 metric.

### DeepCDR — ✅ FEASIBLE / Medium
- **Cell:** `run_DeepCDR.py:177-179`가 omics 차원을 **CSV 너비에서 런타임 추론**, `model.py:23-29`가
  그 dim으로 Input 생성 — **gene 패널 하드코딩 전무.** 우리 909유전자 정당 투입:
  RNA→gexpr(Dense), MET→methy(Dense) 깔끔. SNP→mutation은 34k sparse binary용 Conv2D라 909
  정수입력엔 off-design이나 차원 붕괴 없이 구동(민감도용 Dense 변형 선택 가능). **CNV native 브랜치
  없음** → (a) 드롭(3-omics 충실) 또는 (b) `model.py`에 Dense 브랜치 1개 추가(소규모).
- **Drug:** DeepChem `ConvMolFeaturizer` 75d, `.hkl` 그래프. 231 SMILES 직접(원자>100 truncate 주의).
  DeepChem 버전별 75d 순서 상이 → 버전 핀 + 스팟체크(무음 정확성 리스크).
- **라벨:** ln(IC50) 직접 회귀, MSE, 선형 출력. **스케일 없음, 우리와 정확 일치.**
- **fold:** `DataSplit`(:128-136) 파이썬 리스트 분할 → `build_folds`로 교체 최소수술. **단 기본코드는
  test를 early-stopping val로 사용(`MyCallback`) → inner-val 추가로 누수 차단 필요.**
- **환경(최대 마찰):** Keras2.1.4/TF1.13.1/py2.7. **TF1.13은 CUDA10 → RTX4090(sm_89) 미지원 →
  CPU 학습(모델 작아 실현가능) 또는 nvidia-tensorflow(TF1.15 fork, CUDA11)**. py2 소스 2군데 print만
  2to3 포팅. 격리 `deepcdr` conda env.
- **어댑터 수술:** ① 격리 env + 2to3 ② CCLE 로더 → 우리 `raw_data/*_PGKB.csv`+`IC50_GDSC2.csv`
  ③ 231 SMILES `.hkl` 재생성 ④ `DataSplit`→`build_folds` ⑤ inner-val 누수정정 ⑥ CNV 처리 결정
  ⑦ per-pair 예측 CSV 덤프.

### DeepTTA — ✅ FEASIBLE / **Low–Medium (가장 친화적)**
- **Cell:** expression branch가 `MLP`이고 폭이 **상수 하나(`Step3_model.py:94 input_dim_gene=17737`)**.
  원 RNA 파일(17.7k)은 repo에 미포함(하드코딩 절대경로). **`17737→909` 한 줄 + 우리 `RNA_PGKB.csv`
  투입 → 가장 공정한 동일-909뷰. 909유전자 공정성 이슈 non-issue.**
- **Drug:** ESPF/BPE(ChEMBL 2586-토큰 사전, drug-agnostic). **231 SMILES 실측 토크나이즈 → OOV 0개.**
  3개 약물만 50토큰 초과 truncate(모델 설계한계, 모든 GDSC run 동일 → 공정).
- **라벨:** `LN_IC50` 직접, MSE, 스케일 없음 → 우리와 일치.
- **fold:** runtime split(`GetData`), 파일 baked 아님. **단 `Step3_model.py:331`이 `val=testdata`로
  test 누수** → inner-val 추가로 교정. DataFrame(`DRUG_ID,COSMIC_ID,LN_IC50`) 주입.
- **환경(예외적 장점):** README는 py3.5/torch1.4나 API가 표준적이라 **기존 `omicsdrp` env(py3.9/torch2.3)
  로 그대로 포팅 가능. pandas `append→concat` 2줄 수정만.** 레거시 CUDA 고고학 불필요.
- **어댑터 수술:** ① `input_dim_gene=909` ② `GetData` 우회 + inner-val ③ `getRna`→우리 RNA(스케일러
  inner-train만 fit) ④ ESPF 인코더 그대로 ⑤ pandas 수정+`head(10000)` cap 제거 ⑥ epoch 상향(기본 3
  너무 짧음)+inner-val early stop ⑦ 외부 채점.

### DRPreter — ✅ FEASIBLE / Medium (**+ 공정성 캐비엇**)
- **Cell(핵심 쟁점):** `cellline_graph.py`가 cell당 disjoint 그래프 — CCLE 2369유전자 = **34 KEGG
  pathway의 union**, STRING PPI(≥0.99) intra-pathway edge. 모델은 dim을 런타임 추론(2369 하드코딩
  아님)이라 재구성 가능. **그러나 우리 909유전자 중 374개만 KEGG 패널과 교집합** → 나머지 535개는
  pathway 미할당으로 무음 드롭. 34 pathway 전부 ≥1유전자 유지돼 크래시는 없으나 **pathway 2개가
  단일 유전자로, ~5개가 ≤4유전자로 붕괴** → pathway-token Transformer inductive bias 심각 약화.
  선택: (a) 우리 909(실질 374) on 붕괴된 KEGG 그래프, 또는 (b) DRPreter native 2369 CCLE expression
  (유전자 패널·발현 소스 모두 다른 데이터 = 동일입력 아님). **fair run이면 (a), 참고용 상한으로 (b) 병기 권장.**
- **omics 폭:** DRPreter default는 **RNA 단일**(`num_feature=1`) → like-for-like는 우리 RNA만 투입
  (우리 모델의 멀티오믹스 이점은 DRPreter가 소비 불가 → 서술로 구분).
- **Drug:** `smiles2graph`(77d, dgllife), 하드코딩 없음. 231 SMILES 직접(약물명 키).
- **라벨:** native ln(IC50) MSE, 스케일 없음 → 완전 일치.
- **fold:** `utils.load_data`의 `train_test_split` 2곳을 `build_folds` 인덱스로 교체(~20줄). 순수
  index 기반이라 **mixed/unseen_cell/unseen_drug 동일 지원**. outer-train서 val carve 필요.
- **환경:** torch1.9+cu111/PyG1.7.1/dgl0.6.1 → **cu111은 RTX4090 미지원(하드블로커).** API 표면 작아
  **torch2.x+cu121+PyG2.x 포팅 필수**(GATConv/GINConv/JK/global pool 모두 2.x 존재; pandas append +
  dgllife 핀 소폭 수정). `omicsdrp`와 동일 현대 스택.
- **어댑터 수술:** ① 우리 RNA로 cell_dict(pathway pkl+PPI를 374유전자로 슬라이스) ② 231 SMILES 그래프
  재생성 ③ 우리 IC50 df(일관 키, `--sim False`) ④ `load_data`→`build_folds`+val carve, 5-fold×3-split
  ⑤ PyG2.x 포팅 ⑥ 외부 채점.

---

## 재구현 추천 (실현가능성 근거)

| 우선순위 | 모델 | 근거 |
|---|---|---|
| **1 (필수)** | **DeepTTA** | 최저 마찰(기존 env, 라벨 일치, 909 non-issue, OOV 0), 리뷰3 지명. 빠른 첫 성과. |
| **2 (필수)** | **GraphDRP** | 리뷰3 지명 + **리뷰11 효율성 핵심 경량 graph 비교군**("graph 모델이 훨씬 가볍다" 검증). 어댑터 친화. |
| **3 (권장)** | **DeepCDR** | 우리와 가장 유사한 **멀티오믹스** apples-to-apples(오믹스 통합 효과 대조), 리뷰3 지명. 단 TF1.x 환경 마찰(CPU/nvidia-tf) 감수. |
| **4 (선택)** | **DRPreter** | GDSC2 native ln(IC50)로 데이터정합 最상이나, 909→374 pathway 붕괴 캐비엇을 명시 서술해야 공정. 여유 시 추가. |

- **최소 조합(리뷰3 "최소 1개" 충족):** DeepTTA + GraphDRP.
- **권장 조합:** DeepTTA + GraphDRP + DeepCDR (리뷰3 지명 3종 완성 + 리뷰11 경량비교 + 멀티오믹스 대조).
- **DRPreter는 4번째로**, 공정성 캐비엇 서술을 감수할 수 있으면 데이터정합성 근거로 추가.

## 공통 어댑터 하네스 (다음 설계 대상)
모든 경쟁모델이 공유할 얇은 하네스:
1. 우리 GDSC2 (cell,drug,ln_IC50) pair universe + `build_folds` 인덱스(mixed/unseen_cell/unseen_drug) 주입.
2. 모델별 native cell/drug featurization 유지하되 **우리 909유전자 omics + 231 SMILES** 소비.
3. 스케일러는 **inner-train cells만 fit**(우리 leakage boundary 준수).
4. early-stopping은 **inner-val**로(test 누수 모델은 정정), 성능은 held-out outer-test 단일 평가.
5. 라벨 정규화 모델([0,1])은 예측을 ln(IC50)로 역변환.
6. per-pair 예측 덤프 → 통일 RMSE/PCC/R² + 파라미터 수/추론시간(리뷰11) 채점.

**clone 위치:** `scratchpad/benchmark_repos/{GraphDRP,DeepCDR,DeepTTC,DRPreter}` (감사용, 어댑터는 별도 구성 예정).
