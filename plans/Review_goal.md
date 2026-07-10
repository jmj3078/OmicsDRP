## 난이도 상  재실험/재분석 필수. 실패 시 논문 프레이밍 자체를 수정해야 함

### **리뷰1. Ablation study 부재 (Reviewer #1-9, #4-4)**

두 리뷰어가 독립적으로 요구. 게다가 “어떤 구성요소가 성능에 기여하는지”를 명확하게 서술하기. 

- 현황상 omics ablation은 4-omics 컬럼 subset 문제라 데이터는 준비돼 있음. 하지만 모델은 단일 클래스 단위로 코딩되어있어 대규모 코드 리팩토링이 요구됨.
- 작업:
    1. `HiarachialAttentionModel`에 variant 스위치 도입 (사용할 omics 컬럼, attention on/off, gene selection 방식).
    2. 최소 variant set: single-omics 4종(SNP/meth/CNV/RNA 각각) + full 4-omics + attention 제거 모델.
    3. **동일 seed·동일 fold split·동일 하이퍼파라미터**로 통제. 5-fold CV 재학습.
    4. 결과를 표 + barplot으로 정리 (metric ± std, fold별 값 보존).

### 리뷰2. 외부 검증 데이터셋 부재 (Reviewer #2-1, #2-2)

**왜 상**: 현재 GDSC 내부 5-fold CV만 → “internal validation only” 지적 회피 불가. CV 에 대한 Leakage부분도 더더욱 큰 문제가 되는 부분이 됨.
DepMap 코호트 신규 획득·전처리 + 플랫폼 간 batch effect 처리. 가장 적절한 외부 Test Dataset을 탐색해야함.

- **DepMap을 (a) 학습 확장 또는 (b) 독립 외부 검증셋으로 활용. batch effect 보정 방식 결정 필요.**
- **다른 논문의 external validation 방식을 참조해 평가 프로토콜 맞출 것.**

### 리뷰3. Table 3 벤치마킹 비교 타당성 (Reviewer #1-5, #4-2)

**왜 상**: “논문에서 가져온 값 나열”은 데이터셋 버전·전처리·평가 프로토콜이 달라 직접 비교 불성립.
현재 본문의 한계 문장(“direct benchmarking … was challenging”)으로는 부족하다는 게 두 리뷰어 공통.

- **(a) 직접 재구현 비교**: **DeepCDR/GraphDRP/DeepTTA 중 1~2개를 동일 파이프라인(같은 GDSC2 split·같은 metric)으로 돌려 직접 비교. 이정도 정성은 어느정도 필요할 것으로 보임..**
- **최소 1개라도 그렇게 하는게 좋으며, 논문의 논조 또한 결과에 따라 적절히 수정해줄 필요가 있다.**

### 리뷰4. Validation fold 이중 사용 / data leakage (Reviewer #1-1)

Table 2 전체 수치의 신뢰성을 흔드는 통계적 근본 문제. 단일 리뷰어지만 타당성 핵심.
**현황상 이미 사실로 확인됨** . early stopping과 성능 리포팅이 같은 val fold(위 “현황” 참조). optimistic bias 존재.

- **논조 강등**: 지금 구조를 유지하되, CV 성능을 메인 지표가 아닌 **보조 지표**로 강등해 서술.
- **재실험(권장): nested CV 또는 각 fold 내부에 별도 validation split을 두어 early stopping은 내부 split으로, 성능은 held-out fold로 보고. → Table 2를 정직한 수치로 재산출.**
- **하지만 굳이 CV를 주요 지표로 내새우지 않는다면 재실험까지는 안해도 될지도 모름 (외부 Test데이터셋이 있다면 굳이..?)**

### 리뷰5. Novelty 명확화 (Reviewer #4-1)

**DeepCDR/GraphDRP/PaccMann/DrugCell/DeepTTA/HiDRA와 구조 유사 지적.
“incremental”로 찍히면 major revision 넘어 desk rejection 리스크. 논증 문제지만 최우선.**

- 다양한 omics 데이터 사용의 이점 + omics 통합 효과 등등을 정량적으로 보여야 한다.
- 작업**: (1) self-attention이 실제로 기여하는지 ablation 수치로, (2) Morgan fingerprint를 그래프 표현 대신 쓴 이유를 효율성/해석가능성 트레이드오프로 명문화(리뷰7과 연결), (3) 기존 SOTA와의 과학적 gap 한 문단 정의 (반드시 필요할 것으로 보임, SOTA는 왜 SOTA이고, 우리는 왜 이런가? Discussion에 추가할 것).**

### 리뷰6. SHAP/GO 결과의 정량적 검증 부재  (Reviewer #4-5)

**왜 상**: 현재 SHAP-GO가 “알려진 기전과 일치”라는 정성 확인에 머묾(EGFR_MET, ERBB2_CNV 등).
상관관계를 기전적 증거처럼 제시하는 뉘앙스가 있으며, 통계적인 정량지표로 보이는게 더 좋다.

- 작업: **이미 존재하는 표준 scoring 방식을 그대로 적용**하면 됨(신규 방법 개발 아님).
    - **known drug-target gene recovery rate**
    - **무작위 유전자셋 대비 enrichment 비교 (null distribution)**
    - **fold 간 feature importance 재현성 (상관/순위 일치도)**

### 리뷰7. Morgan fingerprint 선택 정당화  (Reviewer #3-2, #4-7)

**왜 상**: 512-bit Morgan FP 선택 근거 부족. #4는 이를 unseen-drug 성능 저하 원인 후보로 직접 연결.
서술만으로는 부족 → drug representation을 변수로 둔 **ablation 요구**.

- **작업: drug encoder를 바꿔가며(Morgan FP vs ChemBERTa/MolFormer/GROVER 등) 모델 복잡도 변화 + 성능 변화를 두 축으로 비교.**
CV/test metric 정량화. 최소한 fingerprint의 효율성/해석가능성 트레이드오프를 명시 논증하고, 그래프 기반 인코더 비교를 향후 연구로 제안.
- 리뷰1의 variant 프레임워크를 drug branch 쪽으로 확장하면 재사용 가능.

---

## 난이도 중  추가 분석 또는 Discussion 보강. 기존 데이터로 대응

재학습 없이 기존 임베딩·예측·파라미터로 지표만 추가하거나, 논조를 수정하는 항목들.
단 일부는 상 구간(특히 리뷰1) 결과에 논리적으로 의존함.

### 리뷰8. Unseen-drug 일반화 한계 서술 강화  (Reviewer #1-8, #3-1, #4-3, 3명 일치)

- PCC 0.938(standard) → 0.512(unseen-drug) 급락을 논문이 충분히 안 다룸. 논조는 여전히 “robust and generalizable”.
- **실험 아님, 논조 수정: known drug 예측 / unseen drug 예측 결론을 분리. standard-split 성능 기반 일반화 주장(overgeneralization) 제거.**

### 리뷰9. UMAP 클러스터링 주장 정량화  (Reviewer #1-7)

- Fig.3·4의 “MoA별/tissue별 군집” 결론이 시각 판단에만 의존.
- **기존 임베딩에 지표만 추가(scikit-learn): Silhouette, Davies-Bouldin, ARI로 fingerprint space vs embedding space 정량 비교. 상 구간보다 가벼움.**

### 리뷰10. 통계적 유의성 검정  (Reviewer #1-4)

- Table 2의 “best” 표기에 p-value/effect size 없음.
- 5-fold 결과에 paired t-test 또는 Wilcoxon signed-rank test 적용. (fold별 값을 이미 저장하므로 바로 가능)

### 리뷰11. 효율성/단순성 주장의 정량 근거  (Reviewer #1-6)

- “relatively simple architecture”에 파라미터 수·FLOPs·학습/추론 시간 비교 없음.
- **자사 모델 파라미터 수는 즉시 산출. 경쟁 모델은 공개 코드 있는 경우만 직접 비교, 나머지는 문헌값 인용. 중요한것 : 실제로 “효율적이고 단순한지” 검증해봐야함. 그래프 기반의 모델은 상상 이상으로 가볍기 때문에 훨씬 우리 모델이 무거울 수 있다.**

### 리뷰12. 임상적/실무적 함의 논의  (Reviewer #2-7)

- 어떤 omics layer가 실제 필수인지, 종합 프로파일링 투자가 언제 정당화되는지 논의 요구.
- **리뷰1 ablation 결과에 의존  ablation이 끝나야 “어떤 omics가 필수”라는 근거가 생김. 리뷰1 후속.**

### 리뷰13. 약물 재배치(drug repurposing) 적용성 논의  (Reviewer #2-6)

- GDSC 외 신약 적용 가능성. Discussion 확장.
- unseen-drug 성능 저하(리뷰7·8)와 직결 → 함께 서술해야 논리 일관.

### 리뷰14. Gene selection bias 논의 확장  (Reviewer #4-6)

- PharmGKB 909 유전자 선택이 이미 알려진 pharmacogenomic gene으로 편향. 현재 짧게만 언급.
- 이 편향이 **SHAP 해석 전체에 미치는 영향**을 명시 논의. 리뷰6과 묶어서.

### 리뷰15. GitHub 사용 가이드/재현성  (Reviewer #2-5)

- 과학적 엄밀성과 별개, 재현성·사용성. README 보강 + 예제 스크립트 (/test 폴더 제공).

---

## 난이도 하 : 서술 명확화. 실험 불필요

문장 추가/수정으로 끝나는 항목들. 대부분 코드에 이미 있는 값을 논문에 반영하는 수준.

- **Random seed 명시** (#1-2): `set_seed(2024)`, `KFold(random_state=42)` 값 논문 기재.
- **Baseline 프로토콜 동일성 명확화** (#1-3): 동일 프로토콜을 썼다면 문장 한두 개.
- **Baseline DL 모델 선정 기준** (#2-3): 체계적 문헌 검색이었는지 서술 추가.
- **모델 적용 범위(GDSC 약물 한정 여부)** (#2-4): SMILES만 있으면 임의 화합물 적용 가능한지 한 문장.
- **Figure 5B, 5D 가독성** (#1 minor 1,2): 폰트 크기·시각화 개선.
- **관련 연구 추가 인용** (#1 minor 3): 제시된 4개 DOI를 Intro/Discussion에 반영.
- **Future work 구체화** (#3-3): unseen-drug 개선 방향(그래프 표현, transfer learning 등) 구체 서술.

---

## 작업 순서 제안

1. 평가 프로토콜, 외부 평가 데이터셋을 정해야 이후 모든 재학습과 결과 정리가 가능해짐.
2. 리뷰1 ablation 프레임워크 구축
    1.  모델 파라미터화(omics subset / attention on-off / gene selection / drug representation). 리뷰4·7·12의 근거가 여기서 나옴
3. 리뷰6·9·10·11  기존 산출물에 지표 추가
4. 리뷰2·8·12·13·14 및 하 구간  위 결과가 모이면 서술로 마무리.