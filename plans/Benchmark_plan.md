# Stage-2 벤치마크 후보 조사 (논문 파싱 + GitHub 코드 감사) — 계획

## Context

리뷰3(경쟁모델 직접 재구현 비교)·리뷰11(파라미터/효율성)을 위해 외부 DRP 모델을 벤치마킹한다.
이전 하네스(`omicsdrp/benchmark/` 어댑터 3종, `benchmark_feasibility.md`)는 커밋
**6626ff1 "discard wrong benchmarking frameworks"** 로 **전부 폐기**됐고, `plans/candidates.tsv`는
원본 리스트(리뷰 6편 + 모델 23종)로 되돌아왔다. 즉 벤치마크를 **처음부터 다시** 시작하며,
지금 단계는 그 첫 관문인 **후보 재조사**다. 사용자는 tsv에 "오류가 꽤 있다"고 판단, 그리고
이전 web-수준 감사가 과대평가(README를 신뢰 → 실제 코드와 불일치) 사례를 냈으므로,
**실제 논문 전문 + 실제 repo 코드**에 근거한 엄밀한 재조사가 필요하다.

**최종 벤치마크 규칙(사용자 확정, 이 조사의 판단 기준):**
1. 각 외부 모델은 **자신의 Native 입력 데이터를 그대로** 사용한다 (우리 909-유전자 텐서로 강제 X — 이전 909-공용입력 설계는 폐기됨).
2. 고정: **데이터 split·학습 방식**(Nested CV, unseen_cell, unseen_drug, Ensemble용 CV 재학습) + **평가 metric**.
3. metric에 **모델 파라미터 수** 추가 기록.
→ 따라서 조사의 "적합도(Fit)" 축 = *"이 모델이 자기 Native 피처로 돌면서, 우리 (cell,drug,IC50) pair·우리 fold·우리 metric으로 평가될 수 있는가"*. 핵심 하위질문: **Native cell/drug 피처가 무엇이고, 우리 cell/drug 유니버스에 대해 재생성 가능한가(외부 다운로드 필요 여부)**.

**결정사항(사용자):** 모델 23종 전수 재검증 · **git clone 후 실제 코드 읽기** · candidates.tsv를 **1행=1모델로 깔끔히 재구성**.

## 산출물

1. `plans/candidates.tsv` — **1행=1모델**, 검증·교정된 필드 + 신규 감사 열 (모델 23종). 리뷰 6편은 리포트로 이관.
2. `plans/benchmark_report.md` — 모델별 상세 근거(mechanism/inputs/코드감사) + **모든 주장에 근거 URL/파일경로** + 리뷰논문 배경 요약 + native-입력 벤치마크용 shortlist 추천.
3. 메모리 정리(아래).

## 조사 프로토콜 (모델당 2단계)

### Stage A — 논문 파싱 (원리 + Native 입력)  ← 주 skill: `paper-lookup`
- `paper-lookup` skill로 **실제 논문**을 가져온다: PMC 전문(생의학) / arXiv·bioRxiv / Semantic Scholar(초록+TLDR) / Crossref·OpenAlex(메타) / Unpaywall(OA PDF). IEEE·publisher 페이왈 우회에 유리.
- 보조로 candidates.tsv의 publisher 링크를 `WebFetch`.
- 추출: (1) **Mechanism** = cell encoder / drug encoder / fusion / output head, (2) **Native Cell_Input** = omics 종류(mut/CNV/expr/methyl) + 대략 #features + 출처 데이터셋(GDSC/CCLE/CTRP/TCGA/STRING 등), (3) **Drug_Input** = SMILES→graph / Morgan FP / one-hot / ESPF / descriptor, (4) **Task_Label** = ln(IC50) 회귀 / normalized / AUC / 분류, (5) **Train_Dataset** + 버전(GDSC1/2/v6…).

### Stage B — GitHub 코드 감사 (실제 clone + read)  ← 도구: `WebSearch`+`WebFetch`, `git clone`
- 공식 repo 찾기: **Stage A에서 얻은 논문 data/code-availability 문구를 최우선** 사용, 없으면 `WebSearch "<model> drug response github"`(저자/공식 우선, 비공식·fork·없음 명시). `WebFetch`로 repo 존재 확인.
- **`git clone --depth 1` 로 스크래치패드 개인 하위폴더에 clone** → 실제 파일을 읽는다 (**학습·환경빌드 실행은 하지 않음**, 오직 읽기). 대형/실패 시 `WebFetch`로 핵심 파일 대체.
- 확인 항목(각각 **Full/Partial/None** 등급):
  - **Code_Preprocessing**: raw→모델입력 파생 코드가 실재하는가 (단순히 사전가공 파일만 있는지 vs raw에서 유도하는지).
  - **Code_Training**: 처음부터 학습 스크립트 실재?
  - **Code_Architecture**: 모델 정의(.py) 실재?
  - **Env**: framework+버전(TF1/TF2/PyTorch/PyG), python 버전, **RTX4090/CUDA12에서 구동 가능?(yes/hard/no)**.
  - **Data_Bundled**: 처리/원시 데이터가 repo 내 번들 / 다운로드 / 게이트 / 없음.
  - **License**.

### 종합 (모델당)
- **Native_Feasibility**: 우리 cell/drug 유니버스에 대해 Native 피처를 재생성 가능한가 + **외부 다운로드 필요 플래그**.
- **Repro_Grade**: A=오늘 바로 재학습 / B=노력필요 / C=추론만 / D=사용가능 코드 없음.
- **Benchmark_Fit**: High/Med/Low + 한 줄 이유(규칙#1 native-입력 기준).
- **엄밀성 규칙(필수)**: 모든 비자명 주장에 **근거 URL 또는 clone한 파일경로**를 병기. 검증 못 한 것은 단정하지 말고 "Uncertain"으로 표시.

## 서브에이전트 오케스트레이션

- **에이전트 유형**: `general-purpose` (Skill+WebSearch+WebFetch+Bash/git 모두 필요). Explore는 로컬 검색 전용이라 부적합.
- **배치**: **8개 에이전트 × 약 3모델** (23종), 병렬 백그라운드. 배치안:
  - B1 Menden/CDRScan/DeepDR · B2 MOLI/tCNN(tCNNS)/PaccMann · B3 PathDNN/DeepCDR/DrugCell · B4 DeepDSC/HiDRA/DeepDRK · B5 SWnet/GraphDRP/DRPreter · B6 Precily/DeepTTA/GPDRP · B7 DeepCoVDR/DeepDRA/CSG2A · B8 scDrug+/DTLCDR (+강력 누락후보 TGSA/TransEDRP 경량 플래그).
- **쓰기 충돌 방지**: 에이전트는 candidates.tsv를 **쓰지 않는다**. 표준화된 마크다운 블록으로 결과만 반환 → 메인스레드가 취합·기록.
- **clone 위생**: 각 에이전트는 자기 스크래치패드 하위폴더에 `--depth 1` clone, 읽기 전용, 완료 후 정리 가능. 학습/설치 실행 금지.
- **리뷰논문 6편**: 별도 경량 패스 1개(또는 B8에 부가) — 벤치마킹 대상 아님, DRP 평가 관행·대표 baseline 파악용 배경 요약만.

## 메인스레드 취합

- 8개 보고서 수집 → **정합성 QA 패스**(에이전트 간 불일치·중복·논문수치 sanity 교차확인).
- `candidates.tsv`를 **깔끔한 1행=1모델 TSV**로 재작성: 헤더 + 23개 모델행, 교정+신규 열. 멀티라인 셀 제거.
- `benchmark_report.md` 작성: 모델별 근거 URL/파일경로 + 리뷰 배경 + shortlist 추천.

## 메모리 정리

- **신규**: `omicsdrp-benchmark-investigation.md` — 이 조사 프로토콜 + native-입력 벤치마크 규칙 + candidates.tsv/report 포인터.
- **제거(폐기된 작업 서술 → stale)**: `omicsdrp-benchmark-candidates.md`, `omicsdrp-benchmark-harness.md` (둘 다 discard된 909-공용입력 하네스·모순된 reversal 이력). MEMORY.md 인덱스에서 두 줄 삭제.
- **유지**: `omicsdrp-inference-ensemble.md`(별개의 유효 트랙), 나머지 stage-1 메모리.

## Critical files

- 수정: `plans/candidates.tsv`(재작성), 신규 `plans/benchmark_report.md`.
- 메모리: `~/.claude/projects/-project-OmicsDRP-Review/memory/` 하위 신규 1 + 삭제 2 + `MEMORY.md` 인덱스.
- 조사 자체는 외부 웹/repo 대상이라 `omicsdrp/` 코드 변경 없음 (다음 단계인 하네스 구축에서 재사용 예정: `data.load_raw`, `splits.build_folds`, `metrics.regression_metrics`).

## Verification

- **근거 감사성**: 모든 모델행이 GitHub_URL(또는 명시적 NONE) + Repro_Grade + Benchmark_Fit + 근거 URL/파일경로를 가진다.
- **메인스레드 스팟체크**: 임의 2~3개 모델에 대해 핵심 주장 1개를 독립 재확인(공식 repo URL 해석 여부, requirements의 framework 버전).
- **TSV 정합성**: `pandas.read_csv(sep='\t')`로 파싱 → 23행, 모든 행 컬럼수 일치, 멀티라인 셀 없음.
- **범위 정직성(중요)**: 이 라운드는 *논문 전문 + 실제 repo 코드 읽기*까지다. **clone-and-RUN(원논문 수치 재현·환경 실제 구동)은 포함하지 않는다** — 그건 shortlist 확정 후 다음 단계(이전에 수행됐다 폐기된 실행/재현 단계). 따라서 Repro_Grade는 "선언된 의존성+실제 코드 근거의 예측"이며, 실측이 아님을 report에 명시.

## Next (이 계획 밖, 조사 완료 후)

교정된 candidates.tsv/shortlist를 바탕으로 → native-입력 벤치마크 하네스 재설계(각 모델 native 피처 소비, 우리 fold·metric·param 고정) → clone-and-run 재현 → 3-regime 재학습. 별도 계획으로 진행.
