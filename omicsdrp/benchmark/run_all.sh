#!/usr/bin/env bash
# Stage-2 benchmark orchestrator: train every competitor on OUR GDSC2 data with
# identical folds/metric, then score. Runs the three evaluation regimes:
#   1) Nested-CV metric   : --mode nested --split_mode mixed
#   2) OOD generalization : --mode nested --split_mode unseen_cell|unseen_drug
#   3) External ensemble  : --mode ensemble (weights saved for inference)
#
# GPU SAFETY: this launches full GPU training. Run ONLY when the GPU is free
# (the Stage-1 sweep uses it at ~full utilisation). Check `nvidia-smi` first.
# For a zero-GPU functional check use each adapter's --smoke (forces CPU) instead.
#
#   conda activate omicsdrp
#   cd omicsdrp/benchmark
#   DEVICE=cuda ./run_all.sh                 # full runs (GPU must be free)
#   MODELS="deeptta graphdrp" ./run_all.sh   # subset
set -euo pipefail

cd "$(dirname "$0")"
export DGLBACKEND=pytorch

DEVICE="${DEVICE:-cuda}"
EXPORT_DIR="${EXPORT_DIR:-./export}"
OUT="${OUT:-./BenchmarkResults}"
DATASET="${DATASET:-../../data}"
MODELS="${MODELS:-deeptta graphdrp tgsa}"
SPLITS="${SPLITS:-mixed unseen_cell unseen_drug}"
DO_ENSEMBLE="${DO_ENSEMBLE:-1}"

# 1) freeze the shared export (all 5 folds x 3 split modes) once
if [ ! -f "${EXPORT_DIR}/meta.json" ]; then
  echo "[run_all] building export -> ${EXPORT_DIR}"
  python export_data.py --dataset_path "${DATASET}" --out "${EXPORT_DIR}"
fi

run_model () {  # $1=model dir under adapters/
  local m="$1"
  for split in ${SPLITS}; do
    echo "[run_all] ${m} nested ${split}"
    python "adapters/${m}/run.py" --export "${EXPORT_DIR}" --split_mode "${split}" \
        --mode nested --device "${DEVICE}" --out "${OUT}/$(model_tag "${m}")"
  done
  if [ "${DO_ENSEMBLE}" = "1" ]; then
    echo "[run_all] ${m} ensemble (mixed pool)"
    python "adapters/${m}/run.py" --export "${EXPORT_DIR}" --split_mode mixed \
        --mode ensemble --device "${DEVICE}" --out "${OUT}/$(model_tag "${m}")"
  fi
}

model_tag () {  # adapter dir -> output folder name (matches score.py grouping)
  case "$1" in
    deeptta) echo "DeepTTA" ;;
    graphdrp) echo "GraphDRP" ;;
    tgsa) echo "TGSA" ;;
    *) echo "$1" ;;
  esac
}

for m in ${MODELS}; do run_model "${m}"; done

echo "[run_all] scoring"
python score.py --results "${OUT}" --out "${OUT}/benchmark_summary.csv"
echo "[run_all] done -> ${OUT}/benchmark_summary.csv"
