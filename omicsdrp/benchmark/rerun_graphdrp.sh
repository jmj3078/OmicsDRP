#!/usr/bin/env bash
# One-off re-run of GraphDRP after fixing its label normalisation.
#
# The first sweep trained GraphDRP on raw ln(IC50), but the network ends in
# nn.Sigmoid() (models/ginconv.py:111) so it can only emit (0,1): the output
# saturated and the model died at epoch 2 (flat loss, constant predictions,
# PCC ~0). The adapter now applies GraphDRP's own target transform
# (preprocess.py:243) and inverts predictions back to ln(IC50) for scoring.
#
# There is one GPU, so this waits for any running sweep to finish before
# starting. It calls the adapter directly rather than run_benchmark.py so it
# cannot race that sweep's progress.md.
set -u
cd "$(dirname "$0")"

while pgrep -f "run_benchmark.py" > /dev/null; do
    echo "[queue] waiting for the running sweep to finish … $(date '+%F %T')"
    sleep 300
done
echo "[queue] GPU free, starting GraphDRP re-run at $(date '+%F %T')"

for job in "nested mixed" "nested unseen_cell" "nested unseen_drug" "ensemble mixed"; do
    set -- $job
    echo "=== graphdrp $1/$2 ==="
    conda run --no-capture-output -n benchmark_graphdrp \
        python adapters/graphdrp_adapter.py \
        --split "$2" --regime "$1" --device cuda --out-root ./BenchmarkResults
done

conda run -n omicsdrp python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "../src")
from omicsdrp.notify import send_email

lines = []
for d in sorted(Path("BenchmarkResults/graphdrp").glob("*/")):
    ms = [json.loads(p.read_text()) for p in sorted(d.glob("fold_*/metrics.json"))]
    if ms:
        def avg(k):
            return sum(m[k] for m in ms) / len(ms)
        lines.append(f"{d.name:22s} n={len(ms)} rmse={avg('rmse'):.4f} "
                     f"pcc={avg('pearson'):.4f} scc={avg('spearman'):.4f} "
                     f"r2={avg('r2'):.4f}")
send_email("[benchmark] graphdrp rerun complete (label normalisation fixed)",
           "\n".join(lines) or "no folds found", "jmj3078@gmail.com")
PY
