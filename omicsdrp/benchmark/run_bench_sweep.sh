#!/usr/bin/env bash
# Stable, resumable, detached driver for the Stage-2 benchmark sweep.
#
# Mirrors scripts/run_sweep.sh: run_benchmark.py is idempotent at fold
# granularity (finished folds are skipped), so this wrapper only needs to run
# detached, auto-restart on failure (resuming where it stopped), and hold a
# single-instance lock. run_benchmark.py emails a summary at the end.
#
# GPU SAFETY: this trains for real. Launch ONLY when the GPU is free (the Stage-1
# sweep saturates it). Pin a spare GPU with GPU=1 if one is idle.
#
# Usage:
#   EMAIL_TO=jmj3078@gmail.com ./run_bench_sweep.sh
#   GPU=1 ./run_bench_sweep.sh --models graphdrp tgsa
#   DEVICE=cpu ./run_bench_sweep.sh              # (very slow; functional only)
# Monitor: tail -f BenchmarkResults/sweep_logs/sweep_*.log  OR  cat BenchmarkResults/progress.md
# Stop:    kill "$(cat BenchmarkResults/sweep.pid)"   (progress kept; rerun to resume)
set -uo pipefail

cd "$(dirname "$0")"
ENV_NAME="${ENV_NAME:-omicsdrp}"
OUT_ROOT="${OUT_ROOT:-./BenchmarkResults}"
DEVICE="${DEVICE:-cuda}"
MAX_RETRIES="${MAX_RETRIES:-20}"
LOCK="$OUT_ROOT/sweep.lock"
PIDFILE="$OUT_ROOT/sweep.pid"
LOGDIR="$OUT_ROOT/sweep_logs"

mkdir -p "$OUT_ROOT" "$LOGDIR"
export ENV_NAME OUT_ROOT DEVICE MAX_RETRIES LOCK PIDFILE LOGDIR EMAIL_TO GPU

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "A benchmark sweep is already running (lock: $LOCK)."
    echo "If it crashed, remove the lock:  rmdir '$LOCK'"
    exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/sweep_$STAMP.log"
export STAMP

run() {
    # shellcheck disable=SC1091
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
    [ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch

    echo "=== benchmark sweep start $STAMP (env=$ENV_NAME, device=$DEVICE, gpu=${GPU:-all}) ==="
    local attempt=1
    while true; do
        echo "--- attempt $attempt/$MAX_RETRIES @ $(date) ---"
        python run_benchmark.py --out_root "$OUT_ROOT" --device "$DEVICE" \
            ${EMAIL_TO:+--email-to "$EMAIL_TO"} "$@"
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "=== benchmark sweep COMPLETE @ $(date) ==="
            break
        fi
        echo "!!! run_benchmark exited rc=$rc @ $(date); resuming in 15s ..."
        attempt=$((attempt + 1))
        if [ $attempt -gt "$MAX_RETRIES" ]; then
            echo "=== gave up after $MAX_RETRIES attempts; rerun to continue ==="
            break
        fi
        sleep 15
    done
}

setsid bash -c "$(declare -f run cleanup); trap cleanup EXIT; run $*" \
    >"$LOG" 2>&1 < /dev/null &
CHILD=$!
echo "$CHILD" > "$PIDFILE"
trap - EXIT
echo "Benchmark sweep launched detached. PID $CHILD"
echo "  log:  tail -f $LOG"
echo "  prog: cat $OUT_ROOT/progress.md"
echo "  stop: kill $CHILD   (progress kept; rerun to resume)"
