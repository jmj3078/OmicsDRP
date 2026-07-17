#!/usr/bin/env python
"""Resumable benchmark orchestrator: run every (model, split, mode) condition.

Designed to be driven by ``run_bench_sweep.sh`` (detached, auto-restarting), so
it must be **idempotent and crash-safe**: each adapter skips folds whose outputs
already exist, and a failed condition is logged and skipped rather than aborting
the run. Re-running picks up exactly where it stopped.

Writes ``<out_root>/progress.md`` after every condition and (optionally) emails a
summary on completion via omicsdrp's msmtp notifier (best-effort, never blocks).

    conda activate omicsdrp && cd omicsdrp/benchmark
    python run_benchmark.py --device cuda --email-to jmj3078@gmail.com
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from omicsdrp.notify import send_email          # reuse the Stage-1 notifier
except Exception:                                    # notifier optional
    def send_email(*_a, **_k):
        return False

MODEL_TAG = {"deeptta": "DeepTTA", "graphdrp": "GraphDRP", "tgsa": "TGSA"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--export", default="./export")
    p.add_argument("--out_root", default="./BenchmarkResults")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--models", nargs="+", default=["deeptta", "graphdrp", "tgsa"],
                   choices=list(MODEL_TAG))
    p.add_argument("--splits", nargs="+", default=["mixed", "unseen_cell", "unseen_drug"],
                   choices=["mixed", "unseen_cell", "unseen_drug"])
    p.add_argument("--no-ensemble", dest="ensemble", action="store_false",
                   help="skip the external-inference ensemble models")
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU end-to-end check (uses export_smoke, passes --smoke)")
    return p.parse_args()


def _conditions(models, splits, ensemble):
    """Ordered list of (model, split_mode, mode) to run."""
    conds = [(m, s, "nested") for m in models for s in splits]
    if ensemble:
        conds += [(m, "mixed", "ensemble") for m in models]
    return conds


def _write_progress(path, rows, started):
    lines = ["# Benchmark sweep progress",
             f"_started {started} · updated {dt.datetime.now():%Y-%m-%d %H:%M:%S}_", "",
             "| model | split | mode | status | seconds |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['split']} | {r['mode']} | "
                     f"{r['status']} | {r.get('sec','')} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    a = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.smoke:   # tiny data; device stays as --device (orchestrator default cuda)
        a.export = a.export if os.path.exists(os.path.join(a.export, "meta.json")) else "./export_smoke"
    out_root = os.path.abspath(a.out_root)
    os.makedirs(out_root, exist_ok=True)
    started = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"

    # 1) freeze the shared export once (resumable: skipped if present)
    if not os.path.exists(os.path.join(a.export, "meta.json")):
        print(f"[bench] building export -> {a.export}")
        cmd = [sys.executable, os.path.join(here, "export_data.py"),
               "--dataset_path", a.dataset_path, "--out", a.export]
        if a.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    env = dict(os.environ, DGLBACKEND="pytorch", PYTHONUNBUFFERED="1")
    rows, n_fail = [], 0
    for (m, split, mode) in _conditions(a.models, a.splits, a.ensemble):
        out = os.path.join(out_root, MODEL_TAG[m])
        cmd = [sys.executable, os.path.join(here, "adapters", m, "run.py"),
               "--export", a.export, "--split_mode", split, "--mode", mode,
               "--device", a.device, "--out", out]
        if a.overwrite:
            cmd.append("--overwrite")
        if a.smoke:
            cmd.append("--smoke")
        row = {"model": MODEL_TAG[m], "split": split, "mode": mode, "status": "running"}
        rows.append(row); _write_progress(os.path.join(out_root, "progress.md"), rows, started)
        print(f"[bench] {MODEL_TAG[m]} {split} {mode} ...")
        t0 = time.time()
        rc = subprocess.run(cmd, env=env).returncode
        row["sec"] = int(time.time() - t0)
        row["status"] = "done" if rc == 0 else f"FAILED(rc={rc})"
        if rc != 0:
            n_fail += 1
        _write_progress(os.path.join(out_root, "progress.md"), rows, started)

    # 2) score whatever predictions exist
    try:
        subprocess.run([sys.executable, os.path.join(here, "score.py"),
                        "--results", out_root,
                        "--out", os.path.join(out_root, "benchmark_summary.csv")], env=env)
    except Exception as e:
        print(f"[bench] scoring failed: {e}")

    # 3) best-effort email summary
    n_ok = sum(1 for r in rows if r["status"] == "done")
    subject = f"[OmicsDRP benchmark] {'DONE' if n_fail == 0 else f'{n_fail} FAILED'} — {n_ok}/{len(rows)} ok"
    body = f"started {started}\n\n" + "\n".join(
        f"{r['model']:>9} {r['split']:>11} {r['mode']:>8}  {r['status']}  ({r.get('sec','?')}s)"
        for r in rows) + f"\n\nsummary: {os.path.join(out_root, 'benchmark_summary.csv')}"
    if send_email(subject, body, a.email_to):
        print("[bench] summary email sent")

    print(f"[bench] finished: {n_ok}/{len(rows)} ok, {n_fail} failed")
    sys.exit(1 if n_fail else 0)   # non-zero lets the sweep wrapper auto-retry failures


if __name__ == "__main__":
    main()
