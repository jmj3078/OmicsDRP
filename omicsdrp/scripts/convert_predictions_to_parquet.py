"""One-off: convert legacy ``folds/fold_k_predictions.pkl`` (saved when pyarrow
was missing) to ``.parquet``. Each conversion is verified with an exact
roundtrip (values, dtypes, index) before the .pkl is optionally removed.

    conda activate omicsdrp
    python convert_predictions_to_parquet.py                 # write .parquet, keep .pkl
    python convert_predictions_to_parquet.py --delete-pkl    # also remove verified .pkl
"""
import argparse
import glob
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="./Results")
ap.add_argument("--delete-pkl", action="store_true")
args = ap.parse_args()

pkls = sorted(glob.glob(os.path.join(args.root, "**/folds/*_predictions.pkl"), recursive=True))
for p in pkls:
    q = p[:-len(".pkl")] + ".parquet"
    df = pd.read_pickle(p)
    df.to_parquet(q)
    pd.testing.assert_frame_equal(df, pd.read_parquet(q))
    if args.delete_pkl:
        os.remove(p)
print(f"converted {len(pkls)} files" + (" (pkl removed)" if args.delete_pkl else ""))
