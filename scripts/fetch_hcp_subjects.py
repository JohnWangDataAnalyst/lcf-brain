#!/usr/bin/env python
# scripts/fetch_hcp_subjects.py
# Convert per-subject SC/FC CSV layout into standardized .npy matrices.
# SC: <sc_root>/<SUB>/<SUB>_new_atlas_Yeo.nii.csv
# FC: <fc_root>/<SUB>_Schaefer2018_200Parcels_7Networks.csv

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_csv_matrix(p, dtype=float):
    df = pd.read_csv(p, header=None)
    return df.values.astype(dtype)

def to_corr(S):
    # robust, symmetric conversion to correlation
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    C = (S / d[None, :]) / d[:, None]
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C

def laplacian_from_sc(W, normalize=False, negate=False):
    # Build undirected Laplacian from SC weights.
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-12, None)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = np.diag(d) - W
    if negate:
        L = -L
    return L

def main(args):
    sc_dir = Path(args.sc_root)
    fc_dir = Path(args.fc_root)
    out_sc = Path(args.out_root) / "sc"; out_sc.mkdir(parents=True, exist_ok=True)
    out_fc = Path(args.out_root) / "fc"; out_fc.mkdir(parents=True, exist_ok=True)

    # Subjects list
    if args.subjects:
        subs = [s.strip() for s in Path(args.subjects).read_text().splitlines() if s.strip()]
    else:
        # infer from subfolders in sc_root
        subs = [p.name for p in sc_dir.iterdir() if p.is_dir()]

    kept, missing = 0, []
    for sub in subs:
        sc_csv = sc_dir / sub / f"{sub}_new_atlas_Yeo.nii.csv"
        fc_csv = fc_dir / f"{sub}_Schaefer2018_200Parcels_7Networks.csv"

        if not sc_csv.exists() or not fc_csv.exists():
            missing.append((sub, sc_csv.exists(), fc_csv.exists()))
            continue

        try:
            # ---- SC ----
            W = read_csv_matrix(sc_csv)
            if args.sc_abs:
                W = np.abs(W)
            if args.sc_thresh is not None:
                W = np.where(np.abs(W) >= args.sc_thresh, W, 0.0)

            if args.sc_make_laplacian:
                L = laplacian_from_sc(W, normalize=args.sc_norm_lap, negate=args.sc_negate_lap)
                np.save(out_sc / f"{sub}.npy", L.astype(np.float64))
            else:
                W = 0.5 * (W + W.T)
                np.fill_diagonal(W, 0.0)
                np.save(out_sc / f"{sub}.npy", W.astype(np.float64))

            # ---- FC ----
            FC = read_csv_matrix(fc_csv)
            if args.fc_to_corr:
                FC = to_corr(FC)
            else:
                FC = 0.5 * (FC + FC.T); np.fill_diagonal(FC, 1.0)
            np.save(out_fc / f"{sub}.npy", FC.astype(np.float64))

            kept += 1
        except Exception as e:
            print(f"[error] {sub}: {e}")

    print(f"[done] wrote {kept} subjects to {out_sc} and {out_fc}")
    if missing:
        print("[missing] some subjects skipped (SC/FC not found):")
        for sub, has_sc, has_fc in missing[:20]:
            print(f"  - {sub:>10s}  SC:{has_sc}  FC:{has_fc}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch per-subject SC/FC CSVs into standardized .npy files.")
    ap.add_argument("--sc-root", required=True, help="Root folder containing per-subject SC subfolders.")
    ap.add_argument("--fc-root", required=True, help="Root folder containing FC CSVs (flat).")
    ap.add_argument("--out-root", default="data/hcp_200", help="Output root (creates sc/ and fc/).")
    ap.add_argument("--subjects", default=None, help="Optional text file with one subject ID per line.")

    # SC options
    ap.add_argument("--sc-make-laplacian", action="store_true", help="Convert SC adjacency to (possibly normalized) Laplacian.")
    ap.add_argument("--sc-norm-lap", action="store_true", help="Use normalized Laplacian if making Laplacian.")
    ap.add_argument("--sc-negate-lap", action="store_true", help="Negate the Laplacian (i.e., -L) to match your A_c convention.")
    ap.add_argument("--sc-abs", action="store_true", help="Take absolute value of SC weights before processing.")
    ap.add_argument("--sc-thresh", type=float, default=None, help="Zero out |W| < threshold before Laplacian.")

    # FC options
    ap.add_argument("--fc-to-corr", action="store_true", help="Convert FC CSV (covariance) into correlation.")

    args = ap.parse_args()
    main(args)
