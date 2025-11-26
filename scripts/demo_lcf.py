#!/usr/bin/env python
# scripts/demo_lcf.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- import the core API from your package ----
from lcf.core import (
    build_inputs,
    drift_from_fc_lyap,
    drift_from_mou,
    kc_least_squares,
    ks_lqr,
    cov_to_corr,
    r_lower_triangle,
)

def lower_triangle_vectors(A, B):
    iu = np.tril_indices_from(A, k=-1)
    return A[iu], B[iu]

def load_one_subject(sc_dir, fc_dir):
    sc_files = sorted(Path(sc_dir).glob("*.npy"))
    fc_files = sorted(Path(fc_dir).glob("*.npy"))
    if not sc_files or not fc_files:
        return None, None
    # simple pairing by index
    return np.load(sc_files[0]), np.load(fc_files[0])

def synthetic_subject(n=100, seed=0):
    rng = np.random.default_rng(seed)
    # random Laplacian-like SC
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(1))
    L = D - W
    # synthetic FC (spiked covariance)
    U = np.linalg.qr(rng.normal(size=(n, n)))[0]
    lam = np.linspace(2.0, 0.2, n)
    FC = U @ np.diag(lam) @ U.T
    FC = cov_to_corr(FC)
    return L, FC

def demo(args):
    out_figs = Path("figs"); out_figs.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    if args.synthetic:
        L, FC_emp = synthetic_subject(n=args.n, seed=args.seed)
        print(f"[demo] synthetic subject: n={args.n}")
    else:
        L, FC_emp = load_one_subject(args.sc, args.fc)
        if L is None or FC_emp is None:
            print("[warn] no .npy files found under data/hcp_200/sc or fc; falling back to synthetic.")
            L, FC_emp = synthetic_subject(n=args.n, seed=args.seed)

    n = FC_emp.shape[0]

    # ---------- Step 1: drift ----------
    if args.mou and args.ts is not None:
        X = np.load(args.ts)  # expecting (T, n)
        A_star, Qhat, FC_emp_ts = drift_from_mou(X, dt=args.dt, lam=args.ridge_ts, q_mode=args.q_mode)
        # prefer FC from file if provided; FC_emp_ts shows how MOU sees the TS
        print(f"[drift] MOU: ridge={args.ridge_ts}, q_mode={args.q_mode}")
    else:
        A_star = drift_from_fc_lyap(FC_emp, L=L, gamma=args.gamma, eps=args.eps, q_sigma2=args.sigma2)
        print(f"[drift] FC-lyap: gamma={args.gamma}, eps={args.eps}, sigma2={args.sigma2}")

    # ---------- Step A: inputs ----------
    if args.mode == "eigenspace":
        Bc, Bs, meta = build_inputs(FC_emp, mode="eigenspace", tau=args.tau, return_parts=True)
        print(f"[inputs] eigenspace split: k={meta['k']} important modes (Bs), residual dim={Bc.shape[1]} (Bc)")
    else:
        assert args.ctx is not None and args.sub is not None, \
            "For anatomical mode, pass --ctx and --sub index lists (comma-separated)."
        idx_ctx = np.fromstring(args.ctx, sep=",", dtype=int)
        idx_sub = np.fromstring(args.sub, sep=",", dtype=int)
        Bc, Bs = build_inputs(FC_emp, mode="anatomical", idx_ctx=idx_ctx, idx_sub=idx_sub, r_s=args.r_s)
        print(f"[inputs] anatomical split: cortex cols={Bc.shape[1]}, subcortex cols={Bs.shape[1]} (r_s={args.r_s})")

    # ---------- Step 2: Kc ----------
    Kc, A_cl, Ec = kc_least_squares(A_star, Bc, ridge=args.ridge_kc)
    print(f"[Kc] Ec={Ec:.4g} (||Kc||_F^2), Bc shape={Bc.shape}")

    # ---------- Step 3: Ks ----------
    Q = np.eye(n) * args.q_scale
    R = np.eye(Bs.shape[1]) * args.r_scale
    Ks, A_eff, Es, FC_pred, _ = ks_lqr(A_cl, Bs, Q=Q, R=R, Sigma=np.eye(n))
    print(f"[Ks] Es={Es:.4g} (||Ks||_F^2), Bs shape={Bs.shape}")

    # ---------- Metrics & plot ----------
    r = r_lower_triangle(FC_emp, FC_pred)
    print(f"[fit] lower-triangle corr r={r:.3f}")

    x, y = lower_triangle_vectors(FC_emp, FC_pred)
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(x, y, s=6, alpha=0.6)
    lim = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = max(lim, 1.0)
    plt.plot([-lim, lim], [-lim, lim], lw=1)
    plt.xlabel("FC (empirical) lower tri")
    plt.ylabel("FC (predicted) lower tri")
    plt.title(f"r = {r:.3f} | Ec={Ec:.2f}, Es={Es:.2f}")
    fig_path = out_figs / "demo_scatter.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    print(f"[fig] wrote {fig_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", default="data/hcp_200/sc", help="folder with SC .npy")
    ap.add_argument("--fc", default="data/hcp_200/fc", help="folder with FC .npy")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic subject if no files found (or force)")
    ap.add_argument("--n", type=int, default=100, help="synthetic size if synthetic mode")
    ap.add_argument("--seed", type=int, default=0)
    # drift options
    ap.add_argument("--mou", action="store_true", help="use OU/MOU drift if --ts provided")
    ap.add_argument("--ts", default=None, help="path to time series .npy (T x N) for MOU")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ridge-ts", type=float, default=1e-2)
    ap.add_argument("--q-mode", choices=["diag","iso","full"], default="diag")
    ap.add_argument("--gamma", type=float, default=1.0, help="structural weight for -gamma*L")
    ap.add_argument("--eps", type=float, default=1e-3, help="stability shift -eps*I")
    ap.add_argument("--sigma2", type=float, default=1.0, help="Lyapunov noise scale (FC-only drift)")
    # input modes
    ap.add_argument("--mode", choices=["eigenspace","anatomical"], default="eigenspace")
    ap.add_argument("--tau", type=float, default=0.5, help="variance threshold for eigenspace split")
    ap.add_argument("--ctx", type=str, default=None, help="comma-separated cortex indices (anatomical mode)")
    ap.add_argument("--sub", type=str, default=None, help="comma-separated subcortex indices (anatomical mode)")
    ap.add_argument("--r-s", type=int, default=None, help="rank for Bs in anatomical mode")
    # Kc / Ks hyperparams
    ap.add_argument("--ridge-kc", type=float, default=1e-8)
    ap.add_argument("--q-scale", type=float, default=1.0)
    ap.add_argument("--r-scale", type=float, default=1.0)
    args = ap.parse_args()
    demo(args)
