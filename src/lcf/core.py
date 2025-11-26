# src/lcf/core.py
"""
Core API: three-step linear control framework for SC→FC alignment.

(1) Drift:
    - drift_from_mou(X, ...)           # OU/MOU from time series
    - drift_from_fc_lyap(FC, ...)      # FC-eigenspace Lyapunov (FC-only)

(2) Kc via ridge LS on residual subspace:
    - kc_least_squares(A_star, Bc, ...)

(3) Ks via LQR/Riccati on important subspace:
    - ks_lqr(A_cl, Bs, Q, R, Sigma, ...)
"""

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import solve_continuous_lyapunov, solve_continuous_are

# ---------- small helpers ----------

def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix (stable)."""
    S = np.asarray(S, float)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return (S / d[None, :]) / d[:, None]

def r_lower_triangle(FC_emp: np.ndarray, FC_pred: np.ndarray) -> float:
    """Correlation (z-scored) between lower triangles of two FC (correlation) matrices."""
    FC_emp = np.asarray(FC_emp, float)
    FC_pred = np.asarray(FC_pred, float)
    iu = np.tril_indices_from(FC_emp, k=-1)
    x, y = FC_emp[iu], FC_pred[iu]
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float((x @ y) / len(x))

# ============================================================
# (1) DRIFT: OU/MOU (time series) or FC-eigenspace Lyapunov
# ============================================================

def drift_from_mou(X: np.ndarray, dt: float = 1.0, lam: float = 0.0, q_mode: str = "diag"):
    """
    Estimate continuous-time OU/MOU drift A from time series:
        dX/dt = A X + noise
    Returns: A_hat (N,N), Q (N,N), FC_emp (N,N).
    q_mode in {"diag","iso","full"} for residual diffusion.
    """
    X = np.asarray(X, float)
    X = X - X.mean(0, keepdims=True)
    dX = (X[1:] - X[:-1]) / dt            # (T-1, N)
    X0 = X[:-1]                           # (T-1, N)
    G = X0.T @ X0
    A = (np.linalg.solve(G + lam*np.eye(G.shape[0]), X0.T @ dX)).T  # (N,N)

    R = dX - X0 @ A.T                     # residuals
    Q_full = np.cov(R.T, bias=False)
    if q_mode == "full":
        Q = Q_full
    elif q_mode == "iso":
        sig2 = np.trace(Q_full) / Q_full.shape[0]
        Q = sig2 * np.eye(Q_full.shape[0])
    else:  # "diag"
        Q = np.diag(np.diag(Q_full))

    FC_emp = np.corrcoef(X.T)
    return A, Q, FC_emp


def drift_from_fc_lyap(FC: np.ndarray, L: np.ndarray | None = None,
                       gamma: float = 0.0, eps: float = 0.0, q_sigma2: float = 1.0) -> np.ndarray:
    """
    FC-only initializer in the FC eigenspace (closed-form, stable).
    Let FC = U Λ U^T, assume Q = σ^2 I. Solve mode-wise:
        a_i = -σ^2 / (2 λ_i), then A* = U diag(a_i) U^T (+ optional -γL - εI).
    """
    FC = np.asarray(FC, float)
    lam, U = eigh((FC + FC.T) / 2.0)
    lam = np.clip(lam, 1e-8, None)                 # avoid divide-by-zero
    a = -(q_sigma2) / (2.0 * lam)                  # mode-wise drift
    A_star = U @ np.diag(a) @ U.T
    if L is not None:
        A_star = A_star + (-gamma * L - eps * np.eye(L.shape[0]))
    return A_star

# ============================================================
# (2) Kc via least squares on residual subspace (Bc)
# ============================================================

def kc_least_squares(A_star: np.ndarray, Bc: np.ndarray,
                     A_target: np.ndarray | None = None,
                     ridge: float = 1e-8, K_mask: np.ndarray | None = None):
    """
    Minimize ||A_star + Bc Kc - A_target||_F^2 + ridge||Kc||_F^2.
    If A_target is None, uses a tiny leftward shift of A_star as target.
    Returns: Kc (m,N), A_cl (N,N), Ec (scalar).
    """
    A_star = np.asarray(A_star, float)
    Bc = np.asarray(Bc, float)
    n = A_star.shape[0]
    if A_target is None:
        A_target = A_star - 1e-2 * np.eye(n)

    D = A_target - A_star                       # (N,N)
    BtB = Bc.T @ Bc                             # (m,m)
    Kc = np.linalg.solve(BtB + ridge*np.eye(Bc.shape[1]), Bc.T @ D)  # (m,N)

    if K_mask is not None:
        Kc = Kc * K_mask

    A_cl = A_star + Bc @ Kc
    Ec = float(np.linalg.norm(Kc, 'fro')**2)
    return Kc, A_cl, Ec

# ============================================================
# (3) Ks via LQR/Riccati on important subspace (Bs)
# ============================================================

def ks_lqr(A_cl: np.ndarray, Bs: np.ndarray,
           Q: np.ndarray | None = None, R: np.ndarray | None = None,
           Sigma: np.ndarray | None = None, return_cov: bool = True):
    """
    CARE on (A_cl, Bs, Q, R) with u = -Ks x:
        P = CARE(A,B,Q,R), Ks = R^{-1} B^T P, A_eff = A_cl - Bs Ks.
    If Sigma is given, solve Lyapunov for covariance: A_eff C + C A_eff^T + Sigma = 0.
    Returns: Ks (m,N), A_eff (N,N), Es (scalar), FC_pred_corr (N,N|None), FC_pred_cov (N,N|None).
    """
    A = np.asarray(A_cl, float)
    B = np.asarray(Bs, float)
    n, m = B.shape
    if Q is None: Q = np.eye(n)
    if R is None: R = np.eye(m)

    P = solve_continuous_are(A, B, Q, R)
    Ks = np.linalg.solve(R, B.T @ P)            # u = -Ks x
    A_eff = A - B @ Ks
    Es = float(np.linalg.norm(Ks, 'fro')**2)

    FC_pred_corr = FC_pred_cov = None
    if return_cov:
        Sigma = np.eye(n) if Sigma is None else Sigma
        C = solve_continuous_lyapunov(A_eff, Sigma)   # A C + C A^T + Sigma = 0
        FC_pred_cov = C
        FC_pred_corr = cov_to_corr(C)

    return Ks, A_eff, Es, FC_pred_corr, FC_pred_cov
