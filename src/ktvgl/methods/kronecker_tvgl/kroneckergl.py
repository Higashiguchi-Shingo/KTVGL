"""
Time-varying Kronecker Graphical Lasso (TVKGL) for time-varying covariance estimation.
"""
from copy import deepcopy
import numpy as np
from numpy.linalg import slogdet
from .solver import time_varying_graphical_lasso
from .initialize import init_precision, _init_from_empirical, _normalize_X_seq
from typing import List, Optional, Iterator
import time

def compute_mean(arrays):
    total = sum(a.sum() for a in arrays)
    count = sum(a.size for a in arrays)
    return total / count

def sym(A: np.ndarray) -> np.ndarray:
    "A: shape (dm, dm)"
    if isinstance(A, list):
        T = len(A)
        for t in range(T):
            A[t] = sym(A[t])
        return A
    
    if A.ndim == 2:
        return 0.5 * (A + A.T)
    elif A.ndim == 3:
        T = A.shape[0]
        for t in range(T):
            A[t] = sym(A[t])
        return A
    raise ValueError("Input must be a 2D or 3D array.")

def mode_product(X: np.ndarray, A: np.ndarray, mode: int) -> np.ndarray:
    Xm = np.moveaxis(X, mode, 0)                  # (d_mode, ...)
    d_mode = Xm.shape[0]
    Xm2 = Xm.reshape(d_mode, -1)                  # (d_mode, prod(other))
    Ym2 = A @ Xm2                                  # (d_mode, prod(other))
    Ym = Ym2.reshape(Xm.shape)
    return np.moveaxis(Ym, 0, mode)


def _time_len(X_seq) -> int:
    return len(X_seq)


def _slice_time(X_seq, start: int, end: int):
    return X_seq[start:end]


def _samples_at_t(X_t, n_modes: int) -> list[np.ndarray]:
    if not isinstance(X_t, np.ndarray):
        raise ValueError("Each normalized X_seq[t] must be ndarray.")
    if X_t.ndim != n_modes + 1:
        raise ValueError(
            f"Invalid normalized timestep ndim={X_t.ndim}; expected {n_modes + 1}."
        )
    return [X_t[n] for n in range(X_t.shape[0])]


def _infer_dims(X_seq, n_modes: int) -> tuple[int, ...]:
    T = _time_len(X_seq)
    if T == 0:
        raise ValueError("X_seq must be non-empty.")
    X0 = X_seq[0]
    samples0 = _samples_at_t(X0, n_modes)
    if len(samples0) == 0:
        raise ValueError("X_seq[0] must contain at least one sample.")
    return tuple(samples0[0].shape)


def _time_sample_counts(X_seq, n_modes: int) -> np.ndarray:
    T = _time_len(X_seq)
    counts = np.zeros(T, dtype=float)
    for t in range(T):
        X_t = X_seq[t]
        counts[t] = len(_samples_at_t(X_t, n_modes))
    return counts


"""def initialize_thetas(dims, method="identity", X_list=None, mle_steps=0, eps=1e-6):
    n = len(dims)
    Theta = [np.eye(d) for d in dims]

    if method == "identity" or X_list is None or mle_steps <= 0:
        return Theta

    for _ in range(mle_steps):
        for m in range(n):
            S_hat, D_minus = compute_Shat(X_list, Theta, m)
            S_bar = S_hat / max(D_minus, 1)
            d_m = S_bar.shape[0]
            jitter = eps * np.trace(S_bar) / max(d_m, 1)
            S_bar_reg = sym(S_bar) + jitter * np.eye(d_m)
            Theta[m] = np.linalg.inv(S_bar_reg)
            Theta[m] = sym(Theta[m])

    return Theta"""


def compute_Shat(
    X_seq: np.ndarray,
    Thetas_seq: list[list[np.ndarray]],
    mode: int,
    jitter_scale: float = 1e-8
) -> np.ndarray:
    """
    X_seq: array (T, d1, ..., dM)
    Thetas_seq: list: (T, M, dm, dm), Thetas_seq[t] is list [Θ_t^(1),...,Θ_t^(M)]
    mode: target mode

    Return: S_hats (T, d_mode, d_mode), S_t^(m) for each timestep
    """
    T = X_seq.shape[0]
    dims = X_seq.shape[1:]  # (d1,...,dM)
    M = len(dims)
    assert 0 <= mode < M, "mode index out of range"
    d_m = dims[mode]
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))

    S_hats = np.zeros((T, d_m, d_m), dtype=float)

    for t in range(T):
        #X_t = X_seq[t] - compute_mean(X_seq[t])
        X_t = X_seq[t]
        Thetas_t = Thetas_seq[t]

        Y = X_t
        for ell in range(M):
            if ell == mode:
                continue
            Y = mode_product(Y, Thetas_t[ell], mode=ell)

        Ym = np.moveaxis(Y, mode, 0).reshape(d_m, D_minus)
        Xm = np.moveaxis(X_t, mode, 0).reshape(d_m, D_minus)
        S = (Ym @ Xm.T) / D_minus
        S = sym(S)

        eps = jitter_scale * max(1.0, np.trace(S) / max(d_m, 1))
        S_hats[t] = S + eps * np.eye(d_m)

    return S_hats


def compute_Shat_multi_sample(
    X_seq,
    Thetas_seq: list[list[np.ndarray]],
    mode: int,
    jitter_scale: float = 1e-8
) -> np.ndarray:
    """
    X_seq: list-like (length T), variable #samples per timestep
      - X_seq[t] can be:
        1) ndarray (N_t, d1, ..., dM)
        2) list/tuple of ndarray [(d1, ..., dM), ...] (length N_t)
        3) ndarray (d1, ..., dM)  # interpreted as N_t = 1
    Thetas_seq: list: (T, M, dm, dm), Thetas_seq[t] is list [Θ_t^(1),...,Θ_t^(M)]
    mode: target mode

    Return: S_hats (T, d_mode, d_mode), S_t^(m) for each timestep
    """
    M = len(Thetas_seq[0])
    X_norm = _normalize_X_seq(X_seq, n_modes=M)
    T = _time_len(X_norm)
    if T == 0:
        raise ValueError("X_seq must be non-empty.")
    if len(Thetas_seq) != T:
        raise ValueError("len(X_seq) and len(Thetas_seq) must match.")
    dims = _infer_dims(X_norm, M)  # (d1,...,dM)
    assert 0 <= mode < M, "mode index out of range"
    d_m = dims[mode]
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))

    S_hats = np.zeros((T, d_m, d_m), dtype=float)

    for t in range(T):
        Thetas_t = Thetas_seq[t]
        samples_t = _samples_at_t(X_norm[t], M)
        N_t = len(samples_t)
        if N_t == 0:
            raise ValueError(f"X_seq[{t}] must contain at least one sample.")
        S_t = np.zeros((d_m, d_m), dtype=float)

        for X_tn in samples_t:
            if X_tn.shape != dims:
                raise ValueError(
                    f"Sample shape mismatch at t={t}: expected {dims}, got {X_tn.shape}."
                )

            Y = X_tn
            for ell in range(M):
                if ell == mode:
                    continue
                Y = mode_product(Y, Thetas_t[ell], mode=ell)

            Ym = np.moveaxis(Y, mode, 0).reshape(d_m, D_minus)
            Xm = np.moveaxis(X_tn, mode, 0).reshape(d_m, D_minus)
            S_t += (Ym @ Xm.T) / (D_minus * N_t)

        S_t = sym(S_t)
        eps = jitter_scale * max(1.0, np.trace(S_t) / max(d_m, 1))
        S_hats[t] = S_t + eps * np.eye(d_m)

    return S_hats


def compute_scale_factor(Theta: np.ndarray, gauge: str) -> float:
    """
    Returns: scaled coef, g(Theta)
    """
    d = Theta.shape[0]
    if gauge == "trace":
        return np.trace(Theta) / max(d, 1)
    elif gauge == "fro":
        return np.linalg.norm(Theta, "fro") / np.sqrt(max(d, 1))
    elif gauge == "det":
        sign, logdet = np.linalg.slogdet(Theta)
        return np.exp(logdet / max(d, 1))
    elif gauge is None:
        return 1.0
    else:
        raise ValueError("gauge must be 'trace', 'fro', or 'det'.")


def scale_correct(
    Thetas_t: list[np.ndarray],  # list of ndarray, [Θ_t^(1), ..., Θ_t^(M)], shape of Θ_t^(m): (dm, dm)
    mode: int,
    *,
    gauge: str = "trace",        # 'trace' | 'fro' | 'det'
    strategy: str = "anchor",    # 'anchor' | 'equal' | 'none'
    anchor: int | str = "next"   # 'next' | 'prev' | int (モード番号)
) -> list[np.ndarray]:
    """
    Thetas_t: list of ndarray, [Θ_t^(1), ..., Θ_t^(M)], shape of Θ_t^(m): (dm, dm)
    mode:     target mode
    
    Returns: updated Thetas_t
    """
    M = len(Thetas_t)  # number of modes
    assert 0 <= mode < M

    Thetas_t_copy = [A.copy() for A in Thetas_t]  # copy of Thetas_t

    # (1) compute scaling factor
    Theta_m = sym(Thetas_t_copy[mode]) # Θ_t^(m): (dm, dm)
    s = compute_scale_factor(Theta=Theta_m, gauge=gauge)
    if not np.isfinite(s) or s <= 0:
        return Thetas_t_copy  # avoid scaling if s is invalid

    # (2) scaling
    Thetas_t_copy[mode] = sym(Theta_m / s)

    # (3) scale correction
    if strategy == "anchor":
        if isinstance(anchor, int):
            q = anchor % M
            if q == mode:
                q = (mode + 1) % M
        elif anchor == "prev":
            q = (mode - 1) % M
        else:  # 'next' 既定
            q = (mode + 1) % M
        Thetas_t_copy[q] = sym(Thetas_t_copy[q] * s)
    elif strategy == "equal":
        alpha = s ** (1.0 / max(M - 1, 1))
        for m in range(M):
            if m == mode: 
                continue
            Thetas_t_copy[m] = sym(Thetas_t_copy[m] * alpha)
    else:
        raise ValueError("strategy must be 'anchor', 'equal', or 'none'.")
    return Thetas_t_copy


def scale_correct_seq(
    Thetas_seq: list[list[np.ndarray]], # shape: (T, M, dm, dm)
    mode: int,
    *,
    gauge: str = "trace",
    strategy: str = "anchor",
    anchor: int | str = "next"
) -> list[list[np.ndarray]]:
    """
    Thetas_seq: list of list, {Thetas_t} (t = 0,1,...,T-1),
    Thetas_t: list, {Theta_t^(1), ..., Theta_t^(M)} (m = 1 ~ M)
    Theta_t: np.ndarray, shape = (d_m, d_m), precision matrix

    Returns: list of scaled precision matrices
    """
    T = len(Thetas_seq)
    Thetas_seq_scaled = []
    for t in range(T):
        Tcorr = scale_correct(
            Thetas_t=Thetas_seq[t], 
            mode=mode, 
            gauge=gauge, 
            strategy=strategy, 
            anchor=anchor
        )
        Thetas_seq_scaled.append(Tcorr)
    return Thetas_seq_scaled


def update_one_mode_tvgl(
    X_seq,                                # shape: (T, d1, ..., dM) or multi-sample list-like
    Thetas_seq: list[list[np.ndarray]],   # list: (T, M, dm, dm), array: (dm, dm)
    mode: int,                            
    tvgl_solver,              # tvgl_solver(S_hats, lam_bar, **kwargs) -> np.ndarray (T, d_m, d_m)
    compute_Shat_func,        # compute_Shat(X_seq, Thetas_seq, mode) -> (T, d_m, d_m)
    *,
    lambda_m: float = 0.01,   # float: strength of L1 norm for mode m
    rho_m: float = 1.0,       # float: strength of L2 norm for mode m
    max_iter_tvgl: int = 500,  # number of iteration for TVGL
    gauge="trace",
    strategy="anchor",
    anchor="next",
    verbose=False,
    psi="laplacian",
    init_thetas: np.ndarray = None
):
    """
    Update Thetas_seq once for mode m.

    Returns: 
    Thetas_seq: updated precision matrices
    """
    T = len(Thetas_seq)
    M = len(Thetas_seq[0])
    X_norm = _normalize_X_seq(X_seq, n_modes=M)
    dims = _infer_dims(X_norm, M)
    d_m = dims[mode]
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))
    n_samples = _time_sample_counts(X_norm, M)

    # (a) Compute S^(m)
    S_hats = compute_Shat_func(
        X_seq=X_seq, 
        Thetas_seq=Thetas_seq, 
        mode=mode
    )   # S_hats: shape (T, d_m, d_m)

    # (b) Solve TVGL -> get Theta (i.e., precision matrices)
    prev_matrices = [Thetas_seq[t][mode].copy() for t in range(T)]
    lam_bar = lambda_m / D_minus
    rho_bar = rho_m / D_minus
    try:
        Theta_m_seq = tvgl_solver(
            emp_cov=S_hats, 
            alpha=lam_bar,
            rho=rho_bar,
            max_iter=max_iter_tvgl,
            n_samples=n_samples,
            prev_matrices=prev_matrices,
            verbose=verbose,
            psi=psi,
            init_inv_cov=init_thetas
        ) # (T, d_m, d_m)
    except TypeError:
        Theta_m_seq = tvgl_solver(
            emp_cov=S_hats, 
            alpha=lam_bar,
            rho=rho_bar,
            max_iter=max_iter_tvgl,
            prev_matrices=prev_matrices,
            verbose=verbose,
            psi=psi,
            init_inv_cov=init_thetas
        ) # (T, d_m, d_m)

    # (c) Scale correction
    updated = [] # updated Thetas_seq, shape: (T, M, dm, dm)
    for t in range(T):
        row = [A.copy() for A in Thetas_seq[t]]
        row[mode] = sym(Theta_m_seq[t])
        updated.append(row)

    corrected = scale_correct_seq(
                    updated, 
                    mode=mode, 
                    gauge=gauge, 
                    strategy=strategy, 
                    anchor=anchor
                )

    max_rel_change = 0.0
    for t in range(T):
        new_m = corrected[t][mode]
        old_m = scale_correct_seq(
            Thetas_seq, 
            mode=mode, 
            gauge=gauge, 
            strategy=strategy, 
            anchor=anchor
        )[t][mode]
        denom = max(1e-12, np.linalg.norm(old_m, "fro"))
        rel = np.linalg.norm(new_m - old_m, "fro") / denom
        max_rel_change += rel / T

    Thetas_seq = corrected
    return Thetas_seq, max_rel_change


# kronecker TVGL
def kronecker_timevarying_graphical_lasso(
    X_seq,                                 # shape: (T, d1, ..., dn) or multi-sample list-like
    Thetas_seq: list[list[np.ndarray]],    # list: (T, M, dm, dm), array: (dm, dm)
    lambdas: list[float],                  # list: (M, )
    rhos: list[float],                     # list: (M, )
    tvgl_solver,                           # tvgl_solver(S_hats, lam_bar, **kwargs) -> np.ndarray (T, d_m, d_m)
    compute_Shat_func,                     # compute_Shat(X_seq, Thetas_seq, mode) -> (T, d_m, d_m)
    gauge: str = "trace",
    strategy: str = "anchor",
    anchor: str = "next",
    max_iter: int = 10,
    max_iter_tvgl: int = 50,
    min_iter: int = 4,
    tol_flipflop: float = 1e-4,
    verbose: bool = False,
    psi: str = "laplacian", # "laplacian" or "l1"
    init_thetas: list[np.ndarray] = None,
    early_stopping: bool = True
):
    X_seq = _normalize_X_seq(X_seq, n_modes=len(Thetas_seq[0]))
    pre_obj = None
    M = len(Thetas_seq[0])

    for iter in range(max_iter):
        print("Iteration:", str(iter))
        for mode in range(M):
            if verbose: print("mode:", str(mode))
            Thetas_seq, max_rel_change = update_one_mode_tvgl(
                X_seq=X_seq,
                Thetas_seq=Thetas_seq,
                mode=mode,
                lambda_m=lambdas[mode],
                rho_m=rhos[mode],
                tvgl_solver=tvgl_solver,
                compute_Shat_func=compute_Shat_func,
                max_iter_tvgl=max_iter_tvgl,
                gauge=gauge,
                strategy=strategy,
                anchor=anchor,
                verbose=verbose,
                psi=psi,
                init_thetas=init_thetas[mode] if init_thetas is not None else None
            )
        obj = objective_function(X_seq, Thetas_seq, lambdas, rhos)

        print("Score:", str(obj))

        if pre_obj is not None:
            if ((pre_obj - obj) / np.abs(pre_obj)) < tol_flipflop:
                if iter >= min_iter - 1:
                    if verbose: print("(Outer loop) Early stopping at iteration:", str(iter))
                    if early_stopping: break
        pre_obj = obj

    return Thetas_seq


class KroneckerTVGL:
    def __init__(self, 
                 lambdas, 
                 rhos, 
                 tvgl_solver, 
                 init_method="empirical",
                 gauge="trace", 
                 strategy="equal", 
                 anchor="next", 
                 max_iter=10,
                 min_iter=5,
                 max_iter_tvgl=500,
                 tol_flipflop=1e-4,
                 verbose=False,
                 psi="laplacian", # "laplacian" or "l1"
                 early_stopping=True
    ):
        self.lambdas = lambdas
        self.rhos = rhos
        self.tvgl_solver = tvgl_solver
        self.compute_Shat_func = compute_Shat_multi_sample
        self.gauge = gauge
        self.strategy = strategy
        self.anchor = anchor
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_iter_tvgl = max_iter_tvgl
        self.init_method = init_method
        self.tol_flipflop = tol_flipflop
        self.verbose = verbose
        self.psi = psi
        self.early_stopping = early_stopping

    def fit(self, X_seq):
        self.X_seq = _normalize_X_seq(X_seq, n_modes=len(self.lambdas))
        self.dims = _infer_dims(self.X_seq, len(self.lambdas))

        # Initialize Thetas_seq by identity matrices
        Thetas_seq = init_precision(self.X_seq, method=self.init_method, lambdas=self.lambdas, n_modes=len(self.lambdas))

        # Run Kronecker time-varying graphical lasso
        _out = kronecker_timevarying_graphical_lasso(
            X_seq=self.X_seq,
            Thetas_seq=Thetas_seq,
            lambdas=self.lambdas,
            rhos=self.rhos,
            tvgl_solver=self.tvgl_solver,
            compute_Shat_func=self.compute_Shat_func,
            gauge=self.gauge,
            strategy=self.strategy,
            anchor=self.anchor,
            max_iter=self.max_iter,
            min_iter=self.min_iter,
            max_iter_tvgl=self.max_iter_tvgl,
            tol_flipflop=self.tol_flipflop,
            verbose=self.verbose,
            psi=self.psi,
            early_stopping=self.early_stopping
        )
        self.Thetas_seq = _out

        return self.Thetas_seq

    @staticmethod
    def compute_covariances_from_thetas(
        Thetas_seq: list[list[np.ndarray]]
    ) -> list[list[np.ndarray]]:
        """
        Compute covariance matrices by inverting each precision matrix in Thetas_seq.
        """
        covariances_seq = []
        for thetas_t in Thetas_seq:
            cov_t = []
            for theta_m in thetas_t:
                cov_m = np.linalg.pinv(theta_m, hermitian=True)
                cov_t.append(sym(cov_m))
            covariances_seq.append(cov_t)
        return covariances_seq

    @staticmethod
    def compute_partial_correlations_from_thetas(
        Thetas_seq: list[list[np.ndarray]],
        eps: float = 1e-12
    ) -> list[list[np.ndarray]]:
        """
        Compute partial correlation matrices from precision matrices in Thetas_seq.
        """
        partial_corr_seq = []
        for thetas_t in Thetas_seq:
            pc_t = []
            for theta_m in thetas_t:
                theta = sym(theta_m)
                d = np.clip(np.diag(theta), eps, None)
                inv_std = 1.0 / np.sqrt(d)
                scale = np.outer(inv_std, inv_std)
                pc = -theta * scale
                np.fill_diagonal(pc, 1.0)
                pc_t.append(sym(pc))
            partial_corr_seq.append(pc_t)
        return partial_corr_seq
    
    def get_scaled_Thetas(self):
        Thetas_seq = deepcopy(self.Thetas_seq)
        for t in range(len(self.Thetas_seq)):
            for mode in range(len(self.dims)):
                Thetas_seq[t][mode] = Thetas_seq[t][mode] / (np.trace(Thetas_seq[t][mode]) / self.dims[mode])     
        return Thetas_seq


def objective_function(X_seq, Thetas_seq, lambdas, rhos, mode=0):
    T = len(Thetas_seq)
    M = len(Thetas_seq[0])
    X_norm = _normalize_X_seq(X_seq, n_modes=M)
    dims = _infer_dims(X_norm, M)  # (d1,...,dM)
    assert 0 <= mode < M, "mode index out of range"

    # Compute the objective function value
    obj = 0.0
    S_hat = compute_Shat_multi_sample(X_norm, Thetas_seq, mode)

    # log-likelihood terms
    for t in range(T):
        D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))
        obj += D_minus * np.trace(S_hat[t] @ Thetas_seq[t][mode])
        for m in range(M):
            D_minus = int(np.prod([dims[k] for k in range(M) if k != m]))
            obj -= D_minus * logdet(Thetas_seq[t][m])

    # L1 norm terms
    for t in range(T):
        for m in range(M):
            obj += lambdas[m] * l1_od_norm(Thetas_seq[t][m])

    # TV regularization terms
    for t in range(1,T):
        for m in range(M):
            obj += rhos[m] * squared_norm(Thetas_seq[t][m] - Thetas_seq[t-1][m])
    return obj

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.
    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order='K')
    return np.dot(x, x)

def logdet(A):
    """Compute log(det(A)) for A symmetric.
    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.

    Parameters
    ----------
    A : array-like
        The matrix.
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld

def logl(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return logdet(precision) - np.sum(emp_cov * precision)

def loss(S, K, n_samples=None):
    """Loss function for time-varying graphical lasso."""
    if n_samples is None:
        n_samples = np.ones(S.shape[0])
    return sum(
        -ni * logl(emp_cov, precision)
        for emp_cov, precision, ni in zip(S, K, n_samples))

def l1_norm(precision):
    """L1 norm."""
    return np.abs(precision).sum()

def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return l1_norm(precision) - np.abs(np.diag(precision)).sum()


class StreamKroneckerTVGL:
    def __init__(self,
        lambdas,
        rhos,
        tvgl_solver,
        window_size: int,
        step_size: int = 1,
        inverse_ridge: float = 1e-6,
        init_method: str = "empirical",
        gauge: str = "trace",
        strategy: str = "equal",
        anchor: str = "next",
        max_iter: int = 10,
        max_iter_tvgl: int = 500,
        tol_flipflop: float = 1e-4,
        verbose: bool = False,
        psi: str = "laplacian"
    ):
        assert window_size >= 2, "window_size must be >= 2"
        assert step_size >= 1, "step_size must be >= 1"
        self.window_size   = int(window_size)
        self.step_size     = int(step_size)
        self.inverse_ridge = float(inverse_ridge)
        self.lambdas = lambdas
        self.rhos = rhos
        self.tvgl_solver = tvgl_solver
        self.compute_Shat_func = compute_Shat_multi_sample
        self.gauge = gauge
        self.strategy = strategy
        self.anchor = anchor
        self.max_iter = max_iter
        self.max_iter_tvgl = max_iter_tvgl
        self.init_method = init_method
        self.tol_flipflop = tol_flipflop
        self.verbose = verbose
        self.psi = psi

        self._prev_thetas_seq: Optional[List[List[np.ndarray]]] = None

        self.latest_thetas_per_mode_: List[List[np.ndarray]] = []
        self.first_thetas_per_mode_: List[List[np.ndarray]] = []
        self.history_: List[List[List[np.ndarray]]] = []
        self.runtimes: List[float] = []

    def _init_for_window(self, X_window) -> List[List[np.ndarray]]:
        W = self.window_size
        M = len(self.lambdas)
        X_window = _normalize_X_seq(X_window, n_modes=M)
        dims = _infer_dims(X_window, M)

        if self._prev_thetas_seq is None:
            return init_precision(X_window, method=self.init_method, lambdas=self.lambdas)  # list[W][m] -> (d_m,d_m)

        overlap = max(0, W - self.step_size)
        init_list: List[List[np.ndarray]] = []

        for t in range(overlap):
            init_list.append([self._prev_thetas_seq[t + self.step_size][m].copy()
                              for m in range(M)])

        for t in range(overlap, W):
            X_t = X_window[t]
            init_list.append(_init_from_empirical(X_t, n_modes=M))

        return init_list  # list[W][m] -> (d_m,d_m)


    def _solve_window(self, X_window, is_init=False) -> List[List[np.ndarray]]:
        X_window = _normalize_X_seq(X_window, n_modes=len(self.lambdas))
        Thetas_seq = self._init_for_window(X_window)
        init_thetas = Thetas_seq[0].copy()
        if is_init:
            init_thetas = None

        out = kronecker_timevarying_graphical_lasso(
            X_seq=X_window,
            Thetas_seq=Thetas_seq,
            lambdas=self.lambdas,
            rhos=self.rhos,
            tvgl_solver=self.tvgl_solver,
            compute_Shat_func=self.compute_Shat_func,
            gauge=self.gauge,
            strategy=self.strategy,
            anchor=self.anchor,
            max_iter=self.max_iter,
            max_iter_tvgl=self.max_iter_tvgl,
            tol_flipflop=self.tol_flipflop,
            verbose=self.verbose,
            psi=self.psi,
            init_thetas=init_thetas,
            early_stopping=True
        )
        Thetas_seq = out

        return Thetas_seq  # list[W][m] -> (d_m,d_m)

    def fit_stream(self, X_seq) -> List[List[np.ndarray]]:
        """
        Returns
        -------
        latest_thetas_per_mode : list[step][m] -> ndarray (d_m, d_m)
        """
        X_seq = _normalize_X_seq(X_seq, n_modes=len(self.lambdas))
        self.dims = _infer_dims(X_seq, len(self.lambdas))
        T_total = _time_len(X_seq)
        W = self.window_size
        s = self.step_size
        assert T_total >= W, f"T(={T_total}) >= window_size(={W}) must hold."

        self.latest_thetas_per_mode_.clear()
        self.history_.clear()
        self._prev_thetas_seq = None  # reset

        # init window
        start = 0
        print("Processing window: ", str(start), "~", str(start + W - 1))
        Xw = _slice_time(X_seq, start, start + W)
        Thetas_seq = self._solve_window(Xw, is_init=True) # list[W][m]
        self._prev_thetas_seq = Thetas_seq
        self.latest_thetas_per_mode_.append([Thetas_seq[-1][m].copy()
                                             for m in range(len(self.dims))])
        # self.history_.append([[Th.copy() for Th in Thetas_seq_t] for Thetas_seq_t in Thetas_seq])

        # sliding windows
        for start in range(s, T_total - W + 1, s):
            print("--------------------------------")
            print("Processing window: ", str(start), "~", str(start + W - 1))
            start_time = time.time()

            Xw = _slice_time(X_seq, start, start + W)
            Thetas_seq = self._solve_window(Xw)
            self._prev_thetas_seq = Thetas_seq
            self.latest_thetas_per_mode_.append([Thetas_seq[-1][m].copy()
                                                 for m in range(len(self.dims))])
            self.first_thetas_per_mode_.append([Thetas_seq[0][m].copy()
                                                 for m in range(len(self.dims))])
            
            self.history_.append([[Th.copy() for Th in Thetas_seq_t] for Thetas_seq_t in Thetas_seq])

            self.runtimes.append(time.time() - start_time)
            print("Elapsed time (s):", str(self.runtimes[-1]))

        return self.first_thetas_per_mode_, self.latest_thetas_per_mode_

    def stream_steps(self, X_seq: np.ndarray) -> Iterator[List[np.ndarray]]:
        latest = self.fit_stream(X_seq)
        for thetas_tail in latest:
            yield thetas_tail

    def reset(self):
        self._prev_thetas_seq = None
        self.latest_thetas_per_mode_.clear()
        self.history_.clear()
