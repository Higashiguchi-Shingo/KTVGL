from copy import deepcopy
import numpy as np
from numpy.linalg import slogdet

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

def _shrink_offdiag_preserve_diag(S, factor=0.95):
    """
    Keep the diagonal elements unchanged and scale only the off-diagonal elements by a factor.
    S_shr = factor*S + (1-factor)*diag(S)
    """
    S2 = (factor) * S.copy()
    np.fill_diagonal(S2, np.diag(S))
    return sym(S2)

def init_precision(
    X_seq,
    factor=0.95,
    ridge_rel=1e-6,
    method="empirical"
):
    """
    Parameters
    ----------
    X : ndarray, shape (d1,...,dM)
    factor : float, default 0.95
    ridge_rel : float, default 1e-6
    gauge : {"trace", None}, default "trace"
    return_covariances : bool, default False

    Returns
    -------
    thetas : list of ndarray
    (optional) S_list, S_shr_list : list of ndarray
    """
    X = np.asarray(X_seq)
    T = X.shape[0]
    dims = X.shape[1:]
    M = len(dims)
    P = int(np.prod(dims))

    thetas = []
    if method == "empirical":
        for m, d_m in enumerate(dims):
            D_minus = P // d_m

            S_m = np.zeros((d_m, d_m), dtype=float)
            for t in range(T):
                Xt = X[t]
                Xmt = np.moveaxis(Xt, m, 0).reshape(d_m, D_minus)
                S_m += (Xmt @ Xmt.T) / (D_minus * T)
            S_m = sym(S_m)

            S_shr = _shrink_offdiag_preserve_diag(S_m, factor=factor)
            eps = ridge_rel * (np.trace(S_shr) / d_m)
            S_tilde = S_shr + eps * np.eye(d_m)

            Theta_m = np.linalg.pinv(S_tilde, hermitian=True)
            Theta_m = sym(Theta_m)

            tr = np.trace(Theta_m)
            if tr != 0:
                Theta_m *= (d_m / tr)

            thetas.append(Theta_m)
    
    elif method == "identity":
        for m, d_m in enumerate(dims):
            Theta_m = np.eye(d_m)
            thetas.append(Theta_m)

    return thetas


def compute_Shat(
    X_seq: np.ndarray,
    Thetas: list[np.ndarray],
    mode: int,
    jitter_scale: float = 1e-8
) -> np.ndarray:
    """
    X_seq: array (T, d1, ..., dM)
    Thetas_seq: list: (M, dm, dm), Thetas_seq[t] is list [Θ_t^(1),...,Θ_t^(M)]
    mode: target mode

    Return: S_hats (T, d_mode, d_mode), S_t^(m) for each timestep
    """
    T = X_seq.shape[0]
    dims = X_seq.shape[1:]  # (d1,...,dM)
    M = len(dims)
    assert 0 <= mode < M, "mode index out of range"
    d_m = dims[mode]
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))

    Shat = np.zeros((d_m, d_m), dtype=float)

    for t in range(T):
        X_t = X_seq[t]

        Y = X_t
        for ell in range(M):
            if ell == mode:
                continue
            Y = mode_product(Y, Thetas[ell], mode=ell)

        Ym = np.moveaxis(Y, mode, 0).reshape(d_m, D_minus)
        Xm = np.moveaxis(X_t, mode, 0).reshape(d_m, D_minus)
        Shat += (Ym @ Xm.T) / (D_minus * T)

    Shat = sym(Shat)
    eps = jitter_scale * max(1.0, np.trace(Shat) / max(d_m, 1))
    Shat += eps * np.eye(d_m)

    return Shat


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
        else:
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


def update_one_mode_glasso(
    X_seq: np.ndarray,          # shape: (T, d1, ..., dM)
    Thetas: list[np.ndarray],   # list of array: (dm, dm), length = M
    mode: int,              
    gl_solver,                # gl_solver(S_hats, lam_bar, **kwargs) -> np.ndarray (d_m, d_m)
    compute_Shat_func,        # compute_Shat(X_seq, Thetas_seq, mode) -> (T, d_m, d_m)
    *,
    lambda_m: float = 0.01,   # float: strength of L1 norm for mode m
    max_iter_gl: int = 500,   # number of iteration for TVGL
    gauge="trace",
    strategy="anchor",
    anchor="next",
    verbose=False
):
    """
    Update Thetas once for mode m.

    Returns: 
    Thetas_seq: updated precision matrices
    """
    T = X_seq.shape[0]
    dims = X_seq.shape[1:]
    M = len(dims)
    d_m = dims[mode]
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))

    # (a) Compute S^(m)
    S_hat = compute_Shat_func(
        X_seq=X_seq, 
        Thetas=Thetas, 
        mode=mode
    )   # S_hats: shape (d_m, d_m)

    # (b) Solve GL -> get Theta (i.e., precision matrices)
    prev_matrices = Thetas[mode].copy()
    lam_bar = lambda_m / D_minus
    _out = gl_solver(
        emp_cov=S_hat, 
        alpha=lam_bar,
        max_iter=max_iter_gl,
        #precision_init=prev_matrices,
    ) # (d_m, d_m)
    Theta_m = _out[1]

    # (c) Scale correction
    updated = [A.copy() for A in Thetas] # updated Thetas, shape: (M, dm, dm)
    updated[mode] = sym(Theta_m)

    corrected = scale_correct(
                    updated, 
                    mode=mode, 
                    gauge=gauge, 
                    strategy=strategy, 
                    anchor=anchor
                )
    Thetas = corrected

    return Thetas

# kronecker TVGL
def kronecker_graphical_lasso(
    X_seq: np.ndarray,                   # shape: (T, d1, ..., dn)
    Thetas: list[list[np.ndarray]],      # list: (M, dm, dm), array: (dm, dm)
    lambdas: list[float],                # list: (M, )
    gl_solver,                           # gl_solver(S_hats, lam_bar, **kwargs) -> np.ndarray (T, d_m, d_m)
    compute_Shat_func,                   # compute_Shat(X_seq, Thetas_seq, mode) -> (T, d_m, d_m)
    gauge: str = "trace",
    strategy: str = "anchor",
    anchor: str = "next",
    max_iter: int = 10,
    max_iter_gl: int = 50,
    tol_flipflop: float = 1e-4,
    verbose: bool = False
):
    pre_obj = None
    M = len(X_seq.shape[1:])

    for iter in range(max_iter):
        if verbose: print("Iteration:", str(iter))
        for mode in range(M):
            if verbose: print("mode:", str(mode))
            Thetas = update_one_mode_glasso(
                X_seq=X_seq,
                Thetas=Thetas,
                mode=mode,
                lambda_m=lambdas[mode],
                gl_solver=gl_solver,
                compute_Shat_func=compute_Shat_func,
                max_iter_gl=max_iter_gl,
                gauge=gauge,
                strategy=strategy,
                anchor=anchor,
                verbose=verbose
            )
        obj = objective_function(X_seq, Thetas, lambdas)
        if verbose: print("Score:", str(obj))

        if pre_obj is not None and ((pre_obj - obj) / np.abs(pre_obj)) < tol_flipflop:
            print("Early stopping at iteration:", str(iter))
            break
        pre_obj = obj

    return Thetas



class KroneckerGL:
    def __init__(self, 
        lambdas, 
        gl_solver, 
        init_method="empirical",
        gauge="trace", 
        strategy="equal", 
        anchor="next", 
        max_iter=10,
        max_iter_gl=500,
        tol_flipflop=1e-4,
        verbose=False
    ):
        self.lambdas = lambdas
        self.gl_solver = gl_solver
        self.compute_Shat_func = compute_Shat
        self.gauge = gauge
        self.strategy = strategy
        self.anchor = anchor
        self.max_iter = max_iter
        self.max_iter_gl = max_iter_gl
        self.init_method = init_method
        self.tol_flipflop = tol_flipflop
        self.verbose = verbose

    def fit(self, X_seq):
        self.X_seq = X_seq
        self.dims = X_seq.shape[1:]

        # Initialize Thetas_seq
        Thetas = init_precision(X_seq, method=self.init_method)

        # Run Kronecker graphical lasso
        self.Thetas_seq = kronecker_graphical_lasso(
            X_seq=X_seq,
            Thetas=Thetas,
            lambdas=self.lambdas,
            gl_solver=self.gl_solver,
            compute_Shat_func=self.compute_Shat_func,
            gauge=self.gauge,
            strategy=self.strategy,
            anchor=self.anchor,
            max_iter=self.max_iter,
            max_iter_gl=self.max_iter_gl,
            tol_flipflop=self.tol_flipflop,
            verbose=self.verbose
        )

        return self.Thetas_seq


def objective_function(X_seq, Thetas, lambdas, mode=0):
    T = X_seq.shape[0]
    dims = X_seq.shape[1:]  # (d1,...,dM)
    M = len(dims)
    assert 0 <= mode < M, "mode index out of range"

    # Compute the objective function value
    obj = 0.0
    S_hat = compute_Shat(X_seq, Thetas, mode)

    # log-likelihood terms
    D_minus = int(np.prod([dims[k] for k in range(M) if k != mode]))
    obj += D_minus * np.trace(S_hat @ Thetas[mode])
    for m in range(M):
        D_minus = int(np.prod([dims[k] for k in range(M) if k != m]))
        obj -= D_minus * logdet(Thetas[m])

    # L1 norm terms
    for m in range(M):
        obj += lambdas[m] * l1_od_norm(Thetas[m])
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
