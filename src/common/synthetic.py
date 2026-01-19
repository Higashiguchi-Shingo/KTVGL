import numpy as np
from typing import Sequence, List, Optional, Tuple


def sym(A): return 0.5*(A + A.T)

def project_trace(A, target):
    return A * (target / max(np.trace(A), 1e-12))

def mode_product(X, A, mode):
    Xm = np.moveaxis(X, mode, 0)                 # (d_m, ...)
    Y = (A @ Xm.reshape(Xm.shape[0], -1)).reshape(Xm.shape)
    return np.moveaxis(Y, 0, mode)

def make_sparse_spd(d, p_edge=0.05, w_range=(-3.0, 3.0), seed=None):
    """
    Erdős-Rényi
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d))
    mask = rng.uniform(size=(d, d)) < p_edge
    mask = np.triu(mask, 1); mask = mask | mask.T
    w = rng.uniform(w_range[0], w_range[1], size=(d, d))
    sign = rng.choice([-1.0, 1.0], size=(d, d))
    A[mask] = (w*sign)[mask]
    A = sym(A)
    diag = np.sum(np.abs(A), axis=1) + 0.5
    Theta = -A
    np.fill_diagonal(Theta, diag)
    Theta = sym(Theta)
    #Theta = project_trace(Theta, d)  # gauge
    return Theta

def piecewise_thetas(d, breakpoints: Sequence[int], seeds: Optional[Sequence[Optional[int]]] = None, **kwargs):
    """
    breakpoints: ex. [0, 50, 100] -> Generate two for the intervals [0,49] and [50,99]
    """
    K = len(breakpoints) - 1
    if seeds is None:
        seeds = [None] * K
    assert len(seeds) == K, "length of seeds should be (len(breakpoints)-1)."
    thetas = []
    for i in range(K):
        th = make_sparse_spd(d, seed=seeds[i], **kwargs)
        thetas.append(th)
    return thetas

def expand_over_time(thetas_segments: Sequence[np.ndarray], breakpoints: Sequence[int], T: int):
    out = []
    for i in range(len(breakpoints)-1):
        a, b = breakpoints[i], breakpoints[i+1]
        out += [thetas_segments[i]] * (b - a)
    assert len(out) == T, f"expected T={T}, actual={len(out)}"
    return out

def sample_kron_gaussian(T: int, dims: Sequence[int], thetas_time: List[List[np.ndarray]], seed=None):
    """
    thetas_time[t] = [Theta_t^(0), ..., Theta_t^(n-1)] is received,
    and X_seq[t] (shape: dims) is generated one sample at a time.
    Supports any number of modes n = len(dims).
    """
    rng = np.random.default_rng(seed)
    n = len(dims)
    X_seq = np.empty((T, *dims))
    for t in range(T):
        X = rng.standard_normal(size=tuple(dims))
        # Cholesky（Theta = L L^T）→ Sigma^{1/2} = L^{-T}
        for m in range(n):
            L = np.linalg.cholesky(thetas_time[t][m])
            L_inv_T = np.linalg.inv(L).T
            X = mode_product(X, L_inv_T, mode=m)
        X_seq[t] = X
    return X_seq

def small_to_zero(Thetas_time: List[List[np.ndarray]], tol: float = 1e-6):
    for t in range(len(Thetas_time)):
        for m in range(len(Thetas_time[t])):
            M = Thetas_time[t][m]
            M[np.abs(M) < tol] = 0.0
    return Thetas_time


def generate_kron_data(
    dims: Sequence[int],
    breaks_list: Sequence[Sequence[int]],
    seeds_list: Optional[Sequence[Sequence[Optional[int]]]] = None,
    T: Optional[int] = None,
    *,
    p_edge: float = 0.25,
    w_range: Tuple[float, float] = (-3.0, 3.0),
    tol: float = 0.1,
    gaussian_seed: Optional[int] = 42
):
    """
    For an arbitrary number of modes N, create a time series of piecewise constant mode accuracy matrices,
    and generate a tensor time series from a Gaussian with a Kronecker structure following this.

    Args:
        dims: Dimensions of each mode (length N) Example: [d1, d2, d3]
        breaks_list: List of breakpoint sequences for each mode (length N)
            Example: [[0,40,70,100], [0,50,100], [0,60,100]]
        seeds_list: List of seed sequences for each interval of each mode (length N, each element length K_m=breaks[m]-1)
                    If None, all intervals are None
        T: Total time steps. If omitted, it is automatically determined assuming all modes have the same breaks[-1]
        p_edge, w_range: Parameters for the distribution passed to `make_sparse_spd`
        tol: Threshold for `small_to_zero`
        gaussian_seed: Random seed for observation generation

    Returns:
        X_seq: shape = (T, *dims)
        thetas_time: list (length T) of list(length N) of np.ndarray(d_m, d_m)
    """
    dims = list(dims)
    N = len(dims)
    assert len(breaks_list) == N, "Length of breaks_list must match number of modes N."

    ends = [bl[-1] for bl in breaks_list]
    if T is None:
        assert all(e == ends[0] for e in ends), "If T is omitted, all modes must have the same breaks[-1]."
        T = ends[0]
    else:
        assert all(e == T for e in ends), "T and breaks[-1] must match."

    if seeds_list is None:
        seeds_list = [ [None] * (len(breaks_list[m]) - 1) for m in range(N) ]
    else:
        assert len(seeds_list) == N, "Length of seeds_list must match number of modes N."
        for m in range(N):
            assert len(seeds_list[m]) == len(breaks_list[m]) - 1, f"Length of seeds_list[{m}] must match number of intervals."

    Theta_time_per_mode: List[List[np.ndarray]] = []
    for m in range(N):
        d_m = dims[m]
        breaks_m = breaks_list[m]
        seeds_m  = seeds_list[m]
        segs_m = piecewise_thetas(
            d=d_m,
            breakpoints=breaks_m,
            seeds=seeds_m,
            p_edge=p_edge,
            w_range=w_range
        )
        Theta_time_m = expand_over_time(segs_m, breaks_m, T)
        Theta_time_per_mode.append(Theta_time_m)

    thetas_time = [ [sym(Theta_time_per_mode[m][t]) for m in range(N)] for t in range(T) ]
    thetas_time = small_to_zero(thetas_time, tol=tol)

    # generate observations
    X_seq = sample_kron_gaussian(T, tuple(dims), thetas_time, seed=gaussian_seed)
    return X_seq, thetas_time