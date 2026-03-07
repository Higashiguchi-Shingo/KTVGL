import numpy as np
import os, sys
#sys.path.append(os.path.abspath('..'))
from ktvgl.methods.kronecker_gl import KroneckerGL
from sklearn.covariance import graphical_lasso


def _sym(A):
    return 0.5 * (A + A.T)

def _shrink_offdiag_preserve_diag(S, factor=0.95):
    S2 = (factor) * S.copy()
    np.fill_diagonal(S2, np.diag(S))
    return _sym(S2)

def _init_from_empirical(
    X,
    n_modes=None,
    factor=0.95,
    ridge_rel=1e-6
):
    """
    From the observation tensor X (d1,...,dM) at time 1,
    Compute S^(m) for each mode m  ->  Reduce by off-diagonal elements by 0.95 times 
      ->  Create initial values for Θ^(m) via (pseudo)inverse.

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
    def _samples_from_timestep(X_t, n_modes_):
        if isinstance(X_t, np.ndarray):
            if n_modes_ is None:
                if X_t.ndim >= 3:
                    return [X_t[n] for n in range(X_t.shape[0])]
                if X_t.ndim >= 2:
                    return [X_t]
                raise ValueError("Each sample must have ndim >= 2.")
            if X_t.ndim == n_modes_:
                return [X_t]
            if X_t.ndim == n_modes_ + 1:
                return [X_t[n] for n in range(X_t.shape[0])]
            raise ValueError(
                f"Invalid timestep ndim={X_t.ndim}; expected {n_modes_} or {n_modes_ + 1}."
            )

        if isinstance(X_t, (list, tuple)):
            if len(X_t) == 0:
                raise ValueError("Each timestep must contain at least one sample.")
            samples = [np.asarray(x) for x in X_t]
            if n_modes_ is not None:
                for s in samples:
                    if s.ndim != n_modes_:
                        raise ValueError(
                            f"Invalid sample ndim={s.ndim}; expected {n_modes_}."
                        )
            return samples

        raise ValueError("X must be ndarray or list/tuple of ndarrays.")

    samples = _samples_from_timestep(X, n_modes)
    if len(samples) == 0:
        raise ValueError("X must contain at least one sample.")

    dims = samples[0].shape
    M = len(dims)
    P = int(np.prod(dims))
    N_t = len(samples)

    thetas = []
    for m, d_m in enumerate(dims):
        D_minus = P // d_m
        S_m = np.zeros((d_m, d_m), dtype=float)
        for Xn in samples:
            if Xn.shape != dims:
                raise ValueError(
                    f"Sample shape mismatch: expected {dims}, got {Xn.shape}."
                )
            Xm = np.moveaxis(Xn, m, 0).reshape(d_m, D_minus) # mode-m unfolding: (d_m, D_-m)
            S_m += (Xm @ Xm.T) / (D_minus * N_t) # average over samples in timestep
        S_m = _sym(S_m)

        S_shr = _shrink_offdiag_preserve_diag(S_m, factor=factor)
        eps = ridge_rel * (np.trace(S_shr) / d_m)
        S_tilde = S_shr + eps * np.eye(d_m)

        Theta_m = np.linalg.pinv(S_tilde, hermitian=True)
        Theta_m = _sym(Theta_m)

        tr = np.trace(Theta_m)
        if tr != 0:
            Theta_m *= (d_m / tr)

        thetas.append(Theta_m)

    return thetas


def _normalize_X_seq(X_seq, n_modes: int):
    """
    Normalize input to list[t] of ndarray with shape (N_t, d1, ..., dM).
    """
    if isinstance(X_seq, np.ndarray):
        if X_seq.ndim == n_modes + 1:   # (T, d1, ..., dM)
            return [X_seq[t][None, ...] for t in range(X_seq.shape[0])]
        if X_seq.ndim == n_modes + 2:   # (T, N, d1, ..., dM)
            return [X_seq[t] for t in range(X_seq.shape[0])]
        raise ValueError(
            f"Invalid X_seq ndarray ndim={X_seq.ndim}; expected {n_modes + 1} or {n_modes + 2}."
        )

    if isinstance(X_seq, (list, tuple)):
        if len(X_seq) == 0:
            raise ValueError("X_seq must be non-empty.")
        norm = []
        for t, X_t in enumerate(X_seq):
            if isinstance(X_t, np.ndarray):
                if X_t.ndim == n_modes:       # (d1, ..., dM)
                    Xt = X_t[None, ...]
                elif X_t.ndim == n_modes + 1: # (N_t, d1, ..., dM)
                    Xt = X_t
                else:
                    raise ValueError(
                        f"Invalid X_seq[{t}] ndim={X_t.ndim}; expected {n_modes} or {n_modes + 1}."
                    )
            elif isinstance(X_t, (list, tuple)):
                if len(X_t) == 0:
                    raise ValueError(f"X_seq[{t}] must contain at least one sample.")
                samples = [np.asarray(x) for x in X_t]
                for s in samples:
                    if s.ndim != n_modes:
                        raise ValueError(
                            f"Invalid sample ndim={s.ndim} at t={t}; expected {n_modes}."
                        )
                Xt = np.stack(samples, axis=0)
            else:
                raise ValueError("Each X_seq[t] must be ndarray or list/tuple of ndarrays.")
            norm.append(Xt)
        return norm

    raise ValueError("X_seq must be ndarray or list/tuple.")


def init_precision(X_seq, method="identity", lambdas=None, n_modes=None):
    """
    Initialize the precision matrices (Thetas) for each time slice.
    Input:
        X_seq: np.ndarray, shape (T, d1, ..., dM)
    Output:
        Thetas_seq: List[List[np.ndarray]]
    """
    if n_modes is None:
        if lambdas is None:
            raise ValueError("n_modes or lambdas must be provided.")
        n_modes = len(lambdas)

    X_norm = _normalize_X_seq(X_seq, n_modes=n_modes)
    T = len(X_norm)
    dims = X_norm[0].shape[1:]
    M = len(dims)

    if method == "identity":
        Thetas_seq = [[np.eye(dims[m]) for m in range(M)] for _ in range(T)]
    
    elif method == "empirical":
        Thetas_seq = [[np.eye(dims[m]) for m in range(M)] for _ in range(T)]
        for t in range(T):
            Thetas_seq[t] = _init_from_empirical(X_norm[t], n_modes=M)
    
    elif method == "Glasso":
        # kgl = KroneckerGL(
        #     lambdas=lambdas,
        #     gl_solver=graphical_lasso
        # )
        # Thetas = kgl.fit(X_seq)
        # Thetas_seq = [Thetas for _ in range(T)]
        raise NotImplementedError("init_method='Glasso' is temporarily disabled.")
    return Thetas_seq
