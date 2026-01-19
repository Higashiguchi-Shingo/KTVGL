import numpy as np
import os, sys
#sys.path.append(os.path.abspath('..'))
from methods.KroneckerGL import KroneckerGL
from sklearn.covariance import graphical_lasso


def _sym(A):
    return 0.5 * (A + A.T)

def _shrink_offdiag_preserve_diag(S, factor=0.95):
    S2 = (factor) * S.copy()
    np.fill_diagonal(S2, np.diag(S))
    return _sym(S2)

def _init_from_empirical(
    X,
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
    X = np.asarray(X)
    dims = X.shape
    M = len(dims)
    P = int(np.prod(dims))

    thetas = []
    for m, d_m in enumerate(dims):
        D_minus = P // d_m
        Xm = np.moveaxis(X, m, 0).reshape(d_m, D_minus) # mode-m unfolding: (d_m, D_-m)

        S_m = (Xm @ Xm.T) / D_minus # S^(m) = (1/D_-m) * Xm Xm^T
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


def init_precision(X_seq, method="identity", lambdas=None):
    """
    Initialize the precision matrices (Thetas) for each time slice.
    Input:
        X_seq: np.ndarray, shape (T, d1, ..., dM)
    Output:
        Thetas_seq: List[List[np.ndarray]]
    """
    T = X_seq.shape[0]
    dims = X_seq.shape[1:]

    if method == "identity":
        Thetas_seq = [[np.eye(dims[m]) for m in range(len(dims))] for _ in range(T)]
    
    elif method == "empirical":
        Thetas_seq = [[np.eye(dims[m]) for m in range(len(dims))] for _ in range(T)]
        for t in range(T):
            Thetas_seq[t] = _init_from_empirical(X_seq[t])
    
    elif method == "Glasso":
        kgl = KroneckerGL(
            lambdas=lambdas,
            gl_solver=graphical_lasso
        )
        Thetas = kgl.fit(X_seq)
        Thetas_seq = [Thetas for _ in range(T)]
    return Thetas_seq