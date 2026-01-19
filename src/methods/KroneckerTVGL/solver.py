import numpy as np
import collections
import warnings

def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, collections.abc.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

convergence = namedtuple_with_defaults(
    'convergence', 'obj rnorm snorm e_pri e_dual precision')


def check_norm_prox(function):
    """Validate function and return norm with associated prox."""
    if function == "laplacian":
        prox = prox_laplacian
        norm = squared_norm
    elif function == "l1":
        prox = soft_thresholding
        norm = l1_norm
    else:
        raise ValueError("Value of %s not understood." % function)
    return norm, prox


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
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn('Array type is integer, np.dot may overflow. '
                      'Data should be float type to avoid this issue',
                      UserWarning)
    return np.dot(x, x)

def off_diag_frobenius_square(X: np.ndarray) -> float:
    ret = 0
    T = X.shape[0]
    for t in range(T):
        ret += np.sum(X[t]**2) - np.sum(np.diag(X[t])**2)
    return ret

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
    return logdet(precision) - np.trace(emp_cov @ precision)
    #return logdet(precision) - np.sum(emp_cov * precision)

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

def objective(n_samples, S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)

    if isinstance(alpha, np.ndarray):
        obj += sum(l1_od_norm(a * z) for a, z in zip(alpha, Z_0))
    else:
        obj += alpha * sum(map(l1_od_norm, Z_0))

    if isinstance(beta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(beta, map(psi, Z_2 - Z_1)))
    else:
        obj += beta * sum(map(psi, Z_2 - Z_1))

    return obj

def prox_logdet(a, lamda):
    """
    Time-varying latent variable graphical lasso prox.
    Eq. (5) in the paper.
    """
    D, Q = np.linalg.eigh(a)
    xi = (-D + np.sqrt(np.square(D) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))

def soft_thresholding(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)

def prox_laplacian(a, lamda):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return a / (1 + 2. * lamda)

def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """
    Parameters
    ----------
    rho : float
        Augmented Lagrangian parameter.
    rnorm : float
        Residual norm.
    snorm : float
        Dual norm.
    iteration : int, optional
        Iteration number. Default is None.
    mu : float, optional
        Parameter for updating rho. Default is 10.
    tau_inc : float, optional
        Increment factor for rho. Default is 2.
    tau_dec : float, optional
        Decrement factor for rho. Default is 2.

    Returns
    -------
    rho : float
        Updated rho.
    """
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho

def init_precision(emp_cov, mode='empirical'):
    if isinstance(mode, np.ndarray):
        return mode.copy()

    if mode == 'empirical':
        emp_cov = np.asarray(emp_cov)
        n_times, _, n_features = emp_cov.shape
        covariance_ = emp_cov.copy()
        covariance_ *= 0.95

        K = np.empty_like(emp_cov)
        for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
            c.flat[::n_features + 1] = e.flat[::n_features + 1] # diagonal
            K[i] = np.linalg.pinv(c, hermitian=True)

    elif mode == 'zeros':
        K = np.zeros_like(emp_cov)

    return K



def time_varying_graphical_lasso(
    emp_cov: np.ndarray, # shape: (n_time, n_features, n_features) = (T, dm, dm)
    prev_matrices: np.ndarray = None, # shape: (n_time, n_features, n_features) = (T, dm, dm)
    prev_dual_vars: np.ndarray = None, # shape: (n_time, n_features, n_features) = (T, dm, dm)
    alpha=0.01,            
    rho=1, 
    beta=1, 
    max_iter=300,
    verbose=False, 
    tol=1e-4, 
    rtol=1e-4,
    compute_objective=True, 
    stop_at=None, 
    stop_when=1e-4,
    init='empirical', 
    init_inv_cov=None, # shape: (n_features, n_features)
    psi="laplacian" # "laplacian" or "l1"
):
    """
    Time-Varying Graphical Lasso (TVGL) for time-varying covariance estimation.

    Parameters
    ----------
    emp_cov : array-like, shape (n_time, n_features, n_features) = (T, Dn, Dn)
        Empirical covariance matrix.
    prev_matrices : array-like, shape (n_time, n_features, n_features) = (T, Dn, Dn), optional
        Previous precision matrices.
    prev_dual_vars : array-like, shape (n_time, n_features, n_features) = (T, Dn, Dn), optional
        Previous dual variables.
    alpha : float, optional
        L1 penalty for the precision matrix. Default is 0.01.
        lambda in the paper.
    rho : float, optional
        Augmented Lagrangian parameter. Default is 1.
    beta : float, optional
        L2 penalty for the precision matrix smoothness. Default is 1.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    verbose : bool, optional
        If True, print convergence information. Default is False.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    rtol : float, optional
        Relative tolerance for convergence. Default is 1e-4.
    compute_objective : bool, optional
        If True, compute the objective function. Default is True.
    stop_at : float, optional
        If not None, stop the algorithm when the objective function is close to this value. Default is None.
    stop_when : float, optional
        If not None, stop the algorithm when the relative change in the objective function is less than this value. Default is 1e-4.
    init : str, optional
        Initialization method for the precision matrix. Default is 'empirical'.
    init_inv_cov : array-like, shape (n_features, n_features), optional
        Initial inverse covariance matrix. Default is None.

    Returns
    -------
    Z_0 : array-like, shape (n_time, n_features, n_features) = (T, Dn, Dn)
        Estimated precision matrix.
    covariance_ : array-like, shape (n_samples, n_features, n_features)
        Estimated covariance matrix.
    checks : list, optional
        List of convergence checks. Only returned if return_history is True.
    n_iter : int, optional
        Number of iterations. Only returned if return_n_iter is True.
    """
    
    # psi, prox_psi, psi_node_penalty = check_norm_prox(psi)
    psi, prox_psi = check_norm_prox(psi)

    # initialize precision matrix
    if prev_matrices is not None:
        Z_0 = np.array(prev_matrices.copy())
    else:
        Z_0 = init_precision(emp_cov, mode=init)

    if isinstance(init_inv_cov, np.ndarray):
        Z_0[0,:,:]=init_inv_cov
        #print("(Inner loop) Initial precision matrix set.")

    # initialize Z1 and Z2
    Z_1 = Z_0.copy()[:-1]
    Z_2 = Z_0.copy()[1:]

    Z_0_old = np.zeros_like(Z_0) 
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # initialize dual variables U0, U1, and U2
    if prev_dual_vars is not None:
        U_0 = prev_dual_vars.copy()
        U_1 = prev_dual_vars.copy()[:-1]
        U_2 = prev_dual_vars.copy()[1:]
    else:
        U_0 = np.zeros_like(Z_0)
        U_1 = np.zeros_like(Z_1)
        U_2 = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    n_samples = np.ones(emp_cov.shape[0]) # window size

    checks = [
        convergence(
            obj=objective(
                n_samples, emp_cov, Z_0, Z_0, Z_1, Z_2, alpha, beta, psi))
    ]

    # Note:  K indicates "Theta" in the paper
    for iteration_ in range(max_iter):
        # ---------- update K ----------
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        A += A.transpose(0, 2, 1)
        A /= 2.

        A *= -rho * divisor[:, None, None] / n_samples[:, None, None]
        A += emp_cov # Why Plus?? minus in the paper -> paper is wrong.

        K = np.array(
            [
                prox_logdet(a, lamda=ni / (rho * div))
                for a, div, ni in zip(A, divisor, n_samples)
            ]) # Eq.(5) in the paper

        if isinstance(init_inv_cov, np.ndarray):
            K[0,:,:]=init_inv_cov

        # ---------- update Z_0 ----------
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # ---------- update Z1 and Z2 ----------
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)

        Z_1 = .5 * (A_1 + A_2 - prox_e)
        Z_2 = .5 * (A_1 + A_2 + prox_e)

        if isinstance(init_inv_cov, np.ndarray):
            Z_0[0,:,:]=init_inv_cov
            Z_1[0,:,:]=init_inv_cov

        # ---------- update dual variables (U) ----------
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2
        
        rnorm = np.sqrt(
            off_diag_frobenius_square(K - Z_0) + off_diag_frobenius_square(K[:-1] - Z_1) + off_diag_frobenius_square(K[1:] - Z_2)
        )

        snorm = rho * np.sqrt(
            off_diag_frobenius_square(Z_0 - Z_0_old) + off_diag_frobenius_square(Z_1 - Z_1_old) + off_diag_frobenius_square(Z_2 - Z_2_old)
        )

        obj = objective(
            n_samples, emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi) \
            if compute_objective else np.nan
        
        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
                np.sqrt(
                    off_diag_frobenius_square(Z_0) + off_diag_frobenius_square(Z_1) + off_diag_frobenius_square(Z_2)),
                np.sqrt(
                    off_diag_frobenius_square(K) + off_diag_frobenius_square(K[:-1]) +
                    off_diag_frobenius_square(K[1:]))
                ),
            e_dual=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho *
            np.sqrt(off_diag_frobenius_square(U_0) + off_diag_frobenius_square(U_1) + off_diag_frobenius_square(U_2)),
            # precision=Z_0.copy()
        )

        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % check[:5]
            )

        checks.append(check)
        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                print("Early stopping criterion reached.", iteration_)
                break

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            if verbose:
                print("(Inner loop) Converged at iteration %d." % iteration_)
                print(
                    "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                    "eps_pri: %.4f, eps_dual: %.4f" % check[:5]
                )
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        U_0 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new

    else:
        print("(Inner loop) Objective did not converge.")
        if verbose:
            print(
                    "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                    "eps_pri: %.4f, eps_dual: %.4f" % check[:5]
            )
        #warnings.warn("Objective did not converge.")

    covariance_ = np.array([np.linalg.pinv(x, hermitian=True) for x in Z_0])
    return_list = [Z_0, covariance_]
    return_list.append(checks)
    return_list.append(iteration_ + 1)

    return Z_0
