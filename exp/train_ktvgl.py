import sys, os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json, pickle
import time

sys.path.append(os.path.abspath('../src'))
from methods.KroneckerTVGL import KroneckerTVGL, time_varying_graphical_lasso
from common.utils import load_tensor, kronecker_multiple, hp_to_dirpath
from common.synthetic import generate_kron_data
from common.metric import ktvgl_eval, tvgl_eval
from common.viz import viz_change_of_theta, viz_change_of_theta_flatten, viz_error_of_theta, viz_error_of_theta_flatten


def parse_args():
    parser = argparse.ArgumentParser()
    # I/O setting
    parser.add_argument('--base-dir', type=str, default="../exp_results", help='path to base directory')
    # model setting
    parser.add_argument('--init', type=str, default='empirical', choices=['identity', 'empirical', "Glasso"])
    parser.add_argument('--lambdas', type=float, nargs="+", default=[0.01,0.01], help='sparsity regularization parameter')
    parser.add_argument('--rhos', type=float, nargs="+", default=[2, 2], help='temporal regularization parameter')
    parser.add_argument('--max_iter', type=int, default=100, help='maximum number of iterations')
    parser.add_argument('--max_iter_tvgl', type=int, default=500, help='maximum number of iterations for TVGL')
    parser.add_argument('--tol_flipflop', type=float, default=1e-4, help='tolerance for convergence')
    parser.add_argument('--psi', type=str, default="laplacian", choices=["laplacian", "l1"], help='temporal penalty type')
    parser.add_argument('--verbose', action='store_true', help='whether to print progress')
    parser.add_argument('--gauge', type=str, default='trace', choices=['trace', 'det', 'fro'], help='gauge normalization method')
    # data setting
    parser.add_argument('--dims', type=int, nargs='+', required=True, help='dimensions of each mode. e.g. --dims 20 30 40')
    parser.add_argument('--T', type=int, default=None, help='total time steps. If omitted, all modes must have the same breaks[-1]')
    # JSON settings
    parser.add_argument('--breaks-json', type=json.loads, required=True, help="ex) '[[0,40,70,100],[0,50,100]]'")
    parser.add_argument('--seeds-json', type=json.loads, default=None, help="ex) '[[1,12,43],[44,25]]' or None")
    parser.add_argument('--p-edge', type=float, default=0.25, help='parameter of make_sparse_spd')
    parser.add_argument('--wrange', type=float, nargs=2, default=(-3.0, 3.0), help='parameter of make_sparse_spd')
    args = parser.parse_args()

    # breaks and seeds processing
    N = len(args.dims)
    breaks_list = args.breaks_json
    if not isinstance(breaks_list, list) or any(not isinstance(b, list) for b in breaks_list):
        raise ValueError("--breaks-json must be a two dimensional list.")

    # Broadcast (if you want to share a single list across all modes)
    if len(breaks_list) == 1 and N > 1:
        breaks_list = [list(breaks_list[0]) for _ in range(N)]

    if len(breaks_list) != N:
        raise ValueError(f"The number of breaks_list ({len(breaks_list)}) must match the number of modes ({N}).")

    # seeds
    if args.seeds_json is None:
        seeds_list = [[None]*(len(b)-1) for b in breaks_list]
    else:
        seeds_list = args.seeds_json
        if len(seeds_list) == 1 and N > 1:
            seeds_list = [list(seeds_list[0]) for _ in range(N)]
        if len(seeds_list) != N:
            raise ValueError(f"The number of seeds_list ({len(seeds_list)}) must match the number of modes ({N}).")
        for m in range(N):
            if len(seeds_list[m]) != len(breaks_list[m]) - 1:
                raise ValueError(f"The length of seeds_list[{m}] ({len(seeds_list[m])}) must match breaks[{m}]-1 ({len(breaks_list[m])-1}).")

    # T consistency check
    ends = [b[-1] for b in breaks_list]
    if args.T is None:
        if not all(e == ends[0] for e in ends):
            raise ValueError("When T is omitted, all modes must have the same breaks[-1].")
        T = ends[0]
    else:
        T = args.T
        if not all(e == T for e in ends):
            raise ValueError("The provided T must match each mode's breaks[-1].")

    # Ascending order and boundary check
    for m, br in enumerate(breaks_list):
        if br[0] != 0 or any(br[i] >= br[i+1] for i in range(len(br)-1)) or br[-1] != T:
            raise ValueError(f"breaks_list[{m}] must be in ascending order [0, ..., T] and end with T.")

    args.breaks_list = breaks_list
    args.seeds_list = seeds_list
    args.T = T
    return args


def main():
    args = parse_args()

    # Load or generate data
    X, Thetas_true = generate_kron_data(
        dims=args.dims, 
        breaks_list=args.breaks_list, 
        T=args.T,
        p_edge=args.p_edge,
        w_range=tuple(args.wrange)
    )

    # Settings
    hp = {
        'lambdas': args.lambdas,
        'rhos': args.rhos,
        'max_iter': args.max_iter,
        'max_iter_tvgl': args.max_iter_tvgl,
        'tol_flipflop': args.tol_flipflop,
        'psi': args.psi,
        'gauge': args.gauge,
        'verbose': args.verbose,
        'init_method': args.init,
        'w_range': tuple(args.wrange),
        'p_edge': args.p_edge,
        'dims': args.dims,
        'breaks_list': args.breaks_list,
        'seeds_list': args.seeds_list,
        'T': args.T
    }

    # model
    ktvgl = KroneckerTVGL(
        lambdas=hp['lambdas'], 
        rhos=hp['rhos'], 
        tvgl_solver=time_varying_graphical_lasso,
        init_method=hp['init_method'],
        max_iter=hp['max_iter'], 
        max_iter_tvgl=hp['max_iter_tvgl'],
        tol_flipflop=hp['tol_flipflop'],
        psi=hp['psi'],
        gauge=hp['gauge'],
        verbose=args.verbose
    )

    # Training
    start_time = time.time()
    Thetas_hat = ktvgl.fit(X,)
    run_time = time.time() - start_time
    print(f"Run time: {run_time:.2f} seconds")

    # Evaluation
    results = ktvgl_eval(Thetas_hat, Thetas_true)
    print("Results (KTVGL):", results)
    
    """Thetas_hat_flatten = []
    Thetas_true_flatten = []
    for t in range(args.T):
        Thetas_hat_flatten.append(kronecker_multiple(Thetas_hat[t]))
        Thetas_true_flatten.append(kronecker_multiple(Thetas_true[t]))
    Thetas_hat_flatten = np.array(Thetas_hat_flatten)
    Thetas_true_flatten = np.array(Thetas_true_flatten)
    print("Flatten.")
    results_flatten = tvgl_eval(Thetas_hat_flatten, Thetas_true_flatten)
    print("Results (flattened):", results_flatten)"""

    # Save results
    save_path = hp_to_dirpath(
        hp, 
        base_dir=args.base_dir, 
        method="KTVGL", 
        dataset="synthetic", 
        include_keys=["dims", "breaks_list", "seeds_list", "lambdas", "rhos", "init_method", "gauge"],
        mkdir=True
    )
    """with open(os.path.join(save_path, 'ktvgl_acc.json'), 'w') as f:
        json.dump({"Accuracy": results, "Accuracy_flatten": results_flatten, "Runtime": run_time}, f)"""
    with open(os.path.join(save_path, 'ktvgl_precision.pkl'), "wb") as f:
        pickle.dump({"Thetas_hat": Thetas_hat, "Thetas_true": Thetas_true}, f, protocol=pickle.HIGHEST_PROTOCOL)
    """with open(os.path.join(save_path, 'ktvgl_precision_flatten.pkl'), "wb") as f:
        pickle.dump({"Thetas_hat": Thetas_hat_flatten, "Thetas_true": Thetas_true_flatten}, f, protocol=pickle.HIGHEST_PROTOCOL)"""

    # Visualization
    viz_change_of_theta(Thetas_hat, save_dir=save_path)
    #viz_change_of_theta_flatten(Thetas_hat_flatten, save_dir=save_path)
    viz_error_of_theta(Thetas_hat, Thetas_true, save_dir=save_path)
    #viz_error_of_theta_flatten(Thetas_hat_flatten, Thetas_true_flatten, save_dir=save_path)


if __name__ == "__main__":
    main()