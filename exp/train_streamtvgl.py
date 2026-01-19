import sys, os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json, pickle

sys.path.append(os.path.abspath('../src'))
from methods.TVGL import StreamTVGL
from common.utils import kronecker_multiple, hp_to_dirpath
from common.synthetic import generate_kron_data
from common.metric import tvgl_eval
from common.viz import viz_change_of_theta_flatten, viz_error_of_theta_flatten


def parse_args():
    parser = argparse.ArgumentParser()
    # I/O setting
    parser.add_argument('--base-dir', type=str, default="../exp_results", help='path to base directory')
    # model setting
    parser.add_argument('--init', type=str, default='empirical_mean', choices=['identity', 'empirical_mean', 'GLasso'])
    parser.add_argument('--alpha', type=float, default=0.05, help='sparsity regularization parameter')
    parser.add_argument('--beta', type=float, default=2.0, help='temporal regularization parameter')
    parser.add_argument('--rho', type=float, default=1.0, help='ADMM penalty parameter')
    parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of iterations')
    parser.add_argument('--psi', type=str, default="laplacian", choices=["laplacian", "l1"], help='temporal penalty type')
    parser.add_argument('--verbose', action='store_true', help='whether to print progress')
    parser.add_argument('--window-size', type=int, default=50, help='sliding window size')
    parser.add_argument('--step-size', type=int, default=1, help='stride for sliding window')
    parser.add_argument("--log", type=str, default="latest", choices=["latest", "first"])
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

    # Generate data
    X, Thetas = generate_kron_data(
        dims=args.dims, 
        breaks_list=args.breaks_list, 
        T=args.T,
        p_edge=args.p_edge,
        w_range=tuple(args.wrange)
    )
    X = X.reshape(args.T, -1)
    Thetas_true = []
    for t in range(args.T):
        Thetas_true.append(kronecker_multiple(Thetas[t]))
    Thetas_true = np.array(Thetas_true)

    # Settings
    hp = {
        'alpha': args.alpha,
        'rho': args.rho,
        'beta': args.beta,
        'max_iter': args.max_iter,
        'psi': args.psi,
        'verbose': args.verbose,
        'init_method': args.init,
        'w_range': tuple(args.wrange),
        'p_edge': args.p_edge,
        'dims': args.dims,
        'breaks_list': args.breaks_list,
        'seeds_list': args.seeds_list,
        'T': args.T,
        'window_size': args.window_size,
        'step_size': args.step_size,
        'log': args.log
    }

    # Model
    stream_tvgl = StreamTVGL(
        alpha=hp['alpha'], 
        beta=hp['beta'], 
        max_iter=hp['max_iter'], 
        tol=1e-4, 
        rtol=1e-4, 
        init=hp['init_method'], 
        psi=hp['psi'], 
        verbose=hp['verbose'],
        window_size=hp['window_size'],
        step_size=hp['step_size']
    )
    Thetas_hat_first, Thetas_hat_latest = stream_tvgl.fit_stream(X)
    if hp["log"] == "first":
        Thetas_true = Thetas_true[0 : -hp['window_size'] : hp['step_size']]
        Thetas_hat = Thetas_hat_first
    elif hp["log"] == "latest":
        Thetas_true = Thetas_true[hp['window_size']-1 :: hp['step_size']]
        Thetas_hat = Thetas_hat_latest

    # Evaluation
    runtimes = stream_tvgl.runtimes
    results = tvgl_eval(Thetas_hat, Thetas_true)
    print("Results:", results)

    # Save results
    save_path = hp_to_dirpath(
        hp, 
        base_dir=args.base_dir, 
        method="StreamTVGL", 
        dataset="synthetic", 
        include_keys=["window_size", "step_size", "dims", "breaks_list", "seeds_list", "alpha", "beta", "log"],
        mkdir=True
    )
    with open(os.path.join(save_path, 'StreamTVGL_acc.json'), 'w') as f:
        json.dump({"Accuracy": results, "Runtimes": runtimes}, f)
    with open(os.path.join(save_path, 'StreamTVGL_precision.pkl'), "wb") as f:
        pickle.dump({"Thetas_hat": Thetas_hat, "Thetas_true": Thetas_true}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Visualization
    viz_change_of_theta_flatten(Thetas_hat, save_dir=save_path)
    viz_error_of_theta_flatten(Thetas_hat, Thetas_true, save_dir=save_path)



if __name__ == '__main__':
    main()