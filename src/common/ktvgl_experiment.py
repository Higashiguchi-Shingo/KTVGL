"""CLI workflow for KTVGL; data sources share the same artifact schema."""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import secrets
import time
import warnings

import numpy as np

from .metric import ktvgl_eval
from .diagnostics import (INNER_FIELDS, InnerSummaryWriter, RunLog,
                          precision_diagnostics, summarize_inner)
from .history import HISTORY_FIELDS, HistoryWriter, plot_history, read_history
from .results import (RunArtifacts, canonical_batches, coordinates, data_hash,
                      file_hash, load_observations, save_precision)
from .viz import viz_change_of_theta, viz_error_of_theta


def parse_args(supports_multi=True):
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
    parser.add_argument('--gauge', type=str, default='none', choices=['trace', 'det', 'fro', 'none'], help="gauge normalization method (default: 'none', disables gauge normalization)")
    # data setting
    parser.add_argument('--dims', type=int, nargs='+', required=True, help='dimensions of each mode. e.g. --dims 20 30 40')
    parser.add_argument('--T', type=int, default=None, help='total time steps. If omitted, all modes must have the same breaks[-1]')
    # JSON settings
    parser.add_argument('--breaks-json', type=json.loads, help="ex) '[[0,40,70,100],[0,50,100]]'")
    parser.add_argument('--seeds-json', type=json.loads, default=None, help="ex) '[[1,12,43],[44,25]]' or None")
    parser.add_argument('--p-edge', type=float, default=0.25, help='parameter of make_sparse_spd')
    parser.add_argument('--wrange', type=float, nargs=2, default=(-3.0, 3.0), help='parameter of make_sparse_spd')
    parser.add_argument('--samples-per-t', type=int, default=1, help='number of samples per timestep (fixed)')
    parser.add_argument('--samples-per-t-json', type=json.loads, default=None, help="ex) '[2,1,3,2,...]' (variable per timestep)")
    parser.add_argument('--data-path', type=Path, help='External .npy or .npz observations; omit to generate synthetic data')
    parser.add_argument('--dataset-id', help='Stable ASCII dataset identifier')
    parser.add_argument('--dataset-name', help='Human-readable dataset name')
    parser.add_argument('--dataset-version', help='Dataset revision')
    parser.add_argument('--metadata-json', type=Path, help='Dataset coordinates and preprocessing metadata')
    parser.add_argument('--gaussian-seed', type=int, default=42)
    parser.add_argument('--min-iter', type=int, default=5)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--no-plots', action='store_true', help='Skip all saved figures (history.csv is still written)')
    args = parser.parse_args()
    if any(d < 1 for d in args.dims):
        parser.error('--dims must be positive.')
    for name in ('lambdas', 'rhos'):
        values = getattr(args, name)
        if len(values) != len(args.dims) or any(not np.isfinite(v) or v < 0 for v in values):
            parser.error(f'--{name} requires one finite nonnegative value per mode.')
    if args.max_iter < 1 or args.max_iter_tvgl < 1 or args.min_iter < 1:
        parser.error('Iteration limits must be positive.')
    if not np.isfinite(args.tol_flipflop) or args.tol_flipflop < 0:
        parser.error('--tol_flipflop must be finite and nonnegative.')
    if args.gaussian_seed < 0:
        parser.error('--gaussian-seed must be nonnegative.')
    if args.gauge == 'none':
        args.gauge = None
    if args.data_path is not None:
        if args.breaks_json is not None or args.seeds_json is not None:
            parser.error('External data cannot be combined with synthetic breaks/seeds.')
        if args.samples_per_t_json is not None or args.samples_per_t != 1:
            parser.error('External sample counts are read from the input array.')
        args.breaks_list = None
        args.seeds_list = None
        return args
    if args.breaks_json is None:
        parser.error('--breaks-json is required for synthetic data.')

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

    if args.samples_per_t_json is None:
        if args.samples_per_t < 1:
            raise ValueError("--samples-per-t must be >= 1.")
        samples_per_t = int(args.samples_per_t)
    else:
        if (not isinstance(args.samples_per_t_json, list)) or len(args.samples_per_t_json) != T:
            raise ValueError("--samples-per-t-json must be a list with length T.")
        if any(int(v) < 1 for v in args.samples_per_t_json):
            raise ValueError("All values in --samples-per-t-json must be >= 1.")
        samples_per_t = [int(v) for v in args.samples_per_t_json]

    if not supports_multi and (isinstance(samples_per_t, list) or samples_per_t != 1):
        parser.error('main supports one sample per timestamp; use multi-samples for batches.')
    for seeds in seeds_list:
        if any(seed is not None and (type(seed) is not int or seed < 0) for seed in seeds):
            parser.error('Graph seeds must be nonnegative integers or null.')
    args.breaks_list = breaks_list
    args.seeds_list = seeds_list
    args.T = T
    if args.gauge == "none":
        args.gauge = None
    args.samples_per_t = samples_per_t
    return args


@contextmanager
def measure(timings, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - start


def run_ktvgl(args, model_class, solver, generate, supports_multi=True):
    start = time.perf_counter()
    metadata = {}
    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError('Dataset metadata must be a JSON object.')
    dataset_type = 'real' if args.data_path is not None else 'synthetic'
    dataset_id = args.dataset_id or metadata.get('dataset_id') or (
        args.data_path.stem if args.data_path is not None else 'synthetic')
    dataset = dict(dataset_id=dataset_id,
                   dataset_name=args.dataset_name or metadata.get('dataset_name') or dataset_id,
                   dataset_version=args.dataset_version or metadata.get('dataset_version'),
                   dataset_type=dataset_type, dims=args.dims, T=args.T,
                   source=None, data_hash=None, samples_per_t=None, coordinates=None,
                   preprocessing=dict(applied_by_runner='none',
                                      declared_upstream=metadata.get('preprocessing')),
                   user_metadata=metadata)
    model = model_class(
        lambdas=args.lambdas, rhos=args.rhos, tvgl_solver=solver,
        init_method=args.init, max_iter=args.max_iter, min_iter=args.min_iter,
        max_iter_tvgl=args.max_iter_tvgl, tol_flipflop=args.tol_flipflop,
        psi=args.psi, gauge=args.gauge, verbose=args.verbose,
        early_stopping=not args.no_early_stopping,
    )
    settings = {name: getattr(model, name) for name in inspect.signature(model_class).parameters
                if hasattr(model, name) and not callable(getattr(model, name))}
    defaults = {name: p.default for name, p in inspect.signature(solver).parameters.items()
                if p.default is not inspect.Parameter.empty}
    run = RunArtifacts(args.base_dir, dataset_id, 'KTVGL', dict(
        dataset=dataset, model=settings, inner_solver_defaults=defaults,
        inner_solver_overrides=dict(psi=args.psi, max_iter=args.max_iter_tvgl,
                                    verbose=args.verbose,
                                    alpha_by_mode=[args.lambdas[m] / int(np.prod(args.dims[:m] + args.dims[m+1:]))
                                                   for m in range(len(args.dims))],
                                    beta_by_mode=[args.rhos[m] / int(np.prod(args.dims[:m] + args.dims[m+1:]))
                                                  for m in range(len(args.dims))]),
        requested_arguments=vars(args), generation=None,
        evaluation=dict(aggregate='micro', use_partial=False, truth_tolerance=0.1,
                        AUPRC_definition='average_precision',
                        bestF1_definition='best threshold selected using ground truth'),
        objective=dict(data_fit='sample-summed negative log likelihood; constants and factor 1/2 omitted',
                       temporal=args.psi, sparsity='off-diagonal L1'),
        history=dict(file='history.csv', columns=list(HISTORY_FIELDS),
                     iteration='1-based completed outer sweeps; no initialization row',
                     elapsed_seconds='time since outer-loop entry, excluding initialization',
                     iteration_seconds='sweep plus evaluation/diagnostics, before CSV write',
                     relative_improvement='(previous total - total) / abs(previous total); blank initially or for zero denominator',
                     relative_precision_change='joint Frobenius norm of change / max(previous norm, 1e-12)',
                     missing_values='blank cells; includes non-finite scalar values'),
        inner_diagnostics=dict(file='inner_summary.csv', columns=list(INNER_FIELDS),
                               outer_iteration='1-based', mode='0-based',
                               scope='KTVGL inner ADMM subproblems; final residuals only'),
    ))
    with RunLog(run.path / 'run.log') as run_log:
        run.summary['log'] = 'run.log'
        print('Run directory:', run.path, flush=True)
        timings = {}
        phase = 'data_preparation'
        truth = None
        history = None
        try:
            with measure(timings, phase):
                if args.data_path is not None:
                    source = args.data_path.expanduser().resolve()
                    dataset['source'] = dict(path=str(source), sha256=file_hash(source))
                    batches, truth = load_observations(source, args.dims)
                else:
                    effective_seeds = [[secrets.randbits(32) if seed is None else seed for seed in seeds]
                                       for seeds in args.seeds_list]
                    generation = dict(dims=args.dims, T=args.T, breaks_list=args.breaks_list,
                                      seeds_list=effective_seeds, gaussian_seed=args.gaussian_seed,
                                      p_edge=args.p_edge, w_range=tuple(args.wrange), tol=0.1)
                    if supports_multi:
                        generation['samples_per_t'] = args.samples_per_t
                    run.config['generation'] = generation
                    run.write_config()
                    x, truth = generate(**generation)
                    batches = canonical_batches(x, args.dims)
                if args.T is not None and args.T != len(batches):
                    raise ValueError('--T does not match the input time axis.')
                counts = [len(batch) for batch in batches]
                if not supports_multi and any(n != 1 for n in counts):
                    raise ValueError('main supports one sample per timestamp; use multi-samples for batches.')
                # Both layouts preserve values and the model's sample interpretation.
                if all(n == 1 for n in counts):
                    x = np.stack([batch[0] for batch in batches])
                elif len(set(counts)) == 1:
                    x = np.stack(batches)
                else:
                    x = batches
                dataset.update(T=len(batches), samples_per_t=counts, data_hash=data_hash(batches),
                               coordinates=coordinates(metadata, args.dims, len(batches)))
                run.write_config()
                run.summary['ground_truth_available'] = truth is not None
            phase = 'fit'
            with measure(timings, phase), HistoryWriter(run.path / 'history.csv') as history, \
                    InnerSummaryWriter(run.path / 'inner_summary.csv') as inner_writer:
                estimated = model.fit(x, iteration_callback=history, inner_callback=inner_writer)
            run.summary['history'] = dict(file='history.csv', rows=history.rows)
            run.summary['inner_diagnostics'] = summarize_inner(model.inner_history_)
            inner = run.summary['inner_diagnostics']
            if inner['unconverged_count']:
                warnings.warn(f"{inner['unconverged_count']} of {inner['subproblems']} inner ADMM subproblems did not converge "
                              f"({inner['max_iter_count']} reached max_iter); see inner_summary.csv.", RuntimeWarning)
            if inner['unknown_count']:
                warnings.warn(f"Convergence diagnostics unavailable for {inner['unknown_count']} inner subproblems.", RuntimeWarning)
            phase = 'numerical_diagnostics'
            with measure(timings, phase):
                run.summary['numerical_diagnostics'] = precision_diagnostics(estimated)
            if not run.summary['numerical_diagnostics']['valid_precision']:
                warnings.warn('Estimated precision matrices failed numerical checks; see summary.json.', RuntimeWarning)
            phase = 'precision_save'
            with measure(timings, phase):
                arrays = save_precision(run.path / 'precision.npz', estimated, truth)
                run.summary['precision'] = dict(file='precision.npz', arrays=arrays)
            phase = 'evaluation'
            with measure(timings, phase):
                metrics = ktvgl_eval(estimated, truth) if truth is not None else None
            phase = 'figures'
            with measure(timings, phase):
                if not args.no_plots:
                    figures = run.path / 'figures'
                    plot_history(read_history(run.path / 'history.csv'), figures,
                                 title=f'KTVGL / {dataset_id}')
                    viz_change_of_theta(estimated, save_dir=figures)
                    if truth is not None:
                        viz_error_of_theta(estimated, truth, save_dir=figures)
            timings['total'] = time.perf_counter() - start
            run.summary['warning_count'] = run_log.warning_count
            run.finish('completed', **model.fit_info_, metrics=metrics, timing_seconds=timings)
            print(f'Fit time: {timings["fit"]:.3f} seconds')
            print('Stop reason:', model.fit_info_['stop_reason'])
            if metrics is not None:
                print('Results (KTVGL):', metrics)
            return run.path
        except BaseException as error:
            timings['total'] = time.perf_counter() - start
            if history is not None:
                run.summary['history'] = dict(file='history.csv', rows=history.rows)
            run.summary['inner_diagnostics'] = summarize_inner(model.inner_history_)
            if not (run.path / 'inner_summary.csv').exists():
                run.summary['inner_diagnostics']['file'] = None
            run.write_config()
            run.summary['warning_count'] = run_log.warning_count
            run.finish('failed', **model.fit_info_, timing_seconds=timings,
                       error=dict(type=type(error).__name__, message=str(error), phase=phase))
            raise
