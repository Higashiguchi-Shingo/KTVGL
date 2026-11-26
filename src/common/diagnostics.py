"""Compact subproblem diagnostics and CLI-only run logging."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import inspect
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np

from .results import utc_now


INNER_FIELDS = ('outer_iteration', 'mode', 'n_iter', 'max_iter', 'converged',
                'stop_reason', 'rnorm', 'snorm', 'e_pri', 'e_dual',
                'solver_seconds', 'error')


def invoke_solver(solver, kwargs, info=None):
    """Request optional diagnostics without retrying a solver that raises TypeError."""
    try:
        parameters = inspect.signature(solver).parameters
    except (TypeError, ValueError):
        parameters = None
    def accepts(name):
        return parameters is not None and (name in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
    kwargs = dict(kwargs)
    if parameters is not None and not accepts('n_samples'):
        kwargs.pop('n_samples', None)
    if info is not None:
        info.update(n_iter=None, converged=None, stop_reason='unavailable',
                    rnorm=None, snorm=None, e_pri=None, e_dual=None, error=None)
        if accepts('solver_info'):
            kwargs['solver_info'] = info
    started = time.perf_counter()
    try:
        return solver(**kwargs)
    except BaseException as error:
        if info is not None:
            info.update(converged=False, stop_reason='error',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if info is not None:
            info['solver_seconds'] = time.perf_counter() - started


def record_inner(info, iteration, mode, max_iter, history=None, callback=None, error=None):
    row = {key: None for key in INNER_FIELDS}
    row.update(info or {})
    row.update(outer_iteration=iteration, mode=mode, max_iter=max_iter)
    if error is not None:
        row.update(converged=False, stop_reason='error', error=f'{type(error).__name__}: {error}')
    if history is not None:
        history.append(row)
    if callback is not None:
        callback(row.copy())


def summarize_inner(rows):
    converged = sum(row['converged'] is True for row in rows)
    unknown = sum(row['converged'] is None for row in rows)
    return dict(file='inner_summary.csv', subproblems=len(rows),
                converged_count=converged, unconverged_count=len(rows)-converged-unknown,
                unknown_count=unknown,
                max_iter_count=sum(row['stop_reason'] == 'max_iter' for row in rows),
                error_count=sum(row['stop_reason'] == 'error' for row in rows),
                all_converged=(converged == len(rows)) if rows else None)


class InnerSummaryWriter:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.stream = self.path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream, fieldnames=INNER_FIELDS)
        self.writer.writeheader()
        self.stream.flush()
        return self

    def __call__(self, row):
        values = {}
        for key in INNER_FIELDS:
            value = row.get(key)
            if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
                value = ''
            values[key] = value
        self.writer.writerow(values)
        self.stream.flush()

    def __exit__(self, *exception):
        self.stream.close()


def precision_diagnostics(estimated, symmetry_tolerance=1e-8):
    """Final factor matrices; eigenvalues are evaluated on the symmetric part."""
    nan_count = inf_count = failures = count = 0
    max_asymmetry = 0.
    minimum = np.inf
    for row in estimated:
        for matrix in row:
            a = np.asarray(matrix, dtype=float)
            if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size:
                raise ValueError('Numerical diagnostics require non-empty square matrices.')
            count += 1
            nan_count += int(np.isnan(a).sum())
            inf_count += int(np.isinf(a).sum())
            if not np.isfinite(a).all():
                continue
            scaled = a / max(float(np.abs(a).max()), np.finfo(float).tiny)
            asymmetry = np.linalg.norm(scaled-scaled.T) / max(np.linalg.norm(scaled), np.finfo(float).tiny)
            max_asymmetry = max(max_asymmetry, float(asymmetry))
            try:
                eigenvalues = np.linalg.eigvalsh(a / 2 + a.T / 2)
                if not np.isfinite(eigenvalues).all():
                    failures += 1
                else:
                    minimum = min(minimum, float(eigenvalues.min()))
            except np.linalg.LinAlgError:
                failures += 1
    finite = nan_count == 0 and inf_count == 0 and count > 0
    eigenvalues_known = finite and failures == 0
    return dict(matrix_count=count, nan_count=nan_count, inf_count=inf_count,
                all_finite=finite, symmetry_tolerance=symmetry_tolerance,
                max_relative_symmetry_error=max_asymmetry if finite else None,
                min_eigenvalue=minimum if eigenvalues_known else None,
                eigenvalue_failures=failures,
                valid_precision=bool(eigenvalues_known and minimum > 0 and max_asymmetry <= symmetry_tolerance))


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class RunLog:
    """Tee stdout/stderr and emitted Python warnings during a single CLI run."""
    def __init__(self, path):
        self.path = Path(path)
        self.warning_count = 0

    def __enter__(self):
        self.stack = ExitStack()
        stream = self.stack.enter_context(self.path.open('w', encoding='utf-8'))
        self.stack.enter_context(redirect_stdout(_Tee(sys.stdout, stream)))
        self.stack.enter_context(redirect_stderr(_Tee(sys.stderr, stream)))
        self.stack.enter_context(warnings.catch_warnings())
        def showwarning(message, category, filename, lineno, file=None, line=None):
            self.warning_count += 1
            sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
        warnings.showwarning = showwarning
        print(f'Run started at {utc_now()}')
        return self

    def __exit__(self, exc_type, error, tb):
        try:
            if error is not None:
                traceback.print_exception(exc_type, error, tb, file=sys.stderr)
            print(f'Run ended at {utc_now()}')
        finally:
            self.stack.close()
