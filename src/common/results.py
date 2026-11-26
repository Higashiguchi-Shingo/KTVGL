"""Versioned run artifacts shared by synthetic and external datasets."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import uuid

import numpy as np

SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Write strict JSON; unavailable/non-finite scalar values become null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_batches(x, dims):
    """Validate observations and represent every timestamp as (N_t, *dims)."""
    dims = tuple(dims)
    if isinstance(x, list):
        batches = x
    else:
        x = np.asarray(x)
        if x.ndim == len(dims) + 1:
            batches = [a[None] for a in x]
        elif x.ndim == len(dims) + 2:
            batches = list(x)
        else:
            raise ValueError('Input must have shape (T, *dims) or (T, N, *dims).')
    if not batches:
        raise ValueError('Input must contain at least one timestamp.')
    result = []
    for batch in batches:
        original = np.asarray(batch)
        if not np.issubdtype(original.dtype, np.number) or np.iscomplexobj(original):
            raise ValueError('Observations must be real numeric arrays.')
        a = np.asarray(original, dtype=np.float64)
        if a.ndim != len(dims) + 1 or a.shape[1:] != dims or a.shape[0] < 1:
            raise ValueError('Each batch must have shape (N_t, *dims), with N_t >= 1.')
        if not np.isfinite(a).all():
            raise ValueError('Observations must be finite; handle missing values before fitting.')
        result.append(a)
    return result


def data_hash(batches):
    """Hash canonical float64 samples, including shapes and time boundaries."""
    digest = hashlib.sha256(b'ktvgl-data-v1\0')
    for batch in batches:
        a = np.ascontiguousarray(batch, dtype='<f8')
        digest.update(json.dumps(a.shape).encode() + b'\0')
        digest.update(a.tobytes())
    return digest.hexdigest()


def load_observations(path, dims):
    """NPY: X; NPZ: X or X_0,...,X_(T-1), plus optional true_mode_m arrays."""
    path = Path(path)
    truth = None
    if path.suffix.lower() == '.npy':
        x = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            batch_keys = [k for k in archive.files if re.fullmatch(r'X_\d+', k)]
            if 'X' in archive:
                if batch_keys:
                    raise ValueError('NPZ must use either X or X_0,..., not both.')
                x = archive['X'].copy()
            else:
                expected = [f'X_{t}' for t in range(len(batch_keys))]
                if not expected or set(batch_keys) != set(expected):
                    raise ValueError('NPZ requires X or consecutive X_0,... sample batches.')
                x = [archive[k].copy() for k in expected]
            truth_keys = [k for k in archive.files if k.startswith('true_mode_')]
            if truth_keys:
                expected = [f'true_mode_{m}' for m in range(len(dims))]
                if set(truth_keys) != set(expected):
                    raise ValueError('Ground truth must contain one true_mode_m array per mode.')
                truth = [archive[k].copy() for k in expected]
    else:
        raise ValueError('Input format must be .npy or .npz.')
    batches = canonical_batches(x, dims)
    if truth is not None:
        for m, a in enumerate(truth):
            if a.shape != (len(batches), dims[m], dims[m]) or not np.isfinite(a).all():
                raise ValueError('Ground-truth shape must be (T, d_m, d_m) with finite values.')
        truth = [[truth[m][t] for m in range(len(dims))] for t in range(len(batches))]
    return batches, truth


def coordinates(metadata, dims, length):
    modes = metadata.get('mode_names', [f'mode_{m}' for m in range(len(dims))])
    variables = metadata.get('variable_names', [[str(i) for i in range(d)] for d in dims])
    times = metadata.get('time_values', list(range(length)))
    if not isinstance(modes, list) or len(modes) != len(dims):
        raise ValueError('mode_names must contain one name per mode.')
    if (not isinstance(variables, list) or len(variables) != len(dims)
            or any(not isinstance(v, list) or len(v) != d for v, d in zip(variables, dims))):
        raise ValueError('variable_names must match each mode dimension.')
    if not isinstance(times, list) or len(times) != length:
        raise ValueError('time_values must contain one value per timestamp.')
    return dict(mode_names=modes, variable_names=variables, time_values=times,
                time_kind=metadata.get('time_kind', 'provided' if 'time_values' in metadata else 'index'))


def collect_environment():
    root = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists()), None)
    git = dict(commit=None, branch=None, dirty=None, status=None)
    if root is not None:
        def command(*args):
            return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        try:
            git.update(commit=command('rev-parse', 'HEAD'), branch=command('branch', '--show-current'))
            git['status'] = command('status', '--porcelain')
            git['dirty'] = bool(git['status'])
        except (OSError, subprocess.SubprocessError):
            pass
    versions = {}
    for name in ('ktvgl', 'numpy', 'scipy', 'scikit-learn', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(schema_version=SCHEMA_VERSION, python=platform.python_version(),
                platform=platform.platform(), packages=versions, git=git)


def save_precision(path, estimated, truth=None):
    """Per-mode numeric arrays support unequal mode sizes without pickle."""
    if not len(estimated) or not len(estimated[0]):
        raise ValueError('Precision sequence must be non-empty.')
    arrays = {}
    for name, sequence in (('estimated', estimated), ('true', truth)):
        if sequence is None:
            continue
        if len(sequence) != len(estimated) or any(len(row) != len(estimated[0]) for row in sequence):
            raise ValueError('Precision sequences must have consistent time and mode counts.')
        for m in range(len(estimated[0])):
            a = np.stack([row[m] for row in sequence])
            if a.ndim != 3 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
                raise ValueError('Precision matrices must be finite square matrices.')
            if name == 'true' and a.shape != arrays[f'estimated_mode_{m}'].shape:
                raise ValueError('True and estimated precision shapes must match.')
            arrays[f'{name}_mode_{m}'] = a
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {k: dict(shape=list(v.shape), axes=['time', 'row_variable', 'column_variable'])
            for k, v in arrays.items()}


def load_precision(path):
    """Read arrays back into the fit() time-major list-of-modes format."""
    with np.load(path, allow_pickle=False) as archive:
        def sequence(prefix):
            keys = sorted((k for k in archive.files if k.startswith(prefix)),
                          key=lambda k: int(k[len(prefix):]))
            if not keys:
                return None
            arrays = [archive[k] for k in keys]
            return [[a[t].copy() for a in arrays] for t in range(len(arrays[0]))]
        return sequence('estimated_mode_'), sequence('true_mode_')


class RunArtifacts:
    """Create a unique run, write its config immediately, and record its outcome."""
    def __init__(self, base_dir, dataset_id, method, config):
        for value in (dataset_id, method):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
                raise ValueError('Dataset ID and method must be safe ASCII path components.')
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
        self.path = Path(base_dir).expanduser().resolve() / dataset_id / method / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = dict(config, schema_version=SCHEMA_VERSION, run_id=self.run_id, method=method)
        self.summary = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id, method=method,
                            dataset_id=dataset_id, status='running', started_at=utc_now(),
                            ended_at=None, stop_reason=None, n_iter=None, final_objective=None,
                            ground_truth_available=None, metrics=None, timing_seconds={},
                            precision=None, history=None, inner_diagnostics=None,
                            numerical_diagnostics=None, log=None, warning_count=0, error=None)
        self.write_config()
        write_json(self.path / 'environment.json', collect_environment())
        write_json(self.path / 'summary.json', self.summary)

    def write_config(self):
        write_json(self.path / 'config.json', self.config)

    def finish(self, status, **values):
        self.summary.update(values, status=status, ended_at=utc_now())
        write_json(self.path / 'summary.json', self.summary)
