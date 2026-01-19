import pandas as pd
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from statsmodels.tsa.seasonal import STL
import json
from pathlib import Path
from typing import Iterable, Optional, Any

# load data
def load_tensor(data_path, time_key, modes, values=None, sampling_rate="D", start_date=None, end_date=None, scaler=None, verbose=False, stl=False):
    df = pd.read_csv(data_path)
    tensor = df2tts(df, time_key=time_key, modes=modes, values=values, start_date=start_date, end_date=end_date)

    if stl:
        print("STL decomposition")
        tensor, Xs, _ = ST_decomp(tensor, stl_period=13)

    if verbose:
        for m in modes:
            print(sorted(list(set(df[m]))))

    if scaler=="full":
        tensor =  minmax_scale(tensor.reshape((-1, 1))).reshape(tensor.shape)
    elif scaler=="each":
        tensor = min_max_scale_tensor(tensor)
    elif scaler=="normalize_full":
        tensor = normalize_tensor(tensor)
    elif scaler=="normalize_each":
        tensor = normalize_tensor_each(tensor)
    elif scaler=="centerize_each":
        tensor = minmax_scale(tensor.reshape((-1, 1))).reshape(tensor.shape)
        tensor = centerize_tensor_each(tensor)

    if stl:
        return tensor, Xs
    else:
        return tensor

def df2tts(df, time_key, modes, values=None, sampling_rate="D", start_date=None, end_date=None):
    """ 
    Convert a DataFrame (list) to tensor time series

    Parameters
    ----------
    df (pandas.DataFrame):
        A list of discrete events
    time_key (str):
        A column name of timestamps
    modes (list):
        A list of column names to make tensor timeseries
    values (str):
        A column name of target values (optional)
    sampling_rate (str):
        A frequancy for resampling, e.g., "7D", "12H", "H"
    
    returns
    -------
    tts (numpy.ndarray):
        A tensor time series
    """
    df[time_key] = pd.to_datetime(df[time_key])
    if start_date is not None: df = df[lambda x: x[time_key] >= pd.to_datetime(start_date)]
    if end_date is not None: df = df[lambda x: x[time_key] <= pd.to_datetime(end_date)]
    tmp = df.copy(deep=True)
    shape = tmp[modes].nunique().tolist()
    if values == None: values = 'count'; tmp[values] = 1
    tmp[time_key] = tmp[time_key].round(sampling_rate)
    print("Tensor:")
    print(tmp.nunique()[[time_key] + modes])

    grouped = tmp.groupby([time_key] + modes).sum()[[values]]
    grouped = grouped.unstack(fill_value=0).stack()
    grouped = grouped.pivot_table(index=time_key, columns=modes, values=values, fill_value=0)

    tts = grouped.values
    tts = np.reshape(tts, (-1, *shape))
    return tts

def min_max_scale_np(array):
    min = array.min()
    max = array.max()
    array = (array - min) / (max - min)
    return array

def min_max_scale_tensor(data):
    query_size = data.shape[1]
    geo_size = data.shape[2]
    ret = np.zeros(shape=data.shape)
    for i in range(query_size):
        for j in range(geo_size):
            ret[:,i,j] = min_max_scale_np(data[:,i,j])
    return ret

def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std() + 1e-8
    return (tensor - mean) / std

def normalize_tensor_each(tensor):
    mean = tensor.mean(axis=0)
    std = tensor.std(axis=0) + 1e-8
    return (tensor - mean) / std

def centerize_tensor_each(tensor):
    mean = tensor.mean(axis=0)
    return tensor - mean

def sf_unfold(tensor, gl_mode=None, mode=0):
    '''
    Unfold a tensor along a specified mode and permute the dimensions according to the global mode.

    Example
    -------
    input
        tensor.shape = [10, 2,3,4,5]
        gl_mode = [0,1,1,0]
        mode = 0
    return
        X.shape = [10*5, 3*4, 2]

    input
        tensor.shape = [10, 2,3,4,5]
        gl_mode = [0,1,1,0]
        mode = 2
    return
        X.shape = [10*2*5, 3, 4]
    '''
    ndim=tensor.ndim -1
    if not gl_mode:
        gl_mode = [1 for i in range(ndim)]

    indices = np.where(np.array(gl_mode) == 1)
    indices = list(indices[0])
    if gl_mode[mode] == 0:
        rest = list(set(np.arange(ndim)) - set([*indices,mode]))
        rest_indices = indices
        order = [*rest,*rest_indices,mode]
    elif gl_mode[mode] == 1:
        rest = list(set(np.arange(ndim)) - set([*indices]))
        rest_indices = list(set([*indices]) - set([mode]))

    order = [*rest,*rest_indices,mode]
    order = [val+1 for val in order]
    X = np.transpose(tensor, [0,*order])
    id_dim = 1
    for id in rest_indices:
        id_dim *= tensor.shape[id+1]
    X = X.reshape(-1, id_dim, tensor.shape[mode+1])
    
    return np.squeeze(X)

def lag_concat_unfold(X, mode=0):
    X1 = sf_unfold(X[:-1], mode=mode)
    X2 = sf_unfold(X[1:], mode=mode)
    X = np.concatenate((X1, X2), axis=2)
    return X

def MAE(X, Y):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(X - Y))

def ST_decomp(tensor, stl_period):
    trend = np.zeros(shape=tensor.shape)
    seasonal = np.zeros(shape=tensor.shape)
    resid = np.zeros(shape=tensor.shape)

    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[2]):
            stl = STL(tensor[:,i,j], robust=True, period=stl_period)
            stl_series = stl.fit()
            trend[:,i,j] = stl_series.trend
            seasonal[:,i,j] = stl_series.seasonal
            resid[:,i,j] = stl_series.resid
    
    return trend, seasonal, resid

def kronecker_multiple(arrays):
    """
    Parameters
    ----------
    arrays : list of np.ndarray

    Returns
    -------
    result : np.ndarray, Kronecker product of the input arrays
    """
    if not arrays:
        raise ValueError("Input list is empty.")
    
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.kron(result, arr)
    return result

def empirical_covariance(X):
    """
    Estimate the empirical covariance matrix of X.

    Parameters
    ----------
    X : array-like, 
        shape (n_timestep, n_dimensions, n_features) = (T, D(/n), Dn)
        or shepe (n_dimensions, n_features) = (D(/n), Dn).
        Input data.

    Returns
    -------
    covariance : array-like, shape (n_features, n_features) = (Dn, Dn)
                or shape (n_timestep, n_features, n_features) = (T, Dn, Dn)
    """
    X = np.asarray(X)

    if X.ndim == 2:
        covariance = np.cov(X.T, bias=True)

    elif X.ndim == 3:
        covariance = []
        for t in range(X.shape[0]):
            covariance.append(np.cov(X[t,:,:].T, bias=True))

    return np.asarray(covariance)

def _to_token(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "_".join(_to_token(x) for x in v)
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, float):
        s = f"{v:.10g}"          # Ex.: 0.01 -> '0.01'
        return s.replace(".", "p")
    return str(v)

def hp_to_path(
    hp: dict,
    base_dir: str | Path = "runs",
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    include_keys: Optional[Iterable[str]] = None,
    exclude_keys: Iterable[str] = ("verbose",),
    mkdir: bool = True,
) -> Path:
    if include_keys is not None:
        keys = list(include_keys)
    else:
        keys = sorted(k for k in hp.keys() if k not in exclude_keys)

    pairs = []
    for k in keys:
        if k in exclude_keys:
            continue
        if k in hp and hp[k] is not None:
            pairs.append(f"{k}={_to_token(hp[k])}")
    slug = ",".join(pairs) if pairs else "default"

    parts = [Path(base_dir)]
    if method:
        parts.append(Path(method))
    if dataset:
        parts.append(Path(dataset))
    parts.append(Path(slug))
    p = Path(*parts)

    if mkdir:
        p.mkdir(parents=True, exist_ok=True)

    return p

def _to_token(v):
    if isinstance(v, (list, tuple)):
        return "-".join(str(x) for x in v)
    elif isinstance(v, float):
        return f"{v:g}"
    else:
        return str(v)

def hp_to_dirpath(
    hp: dict,
    base_dir: str | Path = "runs",
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    include_keys: Optional[Iterable[str]] = None,
    exclude_keys: Iterable[str] = ("verbose",),
    mkdir: bool = True,
) -> Path:
    if include_keys is not None:
        keys = list(include_keys)
    else:
        keys = sorted(k for k in hp.keys() if k not in exclude_keys)

    parts = [Path(base_dir)]
    if method:
        parts.append(Path(method))
    if dataset:
        parts.append(Path(dataset))

    for k in keys:
        if k in exclude_keys:
            continue
        if k in hp and hp[k] is not None:
            parts.append(Path(f"{k}={_to_token(hp[k])}"))

    if len(parts) == len([base_dir]) + bool(method) + bool(dataset):
        parts.append(Path("default"))

    p = Path(*parts)

    if mkdir:
        p.mkdir(parents=True, exist_ok=True)

    return p