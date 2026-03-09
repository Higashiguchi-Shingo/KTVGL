# KTVGL

Kronecker Time-Varying Graphical Lasso (KTVGL) implementation.

Paper: [Interpretable Dynamic Network Modeling of Tensor Time Series via Kronecker Time-Varying Graphical Lasso](https://arxiv.org/abs/2602.08197)

## Branch Policy

This `multi-samples` branch is maintained as a package-oriented implementation
for external use.

If you are looking for the paper reproduction code and original experiment setup,
please use the `main` branch.

- `multi-samples`: package usage, API-focused structure, multi-sample support per timestamp
- `main`: paper reproduction scripts and experiment-oriented layout

## Package Layout

Core library code is under:

- `src/ktvgl/common`
- `src/ktvgl/methods`
- `src/ktvgl/datasets`

Experiment scripts are under `exp/`.

## Installation

### Local development

```bash
poetry install
```

### Use from another project (GitHub)

To install this branch explicitly:

```bash
poetry add "git+https://github.com/Higashiguchi-Shingo/KTVGL.git#multi-samples"
```

Then import:

```python
from ktvgl.methods.kronecker_tvgl import KroneckerTVGL, tvgl_solver
from ktvgl.common.synthetic import generate_kron_data
```

## Quick Start

```python
from ktvgl.methods.kronecker_tvgl import KroneckerTVGL, tvgl_solver
from ktvgl.common.synthetic import generate_kron_data

X_seq, thetas_true = generate_kron_data(
    dims=[10, 8],
    breaks_list=[[0, 50], [0, 50]],
    T=50,
    samples_per_t=1,
)

model = KroneckerTVGL(
    lambdas=[0.01, 0.01],
    rhos=[2.0, 2.0],
    tvgl_solver=tvgl_solver,
    init_method="empirical",
)

thetas_hat = model.fit(X_seq)
```

## Input Formats (`X_seq`)

`KroneckerTVGL.fit` supports:

1. Single sample per timestep:
   - `ndarray`, shape `(T, d1, ..., dM)`
2. Fixed multiple samples per timestep:
   - `ndarray`, shape `(T, N, d1, ..., dM)`
3. Variable samples per timestep:
   - `list` of length `T`, each item `ndarray` with shape `(N_t, d1, ..., dM)`

`mode` always refers to tensor modes `(d1, ..., dM)`, not the sample axis.

## Synthetic Data

Synthetic data generation is provided by `ktvgl.common.synthetic.generate_kron_data`.

You can control per-timestep sample count via `samples_per_t`:

- `samples_per_t=1` (default)
- `samples_per_t=<int>` (fixed)
- `samples_per_t=[N_0, ..., N_{T-1}]` (variable)

## Experiments

Run synthetic KTVGL experiment:

```bash
poetry run python exp/train_ktvgl.py \
  --dims 20 30 \
  --breaks-json '[[0,100],[0,100]]' \
  --samples-per-t 5
```

## Notes

- `init_method="Glasso"` is currently disabled.
- Real-world datasets used in experiments are in `data/`.
