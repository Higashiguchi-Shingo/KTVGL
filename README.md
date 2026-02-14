## Kronecker Time-Varying Graphical Lasso

This repository provides the official implementation of "Interpretable Dynamic Network Modeling of Tensor Time Series via Kronecker Time-Varying Graphical Lasso" (WWW2026)

[Paper link: https://arxiv.org/abs/2602.08197]

## Environment Setup

This project uses Poetry for dependency management.

Install dependencies:

```bash
poetry install
```

## Dataset

### Synthetic data

Synthetic datasets are generated using the `generate_kron_data` function
defined in `src/common/synthetic.py`.

Experimental scripts (e.g., `exp/train_ktvgl.py`) automatically import and use
this function to generate the synthetic data during execution.

### Real-world daata

All real-world datasets used in the paper (i.e., GoogleTrends datasets) are located in the `data/` directory.