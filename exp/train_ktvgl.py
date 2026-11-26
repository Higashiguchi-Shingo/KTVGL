"""Train KTVGL on synthetic or external tensor data and save a versioned run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ktvgl.methods.kronecker_tvgl import KroneckerTVGL, tvgl_solver
from ktvgl.common.synthetic import generate_kron_data
from ktvgl.common.ktvgl_experiment import parse_args as _parse_args, run_ktvgl


def parse_args():
    return _parse_args(supports_multi=True)


def main():
    return run_ktvgl(parse_args(), KroneckerTVGL, tvgl_solver,
                     generate_kron_data, supports_multi=True)


if __name__ == '__main__':
    main()
