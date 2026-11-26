"""Train KTVGL on synthetic or external tensor data and save a versioned run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from methods.KroneckerTVGL import KroneckerTVGL, time_varying_graphical_lasso
from common.synthetic import generate_kron_data
from common.ktvgl_experiment import parse_args as _parse_args, run_ktvgl


def parse_args():
    return _parse_args(supports_multi=False)


def main():
    return run_ktvgl(parse_args(), KroneckerTVGL, time_varying_graphical_lasso,
                     generate_kron_data, supports_multi=False)


if __name__ == '__main__':
    main()
