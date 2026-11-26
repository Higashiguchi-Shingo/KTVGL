"""CLI entry point kept separate from the history module imported by models."""
import argparse
from pathlib import Path

from .history import replot_run


def main():
    parser = argparse.ArgumentParser(description='Regenerate convergence figures from a saved KTVGL run.')
    parser.add_argument('run_dir', type=Path)
    for path in replot_run(parser.parse_args().run_dir):
        print(path)


if __name__ == '__main__':
    main()
