"""Outer-iteration history, incremental CSV output, and offline plotting."""
import csv
import json
from pathlib import Path
import time

import numpy as np


HISTORY_FIELDS = (
    'iteration', 'elapsed_seconds', 'iteration_seconds', 'total', 'data_fit',
    'sparsity', 'temporal', 'relative_improvement', 'relative_precision_change',
)


def history_row(iteration, terms, previous_objective, previous_thetas, thetas,
                started, iteration_started):
    """Record one completed sweep; timing starts at entry to the outer loop."""
    difference = 0.
    previous_size = 0.
    for old_row, new_row in zip(previous_thetas, thetas):
        for old, new in zip(old_row, new_row):
            difference += float(np.sum((new - old) ** 2))
            previous_size += float(np.sum(old ** 2))
    relative = None
    if previous_objective is not None and np.isfinite(previous_objective) and previous_objective != 0:
        relative = float((previous_objective - terms['total']) / abs(previous_objective))
    now = time.perf_counter()
    return dict(iteration=iteration, elapsed_seconds=now - started,
                iteration_seconds=now - iteration_started, **terms,
                relative_improvement=relative,
                relative_precision_change=float(np.sqrt(difference) / max(np.sqrt(previous_size), 1e-12)))


class HistoryWriter:
    """Flush each completed outer iteration so failed runs retain their history."""
    def __init__(self, path):
        self.path = Path(path)
        self.rows = 0

    def __enter__(self):
        self.stream = self.path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream, fieldnames=HISTORY_FIELDS)
        self.writer.writeheader()
        self.stream.flush()
        return self

    def __call__(self, row):
        values = {key: row[key] for key in HISTORY_FIELDS}
        for key, value in values.items():
            if value is None or not np.isfinite(value):
                values[key] = ''
        self.writer.writerow(values)
        self.stream.flush()
        self.rows += 1

    def __exit__(self, *exception):
        self.stream.close()


def read_history(path):
    """Load CSV without rounding objective values; blank cells represent missing data."""
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(HISTORY_FIELDS).issubset(reader.fieldnames):
            raise ValueError('CSV does not contain the required KTVGL history columns.')
        rows = []
        for record in reader:
            row = {key: float(record[key]) if record[key] else None for key in HISTORY_FIELDS}
            row['iteration'] = int(row['iteration'])
            rows.append(row)
        return rows


def plot_history(rows, output_dir, title='KTVGL'):
    """Draw saved outer-loop values on linear axes, including negative objectives."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not rows:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {key: np.array([np.nan if row[key] is None else row[key] for row in rows])
              for key in HISTORY_FIELDS}
    paths = []

    def draw(ax, x, y, xlabel, ylabel):
        ax.plot(values[x], values[y], marker='o', markersize=3, linewidth=1.5)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if x == 'iteration':
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=0.3)

    def save(fig, filename, likelihood_note=False):
        path = output_dir / filename
        fig.suptitle(title)
        if likelihood_note:
            fig.text(0.5, 0.015, '* Summed over samples; constants and factor 1/2 omitted.',
                     ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.05 if likelihood_note else 0, 1, 1))
        try:
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    draw(ax, 'iteration', 'total', 'Outer iteration (completed sweep)', 'Objective total')
    save(fig, 'objective.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    draw(axes[0], 'iteration', 'data_fit', 'Outer iteration (completed sweep)', 'Data fit (negative log-likelihood*)')
    draw(axes[1], 'elapsed_seconds', 'data_fit', 'Outer-loop elapsed time (s)', 'Data fit (negative log-likelihood*)')
    save(fig, 'data_fit.png', likelihood_note=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, key, label in zip(axes, ('data_fit', 'sparsity', 'temporal'),
                              ('Data fit (negative log-likelihood*)', 'Sparsity penalty', 'Temporal penalty')):
        draw(ax, 'iteration', key, 'Outer iteration (completed sweep)', label)
    save(fig, 'objective_terms.png', likelihood_note=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    draw(ax, 'elapsed_seconds', 'total', 'Outer-loop elapsed time (s)', 'Objective total')
    save(fig, 'objective_vs_time.png')
    return paths


def replot_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    title = f"{config['method']} / {config['dataset']['dataset_id']}"
    return plot_history(read_history(run_dir / 'history.csv'), run_dir / 'figures', title)
