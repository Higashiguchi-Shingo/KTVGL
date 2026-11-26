import contextlib
import io
import tempfile
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("kgl", "ktvgl", "streamktvgl", "streamtvgl", "tvgl")


class DataGenerated(Exception):
    """Stop the training entry point immediately after generating its data."""


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "exp" / f"train_{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scripts = {name: load_script(name) for name in SCRIPTS}

    def generate_from_cli(self, script, seeds, expected_seeds):
        argv = ["train", "--dims", "3", "3", "--T", "6",
                "--breaks-json", "[[0,3,6],[0,3,6]]", "--p-edge", "1"]
        if seeds is not None:
            argv.extend(["--seeds-json", json.dumps(seeds)])
        generated = []
        generate = script.generate_kron_data

        def capture(**kwargs):
            if script.__name__ == "ktvgl" and seeds is None:
                self.assertEqual([len(row) for row in kwargs['seeds_list']], [2, 2])
                self.assertTrue(all(isinstance(seed, int) for row in kwargs['seeds_list'] for seed in row))
            else:
                self.assertEqual(kwargs.get("seeds_list"), expected_seeds)
            generated.append(generate(**kwargs))
            raise DataGenerated

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            argv.extend(['--base-dir', directory])
            with patch.object(sys, "argv", argv), patch.object(script, "generate_kron_data", capture):
                with self.assertRaises(DataGenerated):
                    script.main()
        return generated[0]

    def test_requested_seeds_reproduce_data_and_changing_seeds_changes_data(self):
        seeds = [[11, 12], [13, 14]]
        changed_seeds = [[21, 22], [23, 24]]
        for name, script in self.scripts.items():
            with self.subTest(script=name):
                x1, theta1 = self.generate_from_cli(script, seeds, seeds)
                x2, theta2 = self.generate_from_cli(script, seeds, seeds)
                x3, theta3 = self.generate_from_cli(script, changed_seeds, changed_seeds)
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_array_equal(theta1, theta2)
                self.assertFalse(np.array_equal(x1, x3))
                self.assertFalse(np.array_equal(theta1, theta3))

    def test_broadcast_and_unspecified_seeds(self):
        for name, script in self.scripts.items():
            with self.subTest(script=name):
                self.generate_from_cli(script, [[11, 12]], [[11, 12], [11, 12]])
                self.generate_from_cli(script, None, [[None, None], [None, None]])


if __name__ == "__main__":
    unittest.main()
