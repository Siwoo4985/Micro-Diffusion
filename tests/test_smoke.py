import importlib.util
import os
import subprocess
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run_command(args):
    return subprocess.check_output(
        [PYTHON] + args,
        cwd=ROOT,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )


class SmokeTests(unittest.TestCase):
    def test_scripts_compile(self):
        run_command(["-m", "py_compile", "train.py", "train_minimal.py", "train_pure.py"])

    def test_train_minimal_smoke(self):
        output = run_command(["train_minimal.py", "--steps", "1", "--samples", "2", "--quiet"])
        self.assertIn("Generated names:", output)

    def test_train_pure_smoke(self):
        output = run_command(
            [
                "train_pure.py",
                "--steps",
                "1",
                "--samples",
                "2",
                "--quiet",
                "--no-temperature-sweep",
                "--no-forward-preview",
            ]
        )
        self.assertIn("Generated names:", output)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch not installed")
    def test_train_torch_smoke(self):
        output = run_command(
            [
                "train.py",
                "--steps",
                "1",
                "--samples",
                "1",
                "--quiet",
                "--no-temperature-sweep",
                "--no-forward-preview",
            ]
        )
        self.assertIn("Generated names:", output)
