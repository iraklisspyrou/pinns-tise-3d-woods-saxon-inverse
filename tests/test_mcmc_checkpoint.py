"""Exercise interrupted sampling and recovery with the real FD likelihood."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from mcmc_checkpoint import save_checkpoint  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("emcee"), "Install the bayesian extra")
class CheckpointIntegrationTests(unittest.TestCase):
    def command(self, output, *extra):
        return [sys.executable, "scripts/fd_mcmc.py", "--output-dir", str(output),
                "--walkers", "12", "--steps", "12", "--burn-in", "2",
                "--n-grid", "160", "--n-theta", "32", "--n-phi", "16",
                "--n-r-spatial", "8", "--posterior-draws", "2",
                "--max-nfev", "3", "--checkpoint-every", "4", *extra]

    def environment(self):
        return {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def run_cli(self, output, *extra):
        return subprocess.run(self.command(output, *extra), cwd=ROOT,
                              env=self.environment(), capture_output=True, text=True)

    def test_killed_run_resumes_to_identical_chain_and_predictive_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, resumed = Path(temporary) / "full", Path(temporary) / "resumed"
            uninterrupted = self.run_cli(full)
            self.assertEqual(uninterrupted.returncode, 0, uninterrupted.stderr)
            process = subprocess.Popen(self.command(resumed), cwd=ROOT,
                                       env=self.environment(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            output = []
            try:
                for line in process.stdout:
                    output.append(line)
                    if "Checkpoint saved:" in line:
                        process.kill()
                        break
                process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
            self.assertTrue((resumed / "mcmc_checkpoint.npz").is_file(), "".join(output))
            self.assertFalse((resumed / "summary.json").exists())
            with np.load(resumed / "mcmc_checkpoint.npz") as archive:
                saved_steps = len(archive["chain"])
            self.assertLess(saved_steps, 12)
            continued = self.run_cli(resumed, "--resume")
            self.assertEqual(continued.returncode, 0, continued.stderr)
            with np.load(full / "posterior_chain.npz") as expected, np.load(resumed / "posterior_chain.npz") as actual:
                for key in ("chain", "log_probability", "posterior_draws", "energy_predictions_all_states"):
                    np.testing.assert_array_equal(actual[key], expected[key])
            expected = json.loads((full / "summary.json").read_text())["mcmc"]
            actual = json.loads((resumed / "summary.json").read_text())["mcmc"]
            self.assertEqual(actual["resumed_from_steps"], saved_steps)
            for key in ("likelihood_evaluations", "prior_rejections"):
                self.assertEqual(actual[key], expected[key])
            self.assertAlmostEqual(actual["mean_acceptance_fraction"], expected["mean_acceptance_fraction"])
            # With all sampling already saved, only rebuild output files.
            again = self.run_cli(resumed, "--resume")
            self.assertEqual(again.returncode, 0, again.stderr)
            with np.load(resumed / "posterior_chain.npz") as archive:
                self.assertEqual(len(archive["chain"]), 12)

    def test_changed_likelihood_and_accidental_restart_preserve_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_cli(output)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = (output / "mcmc_checkpoint.npz").read_bytes()
            for extra in (("--resume", "--sigma-mev", "2.0"), ()):
                rejected = self.run_cli(output, *extra)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertEqual((output / "mcmc_checkpoint.npz").read_bytes(), original)


class AtomicCheckpointTests(unittest.TestCase):
    def test_failed_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            state = np.random.RandomState(1).get_state()
            save_checkpoint(path, np.ones((2, 12, 6)), np.zeros((2, 12)), state, {"version": 1})
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                save_checkpoint(path, np.ones((3, 12, 6)), np.zeros((3, 12)), state, {"invalid": {1}})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
