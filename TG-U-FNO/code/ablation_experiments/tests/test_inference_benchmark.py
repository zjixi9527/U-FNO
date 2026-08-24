import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_wavefield_compare_00_06_sem.py"
)
SPEC = importlib.util.spec_from_file_location("wavefield_benchmark", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return inputs + 1.0


class InferenceBenchmarkTests(unittest.TestCase):
    def test_benchmark_uses_warmup_and_records_every_repeat(self):
        model = CountingModel()
        inputs = torch.zeros(1, 2)

        prediction, statistics = MODULE.benchmark_model_forward(
            model,
            inputs,
            device=torch.device("cpu"),
            warmup_runs=2,
            repeat_runs=5,
        )

        self.assertEqual(model.forward_calls, 7)
        torch.testing.assert_close(prediction, torch.ones_like(inputs))
        self.assertEqual(statistics["warmup_runs"], 2)
        self.assertEqual(statistics["repeat_runs"], 5)
        self.assertEqual(len(statistics["individual_seconds"]), 5)
        self.assertGreaterEqual(statistics["mean_seconds"], 0.0)
        self.assertGreaterEqual(statistics["median_seconds"], 0.0)
        self.assertGreaterEqual(statistics["p95_seconds"], 0.0)

    def test_benchmark_rejects_nonpositive_repeat_count(self):
        with self.assertRaisesRegex(ValueError, "repeat_runs"):
            MODULE.benchmark_model_forward(
                CountingModel(),
                torch.zeros(1, 2),
                device=torch.device("cpu"),
                warmup_runs=0,
                repeat_runs=0,
            )

    def test_wavefield_plot_supports_three_models(self):
        shape = (4, 4, 2, 3)
        fields = {
            "FNO": np.zeros(shape, dtype=np.float32),
            "U-FNO": np.ones(shape, dtype=np.float32),
            "Terrain-gated U-FNO": np.full(shape, 2.0, dtype=np.float32),
            "SEM truth": np.full(shape, 0.5, dtype=np.float32),
        }

        with patch.object(MODULE, "save_figure") as figure_saver:
            MODULE.plot_wavefield_comparison(
                fields=fields,
                time_indices_1based=[1, 2],
                sample_index=23,
                output_dir=Path("unused"),
                field_percentile=99.5,
                dpi=72,
                save_svg=False,
                model_labels=["FNO", "U-FNO", "Terrain-gated U-FNO"],
                ref_label="SEM truth",
            )

        figure_saver.assert_called_once()


if __name__ == "__main__":
    unittest.main()
