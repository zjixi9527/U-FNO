import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


CODE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

import wave3d1
from ablation_common.wave3d1_gated import Uno3D_T10_Gated


class ModelPerformanceTests(unittest.TestCase):
    def test_pointwise_fft_promotes_half_precision_input_to_float32(self):
        operator = wave3d1.pointwise_op_3D(1, 1, 4, 4, 5).half()
        operator.conv = torch.nn.Identity()
        values = torch.randn(1, 1, 4, 4, 5, dtype=torch.float16)
        observed_dtypes = []
        original_rfftn = torch.fft.rfftn

        def record_fft_dtype(tensor, *args, **kwargs):
            observed_dtypes.append(tensor.dtype)
            return original_rfftn(tensor, *args, **kwargs)

        with patch.object(wave3d1.torch.fft, "rfftn", side_effect=record_fft_dtype):
            output = operator(values, 4, 4, 5)

        self.assertEqual(observed_dtypes, [torch.float32])
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 5))

    def test_coordinate_grid_is_cached_and_expanded_by_view(self):
        model = wave3d1.Uno3D_T10(in_width=6, width=1)
        shape = (2, 4, 3, 5, 1)

        first = model.get_grid(shape, torch.device("cpu"))
        second = model.get_grid(shape, torch.device("cpu"))

        self.assertEqual(tuple(first.shape), (2, 4, 3, 5, 5))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(first.stride(0), 0)
        x = torch.linspace(0.0, 2.0 * np.pi, 4)
        torch.testing.assert_close(first[0, :, 0, 0, 0], torch.sin(x))
        torch.testing.assert_close(first[0, :, 0, 0, 2], torch.cos(x))

    def test_pointwise_operator_skips_identity_interpolation(self):
        operator = wave3d1.pointwise_op_3D(1, 1, 4, 4, 4)
        values = torch.randn(1, 1, 4, 4, 4)

        with patch.object(
            wave3d1.torch.nn.functional,
            "interpolate",
            wraps=wave3d1.torch.nn.functional.interpolate,
        ) as interpolate:
            output = operator(values, 4, 4, 4)

        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))
        self.assertEqual(interpolate.call_count, 0)

    def test_gated_model_preserves_baseline_decoder_topology(self):
        baseline = wave3d1.Uno3D_T10(in_width=10, width=1)
        gated = Uno3D_T10_Gated(in_width=10, width=1)

        self.assertEqual(
            gated.conv7.conv.in_channels,
            baseline.conv7.conv.in_channels,
        )

    def test_zero_initialized_residual_gate_is_identity(self):
        skip = torch.randn(2, 4, 3, 5, 6)
        gate = torch.rand(1, 1, 3, 5, 1)
        alpha = torch.zeros(())

        modulated = Uno3D_T10_Gated._apply_residual_gate(skip, gate, alpha)

        torch.testing.assert_close(modulated, skip)

    def test_gate_alpha_receives_gradient_and_round_trips_with_gate_buffers(self):
        model = Uno3D_T10_Gated(in_width=10, width=1)
        skip = torch.ones(2, 4, 3, 5, 6)
        gate = torch.rand(1, 1, 3, 5, 1)

        loss = Uno3D_T10_Gated._apply_residual_gate(skip, gate, model.gate_alpha_0).sum()
        loss.backward()

        self.assertIsNotNone(model.gate_alpha_0.grad)
        self.assertNotEqual(float(model.gate_alpha_0.grad), 0.0)

        model.gate_alpha_0.data.fill_(0.25)
        model.gate_alpha_1.data.fill_(-0.10)
        model.set_gate_maps(
            torch.rand(1, 8, 8, 1, 1),
            torch.rand(1, 4, 4, 1, 1),
        )

        reloaded = Uno3D_T10_Gated(in_width=10, width=1)
        reloaded.set_gate_maps(
            torch.ones(1, 8, 8, 1, 1),
            torch.ones(1, 4, 4, 1, 1),
        )
        reloaded.load_state_dict(model.state_dict())

        torch.testing.assert_close(reloaded.gate_alpha_0, model.gate_alpha_0)
        torch.testing.assert_close(reloaded.gate_alpha_1, model.gate_alpha_1)
        torch.testing.assert_close(reloaded._gate_0, model._gate_0)
        torch.testing.assert_close(reloaded._gate_1, model._gate_1)

    def test_gated_forward_preserves_full_output_shape(self):
        model = Uno3D_T10_Gated(in_width=10, width=1)
        model.set_gate_maps(
            torch.rand(1, 8, 8, 1, 1),
            torch.rand(1, 4, 4, 1, 1),
        )
        inputs = torch.randn(1, 8, 8, 4, 5)

        def fake_operator(module):
            def forward(values, dim1, dim2, dim3):
                return torch.zeros(
                    values.shape[0],
                    module.conv.out_channels,
                    dim1,
                    dim2,
                    dim3,
                    dtype=values.dtype,
                )

            return forward

        with ExitStack() as patches:
            for module in (
                model.conv0,
                model.conv1,
                model.conv2,
                model.conv3,
                model.conv6,
                model.conv7,
                model.conv8,
            ):
                patches.enter_context(
                    patch.object(module, "forward", side_effect=fake_operator(module))
                )
            output = model(inputs)

        self.assertEqual(tuple(output.shape), (1, 8, 8, 4, 3))


if __name__ == "__main__":
    unittest.main()
