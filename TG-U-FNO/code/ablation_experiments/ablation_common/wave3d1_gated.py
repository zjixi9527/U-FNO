"""Uno3D_T10 with topography-gated skip connections.

``Uno3D_T10_Gated`` inherits all architecture from ``Uno3D_T10`` and
overrides only ``forward()`` to apply per-level terrain gate maps
before concatenating encoder features into the decoder path.

Gate maps are registered as non-trainable buffers that must be set
via ``set_gate_maps()`` before training or evaluation.

The gated model preserves the baseline decoder topology.  Each gate is
applied as a zero-initialized residual modulation, so the model starts
exactly at the ungated U-FNO behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_module_directory = Path(__file__).resolve().parents[2]
if str(_module_directory) not in sys.path:
    sys.path.insert(0, str(_module_directory))

# Import the base class and building blocks from the project root.
from wave3d1 import Uno3D_T10  # noqa: E402


class Uno3D_T10_Gated(Uno3D_T10):
    """U-FNO with topography-gated U-Net skip connections.

    Two gate levels are supported:

    * ``gate_0`` — shape ``(1, H, W, 1, 1)``, applied to the
      conv0 → conv7 skip at the original spatial resolution.
    * ``gate_1`` — shape ``(1, H//2, W//2, 1, 1)``, applied to the
      conv1 → conv6 skip at half resolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "_gate_0", torch.ones(1, 1, 1, 1, 1), persistent=True
        )
        self.register_buffer(
            "_gate_1", torch.ones(1, 1, 1, 1, 1), persistent=True
        )
        self.gate_alpha_0 = torch.nn.Parameter(torch.zeros(()))
        self.gate_alpha_1 = torch.nn.Parameter(torch.zeros(()))
        self._use_gate = False

    def set_gate_maps(
        self,
        gate_0: torch.Tensor,
        gate_1: torch.Tensor,
    ) -> None:
        """Store pre-computed terrain gate maps on the correct device."""
        device = self._gate_0.device
        self._gate_0 = gate_0.to(device)
        self._gate_1 = gate_1.to(device)
        self._use_gate = True

    @property
    def use_gate(self) -> bool:
        return self._use_gate

    @staticmethod
    def _resize_gate(
        gate: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Resize a 5-D gate ``(1, H, W, 1, 1)`` to ``(1, H', W', 1, 1)``.

        Uses bilinear interpolation so the gate smoothly adapts to the
        U-Net skip-connection feature map at its particular resolution.
        """
        # Squeeze the trailing singletons to get (1, 1, H, W) for interpolate.
        gate_4d = gate.squeeze(-1).squeeze(-1).unsqueeze(1)  # (1, 1, H, W)
        resized = F.interpolate(
            gate_4d, size=(target_h, target_w), mode="bilinear", align_corners=False,
        )
        # Return channels-first broadcast shape (1, 1, H', W', 1).
        return resized[:, :, :, :, None]

    @staticmethod
    def _apply_residual_gate(
        skip: torch.Tensor,
        gate: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``skip * (1 + alpha * tanh(gate))`` without in-place mutation."""
        return skip * (1.0 + alpha * torch.tanh(gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional terrain-gated skip connections."""
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)

        D1, D2, D3 = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3 * D1 / 4), int(3 * D2 / 4), D3)
        x_c1 = self.conv1(x_c0, D1 // 2, D2 // 2, D3)
        x_c2 = self.conv2(x_c1, D1 // 4, D2 // 4, int(1.0 * D3))
        x_c3 = self.conv3(x_c2, D1 // 4, D2 // 4, int(1.0 * D3))

        x_c6 = self.conv6(x_c3, D1 // 2, D2 // 2, int(1.0 * D3))

        # Gated skip 1: conv1 → conv6 at 1/2 resolution.
        if self._use_gate:
            g1 = self._resize_gate(self._gate_1, x_c1.shape[2], x_c1.shape[3])
            gated_x_c1 = self._apply_residual_gate(x_c1, g1, self.gate_alpha_1)
            x_c6 = torch.cat([x_c6, gated_x_c1], dim=1)
        else:
            x_c6 = torch.cat([x_c6, x_c1], dim=1)

        x_c7 = self.conv7(x_c6, int(3 * D1 / 4), int(3 * D2 / 4), D3)

        # Gated skip 2: conv0 → conv7 at 3/4 resolution.
        # x_c0 shape: (B, 2*factor*width, 3*D1//4, 3*D2//4, D3)  e.g. (B, 2W, 48, 48, 50)
        # gate_0 shape: (1, H, W, 1, 1)  e.g. (1, 64, 64, 1, 1) → resize to (1, 48, 48, 1, 1)
        if self._use_gate:
            g0 = self._resize_gate(self._gate_0, x_c0.shape[2], x_c0.shape[3])
            gated_x_c0 = self._apply_residual_gate(x_c0, g0, self.gate_alpha_0)
            x_c7 = torch.cat([x_c7, gated_x_c0], dim=1)
        else:
            x_c7 = torch.cat([x_c7, x_c0], dim=1)

        x_c8 = self.conv8(x_c7, D1, D2, D3)
        x_c8 = torch.cat([x_c8, x_fc0], dim=1)

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)
        x_fc1 = F.gelu(x_c8)

        x_velocity = self.fc2_x(x_fc1)
        y_velocity = self.fc2_y(x_fc1)
        z_velocity = self.fc2_z(x_fc1)
        return torch.cat((x_velocity, y_velocity, z_velocity), dim=-1)
