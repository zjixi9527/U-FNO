#!/usr/bin/env python3
"""Compute slope-based terrain gate maps from a high-resolution GRD.

Usage
-----
    python scripts/compute_enhanced_gate.py 门源1.grd \\
        --output enhanced_gate.npy \\
        --output-shape 64 64

The output ``.npy`` file is a dict with keys ``gate_0`` and ``gate_1``,
suitable for loading in ``training.py``  via ``--enhanced-gate``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the ablation_experiments directory.
_script_dir = Path(__file__).resolve().parents[1]
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ablation_common.geometry import (
    compute_enhanced_terrain_gate_maps,
    load_grd_dem,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute slope-based terrain gate maps from a GMT DSAA GRD file."
    )
    parser.add_argument("grd_path", type=str, help="Path to the .grd file")
    parser.add_argument("--output", "-o", type=str, default="enhanced_gate.npy",
                        help="Output .npy path")
    parser.add_argument("--output-shape", type=int, nargs=2, default=(64, 64),
                        metavar=("H", "W"),
                        help="Target station-grid shape (default 64 64)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    grd_path = Path(args.grd_path)
    if not grd_path.is_file():
        print(f"ERROR: GRD file not found: {grd_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading DEM from {grd_path} ...")
    dem = load_grd_dem(str(grd_path))
    print(f"  DEM shape: {dem.shape}")
    print(f"  Elevation: [{dem.min():.1f}, {dem.max():.1f}] m, "
          f"mean={dem.mean():.1f}, std={dem.std():.1f}")

    print(f"Computing slope gate ({args.output_shape[0]}x{args.output_shape[1]}) ...")
    gate_maps = compute_enhanced_terrain_gate_maps(
        dem,
        output_grid_shape=tuple(args.output_shape),
    )

    output_path = Path(args.output)
    np.save(str(output_path), gate_maps)
    print(f"Saved enhanced gate to {output_path}")
    print(f"  gate_0: {gate_maps['gate_0'].shape} "
          f"range=[{gate_maps['gate_0'].min():.4f}, {gate_maps['gate_0'].max():.4f}]")
    print(f"  gate_1: {gate_maps['gate_1'].shape} "
          f"range=[{gate_maps['gate_1'].min():.4f}, {gate_maps['gate_1'].max():.4f}]")


if __name__ == "__main__":
    main()
