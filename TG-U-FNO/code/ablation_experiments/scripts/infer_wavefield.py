"""Inference script for wavefield prediction comparison (00 baseline vs 06 terrain gate).

Generates per-sample prediction arrays for all test samples, suitable for
downloading and plotting wavefield comparison figures.

Usage on the supercomputer:

    cd /public/home/hpc221253/pytorch_gpu/3d-menyuan-1/code/ablation_experiments
    python scripts/infer_wavefield.py \\
        --data-dir /public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d \\
        --stations /public/home/hpc221253/pytorch_gpu/3d-menyuan-1/code/数据库构建/STATIONS \\
        --baseline-ckpt code/ablation_experiments/00_baseline/outputs_seed20260720/best.pt \\
        --gate-ckpt code/ablation_experiments/06_terrain_gate/outputs_grd_seed20260720/best.pt \\
        --gate-grd /public/home/hpc221253/pytorch_gpu/3d-menyuan-1/门源1.grd \\
        --test-files 91 \\
        --output-dir ./infer_outputs

Outputs:
    <output_dir>/
        baseline_sample_predictions.npy   (N_samples, 50, 64, 64, 3)
        gate_sample_predictions.npy       (N_samples, 50, 64, 64, 3)
        baseline_errors_abs.npy           (N_samples, 50, 64, 64, 3) — |pred-true|
        gate_errors_abs.npy               (N_samples, 50, 64, 64, 3)
        test_targets.npy                  (N_samples, 50, 64, 64, 3) — ground truth
        sample_l2_errors.npy              (N_samples,) — baseline mean L2 per sample
        config.json                       — inference configuration

After downloading, use the plotting script:
    python scripts/plot_wavefield_comparison.py \\
        --pred-dir <downloaded_output_dir> \\
        --sem-data <path_to_sem_npy_or_h5> \\
        --output wavefield_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ablation_common.data import load_station_grid  # noqa: E402
from ablation_common.geometry import (  # noqa: E402
    build_geometry_features,
    build_terrain_basis,
    compute_enhanced_terrain_gate_maps,
    load_grd_dem,
    recover_source_grid_position,
    source_grid_to_physical,
)


def parse_file_indices(expression: str) -> list[int]:
    stripped = expression.strip()
    if not stripped:
        return []
    indices = []
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" not in token:
            indices.append(int(token))
        else:
            start, end = token.split("-")
            indices.extend(range(int(start), int(end) + 1))
    return indices


def load_hdf5_sources(
    data_dir: str | Path,
    file_indices: list[int],
    samples_per_file: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load source data and displacement targets from HDF5 files.

    Returns:
        sources: (N_files * samples_per_file, 64, 64) raw source amplitudes
        displacement: (N_files * samples_per_file, 50, 64, 64, 3)
        sample_numbers: list of sample numbers (1-based within each file)
    """
    import h5py

    sources = []
    displacements = []
    sample_numbers = []

    for file_idx in file_indices:
        h5_path = Path(data_dir) / f"displacement_data{file_idx}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"missing data file: {h5_path}")

        with h5py.File(h5_path, "r") as hf:
            for i in range(1, samples_per_file + 1):
                src = np.array(hf[f"source{i}"], dtype=np.float32)
                disp = np.array(hf[f"displacement{i}"], dtype=np.float32)

                sources.append(src)
                # Keep only first 50 timesteps, shape: (50, 64, 64, 3)
                disp_50 = disp[:50, :, :, :]
                displacements.append(disp_50)
                sample_numbers.append(i)

    return np.array(sources), np.array(displacements), sample_numbers


def build_model(variant: str, input_channels: int, model_width: int = 4):
    """Build the inference model from variant name."""
    if variant == "baseline":
        # Import Uno3D_T10 from wave3d1
        code_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(code_dir))
        from wave3d1 import Uno3D_T10

        return Uno3D_T10(
            in_width=input_channels + 5,
            width=model_width,
            factor=1,
        )
    elif variant == "terrain_gate":
        from ablation_common.wave3d1_gated import Uno3D_T10_Gated

        return Uno3D_T10_Gated(
            in_width=input_channels + 5,
            width=model_width,
            factor=1,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")


def predict_sample(
    model: torch.nn.Module,
    source: np.ndarray,
    source_xyz: np.ndarray,
    station_grid: np.ndarray,
    terrain_basis: np.ndarray,
    device: torch.device,
    variant: str = "baseline",
    gate_0: np.ndarray | None = None,
    gate_1: np.ndarray | None = None,
) -> np.ndarray:
    """Run single-sample inference following the exact training-time feature pipeline.

    Returns: displacement prediction of shape (50, 64, 64, 3) in global
    CXX/CXY/CXZ order (matches the HDF5 ``displacement`` target).
    """
    model.eval()
    timesteps = 50

    # Recover the source physical location and build the geometry features
    # exactly as AblationDataset.__getitem__ does.
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    geometry = build_geometry_features(
        station_grid,
        source_xyz,
        distance_scale_m=50_000.0,
        terrain_basis=terrain_basis,
    )

    # Feature channels: [source, distance, terrain_direction] for gated;
    # [source] for baseline. Matches ExperimentVariant.input_channels.
    source_feature = source[..., None].astype(np.float32)  # (64, 64, 1)
    if variant == "terrain_gate":
        distance_feature = geometry["distance"][..., None].astype(np.float32)
        feature_stack = np.concatenate(
            [source_feature, distance_feature, geometry["terrain_direction"]],
            axis=-1,
        )  # (64, 64, 5)
    else:
        feature_stack = source_feature  # (64, 64, 1)

    features_5d = np.broadcast_to(
        feature_stack[None, :, :, None, :],
        (1, 64, 64, timesteps, feature_stack.shape[-1]),
    ).copy()  # (1, 64, 64, 50, C)

    # The model's forward() concatenates 5 internal coordinate channels, so the
    # saved weights were trained on (B, H, W, T, C) inputs with
    # C == input_channels (matching variant.input_channels).
    input_tensor = torch.from_numpy(features_5d).to(device)

    with torch.inference_mode():
        prediction = model(input_tensor)  # (1, 64, 64, 50, 3)

    return prediction.cpu().numpy()[0]  # (64, 64, 50, 3)


def main() -> None:
    import time

    parser = argparse.ArgumentParser(
        description="Inference script for wavefield prediction comparison"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing displacement_data*.h5")
    parser.add_argument("--stations", required=True,
                        help="Path to STATIONS file")
    parser.add_argument("--baseline-ckpt", required=True,
                        help="Path to 00_baseline best.pt")
    parser.add_argument("--gate-ckpt", required=True,
                        help="Path to 06_terrain_gate best.pt")
    parser.add_argument("--gate-grd", default="",
                        help="DEM GRD used at training time (门源1.grd); "
                             "if omitted, a station-grid gradient gate is used")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save inference results")
    parser.add_argument("--test-files", default="91",
                        help="Test file indices (e.g., '91' or '91,94-96')")
    parser.add_argument("--samples-per-file", type=int, default=100)
    parser.add_argument("--model-width", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--start-sample", type=int, default=0,
                        help="Start from this sample index (0-based)")
    parser.add_argument("--end-sample", type=int, default=None,
                        help="End sample index (exclusive, 0-based). Defaults to all.")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}")

    # Load station grid and shared terrain basis.
    print("Loading station grid...")
    station_grid = load_station_grid(args.stations)
    terrain_basis = build_terrain_basis(station_grid)
    print(f"  station_grid={station_grid.shape} terrain_basis={terrain_basis.shape}")

    # Build the test file indices.
    import h5py
    test_indices = parse_file_indices(args.test_files)

    # Load all source data
    print("Loading source data...")
    sources, displacements, sample_numbers = load_hdf5_sources(
        args.data_dir, test_indices, args.samples_per_file
    )
    n_samples = len(sources)
    print(f"loaded {n_samples} test samples from files {test_indices}")

    # Slice if requested
    end_sample = args.end_sample if args.end_sample is not None else n_samples
    sources = sources[args.start_sample:end_sample]
    targets = displacements[args.start_sample:end_sample]  # (N, 50, 64, 64, 3)
    n_samples = end_sample - args.start_sample

    # Build models
    print("Building baseline model...")
    baseline_model = build_model("baseline", input_channels=1, model_width=args.model_width)
    baseline_ckpt = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline_model = baseline_model.to(device)
    print(f"  baseline loaded from {args.baseline_ckpt}")

    print("Building terrain-gate model...")
    gate_model = build_model("terrain_gate", input_channels=5, model_width=args.model_width)
    gate_ckpt = torch.load(args.gate_ckpt, map_location="cpu", weights_only=False)
    gate_model.load_state_dict(gate_ckpt["model_state_dict"])
    gate_model = gate_model.to(device)

    # Compute the exact gate maps used at training time (enhanced slope gate
    # from the DEM GRD, or the station-grid gradient gate as a fallback).
    if args.gate_grd:
        dem = load_grd_dem(args.gate_grd)
        gate_maps = compute_enhanced_terrain_gate_maps(
            dem, output_grid_shape=station_grid.shape[:2],
        )
        print(f"  enhanced gate maps computed from GRD: {args.gate_grd}")
    else:
        from ablation_common.geometry import compute_terrain_gate_maps
        gate_maps = compute_terrain_gate_maps(station_grid)
        print("  enhanced gate GRD not given; using station-grid gradient gate")
    g0 = torch.from_numpy(gate_maps["gate_0"]).to(device)
    g1 = torch.from_numpy(gate_maps["gate_1"]).to(device)
    gate_model.set_gate_maps(g0, g1)
    print(f"  gate_0 shape={tuple(gate_maps['gate_0'].shape)} "
          f"range=[{float(gate_maps['gate_0'].min()):.4f}, {float(gate_maps['gate_0'].max()):.4f}]")
    print(f"  gate_1 shape={tuple(gate_maps['gate_1'].shape)} "
          f"range=[{float(gate_maps['gate_1'].min()):.4f}, {float(gate_maps['gate_1'].max()):.4f}]")

    # Run inference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference for {n_samples} samples...")
    all_baseline_preds = np.zeros((n_samples, 50, 64, 64, 3), dtype=np.float32)
    all_gate_preds = np.zeros((n_samples, 50, 64, 64, 3), dtype=np.float32)
    all_sample_l2 = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        t0 = time.perf_counter()
        source = sources[i]
        target = targets[i]

        # Recover the source physical position exactly as training does.
        horizontal_index, depth_index = recover_source_grid_position(source)
        source_xyz = source_grid_to_physical(horizontal_index, depth_index)

        # Baseline prediction
        baseline_pred = predict_sample(
            baseline_model, source, source_xyz, station_grid, terrain_basis,
            device, variant="baseline",
        )

        # Terrain-gate prediction
        gate_pred = predict_sample(
            gate_model, source, source_xyz, station_grid, terrain_basis,
            device, variant="terrain_gate",
            gate_0=gate_maps["gate_0"], gate_1=gate_maps["gate_1"],
        )

        # Predictions are (64, 64, 50, 3); transpose to (50, 64, 64, 3) to
        # match the HDF5 target format.
        baseline_pred = baseline_pred.transpose(2, 0, 1, 3)
        gate_pred = gate_pred.transpose(2, 0, 1, 3)

        all_baseline_preds[i] = baseline_pred
        all_gate_preds[i] = gate_pred

        # Compute per-sample L2 error for both
        baseline_l2 = np.linalg.norm(baseline_pred - target) / max(np.linalg.norm(target), 1e-12)
        gate_l2 = np.linalg.norm(gate_pred - target) / max(np.linalg.norm(target), 1e-12)
        all_sample_l2[i] = baseline_l2

        elapsed = time.perf_counter() - t0
        print(f"  sample {i+1}/{n_samples} — baseline L2={baseline_l2:.6f} gate L2={gate_l2:.6f} ({elapsed:.1f}s)")

    # Save results
    print("\nSaving results...")
    np.save(output_dir / "baseline_sample_predictions.npy", all_baseline_preds)
    np.save(output_dir / "gate_sample_predictions.npy", all_gate_preds)
    np.save(output_dir / "test_targets.npy", targets)

    # Compute error maps: |pred - truth|
    baseline_errors = np.abs(all_baseline_preds - targets)
    gate_errors = np.abs(all_gate_preds - targets)
    np.save(output_dir / "baseline_errors_abs.npy", baseline_errors)
    np.save(output_dir / "gate_errors_abs.npy", gate_errors)

    np.save(output_dir / "sample_l2_errors.npy", all_sample_l2)

    # Save config
    config = {
        "data_dir": str(args.data_dir),
        "stations": str(args.stations),
        "baseline_ckpt": str(args.baseline_ckpt),
        "gate_ckpt": str(args.gate_ckpt),
        "gate_grd": str(args.gate_grd),
        "test_files": args.test_files,
        "file_indices": test_indices,
        "samples_per_file": args.samples_per_file,
        "start_sample": args.start_sample,
        "end_sample": end_sample,
        "n_samples": n_samples,
        "model_width": args.model_width,
        "device": str(device),
        "grid_shape": [64, 64],
        "time_steps": 50,
        "gate_maps": {
            "gate_0_shape": list(gate_maps["gate_0"].shape),
            "gate_1_shape": list(gate_maps["gate_1"].shape),
            "gate_0_min": float(gate_maps["gate_0"].min()),
            "gate_0_max": float(gate_maps["gate_0"].max()),
            "gate_1_min": float(gate_maps["gate_1"].min()),
            "gate_1_max": float(gate_maps["gate_1"].max()),
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary statistics over all samples.
    mean_baseline_l2 = float(np.mean(all_sample_l2))
    print("\n=== Summary ===")
    print(f"Baseline mean L2: {mean_baseline_l2:.6f}")
    print(f"Gate mean L2: {np.mean(gate_l2):.6f}")
    print(f"Improvement: {(1 - np.mean(gate_l2) / mean_baseline_l2) * 100:.2f}%")
    print(f"\nOutput saved to: {output_dir}")
    print("Files:")
    for p in sorted(output_dir.glob("*")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
