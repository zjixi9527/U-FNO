# U-FNO Paper Condition Code

This directory contains the code for the paper ablation conditions `00` to `07`.
The pure FNO architecture comparison `08_fno_baseline` is intentionally excluded
from this upload package.

## Included Conditions

| Directory | Input channels before internal coordinates | Model | Target |
|---|---:|---|---|
| `00_baseline` | 1 | U-FNO baseline | Global `CXX/CXY/CXZ` |
| `01_dem_distance` | 2 | U-FNO | Global `CXX/CXY/CXZ` |
| `02_terrain_direction` | 5 | U-FNO | Global `CXX/CXY/CXZ` |
| `03_local_components` | 5 | U-FNO | Local tangent/tangent/normal components |
| `04_direct_coordinates` | 3 | U-FNO | Global `CXX/CXY/CXZ` |
| `05_geometry_only` | 4 | U-FNO | Global `CXX/CXY/CXZ` |
| `06_terrain_gate` | 5 | Terrain-gated U-FNO | Global `CXX/CXY/CXZ` |
| `07_terrain_gate_shuffled` | 5 | Terrain-gated U-FNO with shuffled gate map | Global `CXX/CXY/CXZ` |

`Uno3D_T10` appends five coordinate channels inside `forward()`, so the model
`in_width` equals the table value plus five.

## Main Files

- `../wave3d1.py`: shared U-FNO model used by conditions `00` to `05`.
- `../FNO_2D.py`: shared utility dependency used by `wave3d1.py`
  (`GaussianNormalizer` and `LpLoss`).
- `ablation_common/wave3d1_gated.py`: terrain-gated U-FNO model used by
  conditions `06` and `07`.
- `ablation_common/data.py`: HDF5 loading and construction of the input
  features for each condition.
- `ablation_common/geometry.py`: source-position recovery, station-grid
  geometry, local basis construction, DEM loading, and terrain gate generation.
- `ablation_common/training.py`: shared training, validation, testing,
  checkpointing, and metric logging loop.
- `scripts/`: inference, gate-map, DEM debug, and plotting helper scripts.
- `tests/`: unit tests for data loading, geometry, training utilities, export,
  and inference benchmarking.

## Data Requirements

Training expects HDF5 files named:

```text
displacement_data1.h5
displacement_data2.h5
...
```

Each file should contain:

```text
source1 ... source100              # shape: (64, 64)
displacement1 ... displacement100  # shape: (>=50, 64, 64, 3)
```

The station file is expected at:

```text
../数据库构建/STATIONS
```

You can also pass it explicitly with `--stations`.

## Run One Condition

From this directory:

```bash
python 00_baseline/train.py \
  --data-dir /path/to/data-3d \
  --stations ../数据库构建/STATIONS \
  --train-files 1-90 \
  --validation-files 92-93 \
  --test-files 91 \
  --samples-per-file 100 \
  --time-steps 50 \
  --epochs 200 \
  --batch-size 1 \
  --learning-rate 0.005 \
  --scheduler-step 50 \
  --scheduler-gamma 0.5 \
  --model-width 4 \
  --seed 20260720 \
  --device cuda \
  --output-dir 00_baseline/outputs_seed20260720
```

Replace `00_baseline/train.py` with another condition directory to run a
different paper condition.

## Slurm Submission Scripts

The upload package keeps the Slurm scripts used for the paper conditions:

- `submit_ablation.sh`: submits `00_baseline` to `03_local_components`.
- `submit_source_encoding.sh`: submits `04_direct_coordinates` and
  `05_geometry_only`.
- `submit_terrain_gate.sh`: submits `06_terrain_gate` and
  `07_terrain_gate_shuffled`.

Before submitting on a different cluster, edit `WORK_ROOT`, `DATA_DIR`, and
`PYTHON_BIN` in the corresponding `run_*.sh` file.

## Reproducibility Notes

- Use the same train/validation/test file indices for every condition.
- Use the same seed, optimizer, learning rate schedule, epoch count, and batch
  size for all conditions.
- Condition `02` and condition `03` use the same input features; only the target
  coordinate system differs.
- Condition `05` uses the same distance and terrain-direction features as
  condition `02`, but removes the original source-cone channel.
- Condition `06` and condition `07` use the same gated U-FNO architecture; `07`
  shuffles the DEM gate spatially as a negative control.
- The excluded `08_fno_baseline` condition is the pure FNO architecture
  comparison and is not part of this GitHub upload package.

## Local Tests

Windows PowerShell:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path + ';' + (Resolve-Path '..').Path
python -m unittest discover -s tests -v
```

Linux/macOS:

```bash
PYTHONPATH=.:.. python -m unittest discover -s tests -v
```

Synthetic-data tests can run locally. Full training and CUDA smoke tests require
the real HDF5 data and the target GPU environment.
