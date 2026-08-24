# Menyuan 3D U-FNO Paper Conditions

This repository contains the code for the paper's U-FNO ablation conditions
`00` to `07`. The pure FNO comparison condition `08_fno_baseline` has been
removed from this upload package.

## Repository Layout

```text
code/
  FNO_2D.py                          # shared normalizer/loss utilities required by wave3d1.py
  wave3d1.py                         # shared U-FNO model for conditions 00-05
  数据库构建/STATIONS                 # station grid used by training scripts
  database_build/                    # English-named copy of dataset builder inputs
  ablation_experiments/
    00_baseline/
    01_dem_distance/
    02_terrain_direction/
    03_local_components/
    04_direct_coordinates/
    05_geometry_only/
    06_terrain_gate/
    07_terrain_gate_shuffled/
    ablation_common/
    scripts/
    tests/
```

Start with `code/ablation_experiments/README.md` for the condition table,
commands, data requirements, and Slurm submission notes.

## Install

```bash
pip install -r code/ablation_experiments/requirements.txt
```

## Quick Test

```bash
cd code/ablation_experiments
PYTHONPATH=.:.. python -m unittest discover -s tests -v
```

The full training runs require the external HDF5 wavefield database and a GPU
environment.
