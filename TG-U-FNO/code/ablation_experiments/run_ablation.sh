#!/bin/bash
#SBATCH -A pi_zhengweiqi
#SBATCH --partition=gpu8Q,gpu4Q,gpu2Q
#SBATCH -q gpuq
#SBATCH --job-name=ufno_ablation
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

WORK_ROOT=/public/home/hpc221253/pytorch_gpu/3d-menyuan-1
DATA_DIR=/public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d
PYTHON_BIN=/public/home/hpc221253/.conda/envs/mypy3/bin/python

if [[ -d "$WORK_ROOT/代码/ablation_experiments" ]]; then
    CODE_DIR="$WORK_ROOT/代码"
elif [[ -d "$WORK_ROOT/code/ablation_experiments" ]]; then
    CODE_DIR="$WORK_ROOT/code"
elif [[ -d "$WORK_ROOT/ablation_experiments" ]]; then
    CODE_DIR="$WORK_ROOT"
else
    echo "Cannot find ablation_experiments below $WORK_ROOT" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter does not exist or is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CODE_DIR/数据库构建/STATIONS" ]]; then
    echo "STATIONS file does not exist: $CODE_DIR/数据库构建/STATIONS" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$CODE_DIR/ablation_experiments"

VARIANTS=(
    00_baseline
    01_dem_distance
    02_terrain_direction
    03_local_components
)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEED=20260720
OUTPUT_DIR="$VARIANT/outputs_seed${SEED}"
set --
if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
    set -- --resume "$OUTPUT_DIR/last.pt"
    echo "resume_checkpoint=$OUTPUT_DIR/last.pt"
fi

echo "variant=$VARIANT"
echo "host=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

srun --unbuffered "$PYTHON_BIN" "$VARIANT/train.py" \
    --data-dir "$DATA_DIR" \
    --stations "$CODE_DIR/数据库构建/STATIONS" \
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
    --seed "$SEED" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --io-chunk-size 16 \
    --hdf5-cache-size 8 \
    --checkpoint-interval 10 \
    "$@" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"
