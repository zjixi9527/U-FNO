#!/bin/bash
#SBATCH -A pi_zhengweiqi
#SBATCH --partition=gpu8Q,gpu4Q,gpu2Q
#SBATCH -q gpuq
#SBATCH -J ufno_07_gate_shuffled
#SBATCH -o logs/ufno_07_gate_shuffled_%j.out
#SBATCH -e logs/ufno_07_gate_shuffled_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "================================"
echo "JOB ID: ${SLURM_JOB_ID:-manual}"
echo "HOST: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================"

PYTHON=/public/home/hpc221253/.conda/envs/mypy3/bin/python
WORKDIR=/public/home/hpc221253/pytorch_gpu/3d-menyuan-1
GRD_PATH="${WORKDIR}/门源1.grd"
cd ${WORKDIR}
if [[ ! -f "${GRD_PATH}" ]]; then
    echo "Enhanced terrain GRD does not exist: ${GRD_PATH}" >&2
    exit 1
fi

${PYTHON} - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
EOF

VARIANT=07_terrain_gate_shuffled
TRAIN_SCRIPT=code/ablation_experiments/${VARIANT}/train.py
OUTPUT_DIR=${WORKDIR}/code/ablation_experiments/${VARIANT}/outputs_grd_seed20260720
DATA_DIR=/public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d
STATIONS_PATH="${WORKDIR}/code/数据库构建/STATIONS"
if [[ ! -f "${STATIONS_PATH}" ]]; then
    echo "STATIONS file does not exist: ${STATIONS_PATH}" >&2
    exit 1
fi

${PYTHON} ${TRAIN_SCRIPT} \
    --data-dir ${DATA_DIR} \
    --stations "${STATIONS_PATH}" \
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
    --num-workers 4 \
    --io-chunk-size 16 \
    --hdf5-cache-size 8 \
    --checkpoint-interval 10 \
    --enhanced-gate-grd "${GRD_PATH}" \
    --device cuda \
    --output-dir ${OUTPUT_DIR}

echo "Finished ${VARIANT}"
