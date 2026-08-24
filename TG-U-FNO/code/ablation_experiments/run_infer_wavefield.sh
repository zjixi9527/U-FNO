#!/bin/bash
#SBATCH -A pi_zhengweiqi
#SBATCH --partition=gpu2Q,gpu4Q,gpu8Q
#SBATCH -q gpuq
#SBATCH -J ufno_wavefield_infer
#SBATCH -o logs/ufno_wavefield_infer_%j.out
#SBATCH -e logs/ufno_wavefield_infer_%j.err
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
cd ${WORKDIR}

# Sanity-check CUDA.
${PYTHON} - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
EOF

DATA_DIR=/public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d
STATIONS_PATH="${WORKDIR}/code/数据库构建/STATIONS"
GRD_PATH="${WORKDIR}/门源1.grd"
BASELINE_CKPT=${WORKDIR}/code/ablation_experiments/00_baseline/outputs_seed20260720/best.pt
GATE_CKPT=${WORKDIR}/code/ablation_experiments/06_terrain_gate/outputs_grd_seed20260720/best.pt
OUTPUT_DIR=${WORKDIR}/code/ablation_experiments/logs/infer_outputs

for p in "${STATIONS_PATH}" "${GRD_PATH}" "${BASELINE_CKPT}" "${GATE_CKPT}"; do
    if [[ ! -f "${p}" ]]; then
        echo "Required file does not exist: ${p}" >&2
        exit 1
    fi
done

${PYTHON} code/ablation_experiments/scripts/infer_wavefield.py \
    --data-dir ${DATA_DIR} \
    --stations "${STATIONS_PATH}" \
    --baseline-ckpt ${BASELINE_CKPT} \
    --gate-ckpt ${GATE_CKPT} \
    --gate-grd "${GRD_PATH}" \
    --test-files 91 \
    --samples-per-file 100 \
    --model-width 4 \
    --device cuda \
    --output-dir ${OUTPUT_DIR}

echo "Finished wavefield inference → ${OUTPUT_DIR}"
