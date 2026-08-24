#!/bin/bash
#SBATCH -A pi_zhengweiqi
#SBATCH --partition=gpu2Q,gpu4Q,gpu8Q
#SBATCH -q gpuq
#SBATCH -J ufno_wavefield_fig
#SBATCH -o logs/ufno_wavefield_fig_%j.out
#SBATCH -e logs/ufno_wavefield_fig_%j.err
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
EXPERIMENT_ROOT=${WORKDIR}/code/ablation_experiments
OUTPUT_DIR=${WORKDIR}/code/ablation_experiments/logs/wavefield_figures/00_06

# 画哪个样本（1 开始）：23 = 测试集里 FNO 误差最大的震源样本（rel-L2 0.1524），
# 也是 06 相对 FNO 改进最大的样本之一，最能体现误差差异。
SAMPLE_INDEX=23
# 代表时间步（按 1 开始，最大 50）。
TIME_INDICES="21,41"

STATIONS_PATH="${WORKDIR}/code/数据库构建/STATIONS"
BASELINE_CKPT=${EXPERIMENT_ROOT}/00_baseline/outputs_seed20260720/best.pt
GATE_CKPT=${EXPERIMENT_ROOT}/06_terrain_gate/outputs_grd_seed20260720/best.pt

for p in "${STATIONS_PATH}" "${BASELINE_CKPT}" "${GATE_CKPT}"; do
    if [[ ! -f "${p}" ]]; then
        echo "Required file does not exist: ${p}" >&2
        exit 1
    fi
done

${PYTHON} code/ablation_experiments/scripts/plot_wavefield_compare_00_06_sem.py \
    --experiment-root ${EXPERIMENT_ROOT} \
    --data-dir ${DATA_DIR} \
    --stations "${STATIONS_PATH}" \
    --test-file 91 \
    --sample-index ${SAMPLE_INDEX} \
    --time-indices "${TIME_INDICES}" \
    --model-width 0 \
    --device cuda \
    --dpi 600 \
    --save-svg \
    --save-arrays \
    --output-dir ${OUTPUT_DIR}

echo "Finished wavefield figure → ${OUTPUT_DIR}"
