#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任一模型组合 vs SEM 波场对比图 —— 现场推理 + 绘图（超算运行）。

对比 2 个模型条件 vs SEM 真值（默认 FNO + 06 两个模型）。

每行 = 一个代表时间步 × 5 列：SEM 真值 | 模型1 | 模型2 | E(模型1) | E(模型2)。
只画 X 方向分量 (Vx)。

绘图规范：
    - 波场列颜色映射 RdBu_r（中心 0 对称），所有时刻统一色标
    - 误差列单侧色标 (YlOrRd)，所有时刻统一色标
    - 误差列左上角标注该时刻空间平均误差

本脚本直接在超算上从 HDF5 + checkpoint 推理并绘图，不依赖批量 .npy。
支持 U-FNO (Uno3D_T10 / Uno3D_T10_Gated) 和纯 FNO (FNO3D) 任意 checkpoint。

通用用法：
    python scripts/plot_wavefield_compare_00_06_sem.py \\
      --model-specs "Baseline U-FNO:00_baseline/outputs_seed20260720/best.pt" \\
                     "Terrain-Gate U-FNO:06_terrain_gate/outputs_grd_seed20260720/best.pt" \\
      --model-width 36 \\
      --data-dir <data-3d 目录> \\
      --output-dir wavefield_fig/00_06_sem
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import h5py
import numpy as np
import torch

COMPONENT_VX = 0  # CXX = X 方向分量 (easting)

# 默认 06/08 组合（可通过 --model-specs 覆盖）。
MODEL_SPECS = OrderedDict(
    [
        ("FNO Baseline", "08_fno_baseline/outputs_width36_seed20260720/best.pt"),
        ("Terrain-Gate U-FNO", "06_terrain_gate/outputs_grd_seed20260720/best.pt"),
    ]
)

COLUMN_LABELS = ["Baseline U-FNO", "Terrain-Gate U-FNO", "SEM truth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model vs SEM wavefield comparison (Vx, 6 rows x K cols)."
    )
    parser.add_argument(
        "--experiment-root",
        default="",
        help="ablation_experiments 目录。留空时自动识别。",
    )
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=None,
        help=(
            "要对比的模型条件，格式 '标签:相对路径'，可多个（顺序即列顺序）。"
            "省略时使用默认 00/06 组合。"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="/public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d",
        help="包含 displacement_data*.h5 的目录。",
    )
    parser.add_argument(
        "--stations",
        default="",
        help="STATIONS 文件路径。留空时自动查找。",
    )
    parser.add_argument(
        "--test-file",
        type=int,
        default=91,
        help="独立测试 HDF5 文件编号，默认 91。",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1,
        help="要预测的震源位置样本编号（1 开始），默认 1。",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=100,
        help="每个 HDF5 文件中的样本数。",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=50,
        help="模型时间步数，Uno3D_T10 必须为 50。",
    )
    parser.add_argument(
        "--time-indices",
        default="21,41",
        help="需要绘制的代表时间步，按 1 开始，以逗号分隔（每个=1 行）。",
    )
    parser.add_argument(
        "--distance-scale-m",
        type=float,
        default=50_000.0,
        help="几何距离归一化尺度，必须与训练一致。",
    )
    parser.add_argument(
        "--model-width",
        type=int,
        default=0,
        help="模型 width。0 表示优先读取 checkpoint config，缺失时使用 4。",
    )
    parser.add_argument(
        "--field-percentile",
        type=float,
        default=99.5,
        help="色标采用所有对比场绝对值的该百分位数，默认 99.5。",
    )
    parser.add_argument(
        "--output-dir",
        default="wavefield_fig/00_06_sem",
        help="结果保存目录。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推理设备。",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="除 PNG 外同时保存 SVG 矢量文件。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG 分辨率。",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=20,
        help="正式计时前的模型前向预热次数，默认 20。",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=100,
        help="单震源前向推理重复计时次数，默认 100。",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="只执行模型推理计时并保存 CSV/JSON，不绘图或保存波场数组。",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="同时保存 target/各模型预测/误差的完整 3 分量波场到 NPZ，"
             "每个场含 (H,W,T,3) 全量及 Vx/Vy/Vz 分量，便于本地自行绘图。",
    )
    return parser.parse_args()


def resolve_experiment_root(user_value: str) -> Path:
    candidates: list[Path] = []
    if user_value:
        candidates.append(Path(user_value).expanduser())

    script_path = Path(__file__).resolve()
    candidates.extend(
        [
            script_path.parent.parent,
            Path.cwd(),
            Path.cwd() / "code" / "ablation_experiments",
            Path.cwd() / "代码" / "ablation_experiments",
            Path("/public/home/hpc221253/pytorch_gpu/3d-menyuan-1/code/ablation_experiments"),
            Path("/public/home/hpc221253/pytorch_gpu/3d-menyuan-1/代码/ablation_experiments"),
            Path("/public/home/hpc221253/pytorch_gpu/3d-menyuan-1/ablation_experiments"),
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (
            (candidate / "ablation_common").is_dir()
            and (
                (candidate.parent / "wave3d1.py").is_file()
                or (candidate.parent.parent / "wave3d1.py").is_file()
            )
        ):
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "无法自动识别 ablation_experiments 目录。请使用 --experiment-root 指定。\n"
        f"已检查：\n{checked}"
    )


def configure_project_imports(experiment_root: Path) -> Path:
    code_root_candidates = [
        experiment_root.parent,
        experiment_root.parent.parent,
    ]
    code_root = None
    for candidate in code_root_candidates:
        if (candidate / "wave3d1.py").is_file():
            code_root = candidate
            break
    if code_root is None:
        raise FileNotFoundError(
            f"在 {code_root_candidates} 中未找到 wave3d1.py。"
        )

    for path in (experiment_root, code_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return code_root


def resolve_stations_path(user_value: str, experiment_root: Path, code_root: Path) -> Path:
    candidates: list[Path] = []
    if user_value:
        candidates.append(Path(user_value).expanduser())

    work_root = code_root.parent
    candidates.extend(
        [
            code_root / "数据库构建" / "STATIONS",
            code_root / "database_build" / "STATIONS",
            work_root / "数据库构建" / "STATIONS",
            work_root / "code" / "数据库构建" / "STATIONS",
            work_root / "代码" / "数据库构建" / "STATIONS",
            experiment_root / "STATIONS",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "无法找到 STATIONS 文件。请使用 --stations 指定实际路径。\n"
        f"已检查：\n{checked}"
    )


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_integer_list(expression: str, minimum: int, maximum: int) -> list[int]:
    values: list[int] = []
    for token in expression.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < minimum or value > maximum:
            raise ValueError(f"数值 {value} 超出允许范围 [{minimum}, {maximum}]。")
        values.append(value)
    if not values:
        raise ValueError("至少需要指定一个时间步。")
    return values


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        raise ValueError("checkpoint 中 model_state_dict 为空。")
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _infer_width(state_dict: dict[str, torch.Tensor], config: dict, default: int = 4) -> int:
    """从 checkpoint 的实际张量推导模型 width（比 config 里的 model_width 更可靠）。"""
    # FNO3D：lift 输出到 width；pointwise 权重形状第一维就是 width。
    pw = state_dict.get("layers.0.pointwise.weight")
    if pw is not None and pw.ndim == 5:
        return int(pw.shape[0])
    # Uno3D / gated：fc0 是 Linear(in_width*2, width)，第二维即 width。
    fc0 = state_dict.get("fc0.weight")
    if fc0 is not None and fc0.ndim == 2:
        return int(fc0.shape[0])
    # 兜底：config。
    return int(config.get("model_width", default))


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    forced_width: int,
    device: torch.device,
):
    from wave3d1 import Uno3D_T10

    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    config = checkpoint.get("config", {})
    input_channels = int(checkpoint.get("input_channels", config.get("input_channels", 0)))
    if input_channels <= 0:
        fc_weight = state_dict.get("fc.weight")
        if fc_weight is None:
            raise KeyError("无法从 checkpoint 读取 input_channels 或 fc.weight。")
        total_in_width = int(fc_weight.shape[1])
        input_channels = total_in_width - 5
    total_in_width = input_channels + 5

    # 宽度优先从 checkpoint 的实际张量形状推导（最可靠），
    # 否则回退到 config 里的 model_width（可能因为复用了别的跑法而失真，如 36）。
    width = int(forced_width) if forced_width > 0 else _infer_width(state_dict, config, default=4)
    if width <= 0:
        raise ValueError(f"无法确定模型 width，forced_width={forced_width}，config={config}")
    fc0_w = state_dict.get("fc0.weight")
    print(
        f"  [宽度] variant={checkpoint.get('variant','?')} forced_width={forced_width} "
        f"config.model_width={config.get('model_width','?')} "
        f"fc0.weight.shape={None if fc0_w is None else tuple(fc0_w.shape)} "
        f"→ 推断 width={width}",
        flush=True,
    )

    is_gated = (
        "_gate_0" in state_dict
        or "gate_alpha_0" in state_dict
        or "_gate_1" in state_dict
        or "gate_alpha_1" in state_dict
    )

    # 纯 FNO：checkpoint 键为 layers.N.spectral.weights.N（Uno3D 用的是 conv*.weights1..4）。
    is_fno = "layers.0.spectral.weights.0" in state_dict or "project_output.weight" in state_dict

    if is_fno:
        # 层数 = FourierLayer 数量，所有层共享同一个 scale……（此处不依赖 scale，直接按实测键构造）
        weight_key = "layers.0.spectral.weights.0"
        layer_count = 4
        if weight_key in state_dict:
            layer_count = sum(
                1 for key in state_dict if key.startswith("layers.") and key.endswith(".pointwise.weight")
            )
        from ablation_common.fno3d import FNO3D
        model = FNO3D(in_width=total_in_width, width=width, layer_count=layer_count)
    elif is_gated:
        from ablation_common.wave3d1_gated import Uno3D_T10_Gated
        model = Uno3D_T10_Gated(in_width=total_in_width, width=width, factor=1)
        gate_0 = state_dict.get("_gate_0")
        gate_1 = state_dict.get("_gate_1")
        if gate_0 is None or gate_1 is None:
            # 部分旧 checkpoint 未将 gate 存入 state_dict，回退到从 GRD 计算。
            print("  警告：checkpoint 中没有持久化 gate 图，将重新计算。", flush=True)
        else:
            model.set_gate_maps(gate_0, gate_1)
    else:
        model = Uno3D_T10(in_width=total_in_width, width=width, factor=1)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    important_missing = [
        key for key in missing
        if not key.endswith("num_batches_tracked")
    ]
    if important_missing or unexpected:
        raise RuntimeError(
            "checkpoint 与模型结构不一致。\n"
            f"missing={important_missing}\n"
            f"unexpected={unexpected}"
        )

    model = model.to(device)
    model.eval()

    metadata = {
        "variant": checkpoint.get("variant", "unknown"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "input_channels": input_channels,
        "total_in_width": total_in_width,
        "model_width": width,
        "is_gated": is_gated,
        "model_type": "fno3d" if is_fno else ("uno3d_gated" if is_gated else "uno3d"),
        "validation_global_relative_l2": checkpoint.get("validation_global_relative_l2", None),
    }
    if is_gated:
        metadata["gate_alpha_0"] = float(model.gate_alpha_0.detach().cpu())
        metadata["gate_alpha_1"] = float(model.gate_alpha_1.detach().cpu())
    return model, metadata


def make_dataset_for_checkpoint(
    checkpoint: dict[str, Any],
    *,
    hdf5_path: Path,
    station_grid: np.ndarray,
    time_steps: int,
    samples_per_file: int,
    distance_scale_m: float,
):
    from ablation_common.data import AblationDataset, ExperimentVariant
    from ablation_common.geometry import build_terrain_basis

    variant = ExperimentVariant(checkpoint.get("variant", "baseline"))

    if getattr(variant, "uses_terrain_basis", False):
        terrain_basis = build_terrain_basis(station_grid)
    else:
        terrain_basis = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            station_grid.shape[:2] + (3, 3),
        )

    dataset = AblationDataset(
        hdf5_paths=[hdf5_path],
        station_grid=station_grid,
        variant=variant,
        time_steps=time_steps,
        samples_per_file=samples_per_file,
        distance_scale_m=distance_scale_m,
        terrain_basis=terrain_basis,
        hdf5_cache_size=1,
    )
    return dataset, variant


def benchmark_model_forward(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    *,
    device: torch.device,
    warmup_runs: int,
    repeat_runs: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Measure model-only, batch-size-one forward latency.

    Model loading, HDF5 access, feature construction and device transfer are
    deliberately excluded. CUDA synchronization makes every recorded interval
    a completed forward pass rather than an asynchronous kernel launch.
    """
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be nonnegative")
    if repeat_runs <= 0:
        raise ValueError("repeat_runs must be positive")

    model.eval()
    prediction: torch.Tensor | None = None
    with torch.inference_mode():
        for _ in range(warmup_runs):
            prediction = model(model_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_seconds: list[float] = []
        for _ in range(repeat_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = model(model_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds.append(time.perf_counter() - start)

    if prediction is None:
        raise RuntimeError("benchmark did not produce a prediction")
    values = np.asarray(elapsed_seconds, dtype=np.float64)
    statistics = {
        "scope": "model_forward_only",
        "batch_size": int(model_input.shape[0]),
        "warmup_runs": int(warmup_runs),
        "repeat_runs": int(repeat_runs),
        "mean_seconds": float(values.mean()),
        "std_seconds": float(values.std(ddof=1)) if repeat_runs > 1 else 0.0,
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "individual_seconds": values.tolist(),
    }
    return prediction, statistics


def load_sample_and_predict(
    *,
    checkpoint_path: Path,
    hdf5_path: Path,
    station_grid: np.ndarray,
    sample_index_1based: int,
    time_steps: int,
    samples_per_file: int,
    distance_scale_m: float,
    forced_width: int,
    device: torch.device,
    benchmark_warmup: int,
    benchmark_repeats: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from ablation_common.training import expand_static_features

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} 中缺少 model_state_dict。")

    model, metadata = build_model_from_checkpoint(
        checkpoint, forced_width=forced_width, device=device,
    )
    dataset, _ = make_dataset_for_checkpoint(
        checkpoint,
        hdf5_path=hdf5_path,
        station_grid=station_grid,
        time_steps=time_steps,
        samples_per_file=samples_per_file,
        distance_scale_m=distance_scale_m,
    )

    dataset_index = sample_index_1based - 1
    features, target = dataset[dataset_index]
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)

    features = features.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_batch = target.unsqueeze(0).to(device=device, dtype=torch.float32)
    model_input = expand_static_features(features, target_batch.shape[-2])

    prediction, benchmark = benchmark_model_forward(
        model,
        model_input,
        device=device,
        warmup_runs=benchmark_warmup,
        repeat_runs=benchmark_repeats,
    )

    prediction_np = prediction[0].detach().cpu().numpy().astype(np.float32)
    target_np = target_batch[0].detach().cpu().numpy().astype(np.float32)

    if prediction_np.shape != target_np.shape:
        raise ValueError(
            f"预测与目标形状不一致：prediction={prediction_np.shape}, "
            f"target={target_np.shape}"
        )
    if prediction_np.ndim != 4 or prediction_np.shape[-1] != 3:
        raise ValueError(
            "预期波场形状为 (H,W,T,3)，实际为 " f"{prediction_np.shape}"
        )

    metadata["checkpoint"] = str(checkpoint_path)
    metadata["checkpoint_disk_bytes"] = checkpoint_path.stat().st_size
    metadata["inference_seconds"] = benchmark["mean_seconds"]
    metadata["inference_benchmark"] = benchmark
    return prediction_np, target_np, metadata


def robust_symmetric_limit(arrays: list[np.ndarray], percentile: float) -> float:
    values = np.concatenate(
        [np.abs(array.astype(np.float64)).ravel() for array in arrays]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, percentile))
    if not math.isfinite(limit) or limit <= 1e-12:
        limit = float(np.max(finite)) if finite.size else 1.0
    return max(limit, 1e-12)


def save_figure(fig: plt.Figure, path_without_suffix: Path, dpi: int, save_svg: bool) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.linalg.norm(
        prediction.astype(np.float64) - target.astype(np.float64)
    )
    denominator = max(np.linalg.norm(target.astype(np.float64)), 1e-12)
    return float(numerator / denominator)


def _spatial_mean_error(error: np.ndarray) -> float:
    """误差场的空间平均（选取有限值），用于误差图角标标注。"""
    finite = error.astype(np.float64)[np.isfinite(error)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def plot_wavefield_comparison(
    *,
    fields: dict[str, np.ndarray],        # {label: displacement (H,W,T,3)}
    time_indices_1based: list[int],
    sample_index: int,
    output_dir: Path,
    field_percentile: float,
    dpi: int,
    save_svg: bool,
    component: int = COMPONENT_VX,
    ref_label: str = "SEM",
    model_labels: list[str] | None = None,
) -> None:
    """Draw a time-by-model comparison with a dynamic number of columns.

    列布局：
        SEM | 各模型波场 | 各模型误差
    波场列用 RdBu_r 对称色标；误差列用单侧色标。两者均在所有时刻内共享
    同一条色标，保证跨行横向可公平比较。
    """
    if not model_labels:
        raise ValueError("model_labels must contain at least one model")
    time_indices_0based = [t - 1 for t in time_indices_1based]
    nrows = len(time_indices_1based)
    ncols = 1 + 2 * len(model_labels)

    field_panels: list[tuple[str, np.ndarray]] = []
    for label in model_labels:
        field_panels.append((label, fields[label][:, :, :, component]))
    ref_arr = fields[ref_label][:, :, :, component]

    # 波场/误差共用的全局色标边界（跨所有时刻一致，保证可横向公平比较）。
    wave_arrays = [field for _, field in field_panels] + [ref_arr]
    max_amp = robust_symmetric_limit(wave_arrays, field_percentile)

    error_arrays = [np.abs(field - ref_arr) for _, field in field_panels]
    max_err = robust_symmetric_limit(error_arrays, 100.0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.0 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for r, t in enumerate(time_indices_0based):
        # 第 0 列：参考波场（SEM 真值）。
        norm_wave = Normalize(vmin=-max_amp, vmax=max_amp)
        ref_img = axes[r, 0].imshow(
            ref_arr[:, :, t],
            origin="lower",
            cmap="RdBu_r",
            norm=norm_wave,
            interpolation="nearest",
        )
        axes[r, 0].set_title(ref_label, fontsize=12)
        axes[r, 0].set_ylabel(f"t={time_indices_1based[r] - 1}", fontsize=11)
        axes[r, 0].set_xlabel("Grid column", fontsize=10)
        axes[r, 0].tick_params(labelsize=8)

        for c, (label, field) in enumerate(field_panels):
            axes[r, c + 1].imshow(
                field[:, :, t],
                origin="lower",
                cmap="RdBu_r",
                norm=norm_wave,
                interpolation="nearest",
            )
            axes[r, c + 1].set_title(label, fontsize=12)
            axes[r, c + 1].set_xlabel("Grid column", fontsize=10)
            axes[r, c + 1].tick_params(labelsize=8)

        # 误差列（第 1+len(field_panels) .. 列尾）。
        for c, (_, field) in enumerate(field_panels):
            err = error_arrays[c][:, :, t]
            delim = 1 + len(field_panels) + c
            err_img = axes[r, delim].imshow(
                err,
                origin="lower",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max_err,
                interpolation="nearest",
            )
            e_t = _spatial_mean_error(err)
            axes[r, delim].set_title(
                f"E({model_labels[c]})  mean={e_t:.4f}", fontsize=11
            )
            axes[r, delim].set_xlabel("Grid column", fontsize=10)
            axes[r, delim].tick_params(labelsize=8)

    # colorbar：波场（对称）与误差（单侧）各一条，纵向分别盖住各自列组。
    fig.colorbar(
        ref_img,
        ax=axes[:, : 1 + len(model_labels)].ravel().tolist(),
        shrink=0.85,
        fraction=0.046,
        pad=0.03,
    )
    fig.colorbar(
        err_img,
        ax=axes[:, 1 + len(model_labels):].ravel().tolist(),
        shrink=0.85,
        fraction=0.046,
        pad=0.03,
    )

    fig.suptitle(
        f"Sample {sample_index}: X-component (Vx) wavefield — "
        f"{' / '.join(model_labels)} vs SEM truth",
        fontsize=15,
    )
    save_figure(
        fig,
        output_dir / f"wavefield_compare_sample{sample_index:03d}_Vx",
        dpi,
        save_svg,
    )


def main() -> None:
    args = parse_args()

    if args.time_steps != 50:
        raise ValueError("当前 Uno3D_T10 模型要求 time-steps=50。")
    if not (0.0 < args.field_percentile <= 100.0):
        raise ValueError("field-percentile 必须位于 (0,100]。")

    time_indices_1based = parse_integer_list(
        args.time_indices, minimum=1, maximum=args.time_steps
    )
    if len(time_indices_1based) < 1:
        raise ValueError("--time-indices 至少需要 1 个时间步。")

    experiment_root = resolve_experiment_root(args.experiment_root)
    code_root = configure_project_imports(experiment_root)
    from ablation_common.data import load_station_grid

    stations_path = resolve_stations_path(args.stations, experiment_root, code_root)
    station_grid = load_station_grid(stations_path)

    data_dir = Path(args.data_dir).expanduser().resolve()
    hdf5_path = data_dir / f"displacement_data{args.test_file}.h5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"测试数据不存在：{hdf5_path}")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"experiment_root={experiment_root}")
    print(f"code_root={code_root}")
    print(f"data={hdf5_path}")
    print(f"stations={stations_path}")
    print(f"device={device}")
    print(f"sample_index={args.sample_index}")
    print(f"time_indices_1based={time_indices_1based}")

    if args.model_specs is not None:
        model_specs: OrderedDict[str, str] = OrderedDict()
        for spec in args.model_specs:
            if ":" in spec:
                label, _, rel_path = spec.partition(":")
            else:
                label = rel_path = spec
            label = label.strip()
            rel_path = rel_path.strip()
            if not label or not rel_path:
                raise ValueError(f"--model-specs 项的 '标签:路径' 格式有误：{spec!r}")
            model_specs[label] = rel_path
    else:
        model_specs = MODEL_SPECS
    if len(model_specs) < 1:
        raise ValueError("至少需要一个模型条件。")
    model_labels = list(model_specs.keys())

    predictions: OrderedDict[str, np.ndarray] = OrderedDict()
    target_reference: np.ndarray | None = None
    metadata_all: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for label, rel_path in model_specs.items():
        checkpoint_path = experiment_root / rel_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

        print(f"\n加载 {label}: {checkpoint_path}", flush=True)
        prediction, target, metadata = load_sample_and_predict(
            checkpoint_path=checkpoint_path,
            hdf5_path=hdf5_path,
            station_grid=station_grid,
            sample_index_1based=args.sample_index,
            time_steps=args.time_steps,
            samples_per_file=args.samples_per_file,
            distance_scale_m=args.distance_scale_m,
            forced_width=args.model_width,
            device=device,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_repeats=args.benchmark_repeats,
        )

        if target_reference is None:
            target_reference = target
        elif not np.array_equal(target_reference, target):
            max_diff = float(
                np.max(np.abs(target_reference.astype(np.float64) - target.astype(np.float64)))
            )
            if max_diff > 1e-7:
                raise RuntimeError(
                    f"{label} 读取到的 target 与第一个模型不一致，max_diff={max_diff:.6e}。"
                )

        predictions[label] = prediction
        metadata_all[label] = metadata
        vx_rel_l2 = relative_l2(prediction[:, :, :, COMPONENT_VX], target[:, :, :, COMPONENT_VX])
        print(
            f"  variant={metadata['variant']} epoch={metadata['epoch']} "
            f"Vx relL2={vx_rel_l2:.8f} "
            f"inference={metadata['inference_seconds']:.6f}s/source"
        )
        if metadata.get("is_gated"):
            print(
                f"  gate_alpha_0={metadata['gate_alpha_0']:.7f}, "
                f"gate_alpha_1={metadata['gate_alpha_1']:.7f}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if target_reference is None:
        raise RuntimeError("没有完成任何模型预测。")

    fields: dict[str, np.ndarray] = OrderedDict()
    for label, prediction in predictions.items():
        fields[label] = prediction
    fields["SEM truth"] = target_reference

    if not args.benchmark_only:
        plot_wavefield_comparison(
            fields=fields,
            time_indices_1based=time_indices_1based,
            sample_index=args.sample_index,
            output_dir=output_dir,
            field_percentile=args.field_percentile,
            dpi=args.dpi,
            save_svg=args.save_svg,
            model_labels=model_labels,
            ref_label="SEM truth",
        )

    if args.save_arrays and not args.benchmark_only:
        # 保存全 3 分量波场 + 各模型误差场，便于本地自行绘图。
        # 每个场均存完整 (H,W,T,3) 版本，以及分量的 Vx/Vy/Vz 快捷数组。
        payload: dict[str, np.ndarray] = {}
        for key, field in fields.items():
            safe_key = "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in key
            )
            payload[safe_key] = field.astype(np.float32)              # (H,W,T,3)
            for ci, cname in enumerate(("Vx", "Vy", "Vz")):
                payload[f"{safe_key}_{cname}"] = field[:, :, :, ci].astype(np.float32)

        for label in model_labels:
            safe_label = "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label
            )
            err = np.abs(fields[label] - target_reference).astype(np.float32)  # (H,W,T,3)
            payload[f"E_{safe_label}"] = err
            for ci, cname in enumerate(("Vx", "Vy", "Vz")):
                payload[f"E_{safe_label}_{cname}"] = err[:, :, :, ci]

        np.savez_compressed(
            output_dir / f"wavefields_sample{args.sample_index:03d}.npz",
            **payload,
        )

    metrics_lines = []
    for label, prediction in predictions.items():
        vx_rel_l2 = relative_l2(prediction[:, :, :, COMPONENT_VX], target_reference[:, :, :, COMPONENT_VX])
        metrics_lines.append(f"{label}: Vx relative_L2 = {vx_rel_l2:.8f}")
    summary = (
        "Wavefield comparison models/SEM (Vx)\n"
        f"experiment_root={experiment_root}\n"
        f"data_file={hdf5_path}\n"
        f"sample_index={args.sample_index}\n"
        f"time_indices_1based={time_indices_1based}\n"
        f"model_labels={model_labels}\n"
        f"device={device}\n"
        "\n"
        + "\n".join(metrics_lines)
        + "\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    benchmark_lines = [
        "model,model_type,mean_seconds,std_seconds,median_seconds,p95_seconds,"
        "warmup_runs,repeat_runs,batch_size,checkpoint_disk_bytes"
    ]
    for label, metadata in metadata_all.items():
        benchmark = metadata["inference_benchmark"]
        benchmark_lines.append(
            f'"{label}",{metadata["model_type"]},'
            f'{benchmark["mean_seconds"]:.9f},{benchmark["std_seconds"]:.9f},'
            f'{benchmark["median_seconds"]:.9f},{benchmark["p95_seconds"]:.9f},'
            f'{benchmark["warmup_runs"]},{benchmark["repeat_runs"]},'
            f'{benchmark["batch_size"]},{metadata["checkpoint_disk_bytes"]}'
        )
    (output_dir / "inference_benchmark.csv").write_text(
        "\n".join(benchmark_lines) + "\n", encoding="utf-8"
    )

    print("\n完成。输出：")
    if not args.benchmark_only:
        print(f"  {output_dir / f'wavefield_compare_sample{args.sample_index:03d}_Vx.png'}")
    print(f"  {output_dir / 'summary.txt'}")
    print(f"  {output_dir / 'checkpoint_metadata.json'}")
    print(f"  {output_dir / 'inference_benchmark.csv'}")
    if args.save_arrays and not args.benchmark_only:
        print(f"  {output_dir / f'wavefields_sample{args.sample_index:03d}.npz'}")


if __name__ == "__main__":
    main()
