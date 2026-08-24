"""Fair, shared training loop for all U-FNO ablation variants."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .data import AblationDataset, ExperimentVariant, list_hdf5_paths, load_station_grid
from .geometry import build_terrain_basis, compute_terrain_gate_maps, load_grd_dem


CODE_DIRECTORY = Path(__file__).resolve().parents[2]
DEFAULT_STATIONS_PATH = CODE_DIRECTORY / "数据库构建" / "STATIONS"


def parse_file_indices(expression: str) -> list[int]:
    """Parse an inclusive expression such as ``1-90,92,95-96``."""
    stripped = expression.strip()
    if not stripped:
        return []

    indices: list[int] = []
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty file-index token")
        if "-" not in token:
            value = int(token)
            if value <= 0:
                raise ValueError("file indices must be positive")
            indices.append(value)
            continue

        bounds = token.split("-")
        if len(bounds) != 2:
            raise ValueError(f"invalid file-index range: {token}")
        start, end = (int(value) for value in bounds)
        if start <= 0 or end <= 0:
            raise ValueError("file indices must be positive")
        if end < start:
            raise ValueError(f"descending file-index range is not allowed: {token}")
        indices.extend(range(start, end + 1))

    if len(indices) != len(set(indices)):
        raise ValueError("file-index expression contains duplicates")
    return indices


def relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    batch_size = prediction.shape[0]
    difference = (prediction - target).reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    numerator = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.clamp(torch.linalg.vector_norm(target_flat, dim=1), min=epsilon)
    return torch.mean(numerator / denominator)


def expand_static_features(features: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Expose static ``(B,H,W,C)`` features across time as a zero-stride view."""
    if features.ndim != 4:
        raise ValueError("static features must have shape (B, H, W, C)")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    return features.unsqueeze(-2).expand(*features.shape[:-1], time_steps, features.shape[-1])


class FileChunkRandomSampler(Sampler[int]):
    """Shuffle file-local chunks to reduce random HDF5 opens without losing coverage."""

    def __init__(
        self,
        *,
        file_count: int,
        samples_per_file: int,
        chunk_size: int,
        generator: torch.Generator,
    ) -> None:
        if file_count <= 0 or samples_per_file <= 0 or chunk_size <= 0:
            raise ValueError("file_count, samples_per_file, and chunk_size must be positive")
        self.file_count = file_count
        self.samples_per_file = samples_per_file
        self.chunk_size = min(chunk_size, samples_per_file)
        self.generator = generator

    def __len__(self) -> int:
        return self.file_count * self.samples_per_file

    def __iter__(self):
        chunks: list[list[int]] = []
        for file_index in range(self.file_count):
            local_order = torch.randperm(
                self.samples_per_file, generator=self.generator
            ).tolist()
            file_offset = file_index * self.samples_per_file
            for start in range(0, self.samples_per_file, self.chunk_size):
                chunks.append(
                    [file_offset + local_index for local_index in local_order[start : start + self.chunk_size]]
                )
        chunk_order = torch.randperm(len(chunks), generator=self.generator).tolist()
        for chunk_index in chunk_order:
            yield from chunks[chunk_index]


def rotate_local_to_global(local_wavefield: torch.Tensor, terrain_basis: torch.Tensor) -> torch.Tensor:
    """Rotate ``(B,H,W,T,local)`` values to global CXX/CXY/CXZ components."""
    if local_wavefield.ndim != 5 or local_wavefield.shape[-1] != 3:
        raise ValueError("local_wavefield must have shape (B, H, W, T, 3)")
    if terrain_basis.shape != local_wavefield.shape[1:3] + (3, 3):
        raise ValueError("terrain_basis is incompatible with the wavefield")
    return torch.einsum("hwkc,bhwtk->bhwtc", terrain_basis, local_wavefield)


def rotate_global_to_local(global_wavefield: torch.Tensor, terrain_basis: torch.Tensor) -> torch.Tensor:
    """Rotate ``(B,H,W,T,global)`` values to local terrain components."""
    if global_wavefield.ndim != 5 or global_wavefield.shape[-1] != 3:
        raise ValueError("global_wavefield must have shape (B, H, W, T, 3)")
    if terrain_basis.shape != global_wavefield.shape[1:3] + (3, 3):
        raise ValueError("terrain_basis is incompatible with the wavefield")
    return torch.einsum("hwkc,bhwtc->bhwtk", terrain_basis, global_wavefield)


@dataclass(frozen=True)
class TrainingConfig:
    variant: str
    data_directory: str
    stations_path: str
    train_files: str
    validation_files: str
    test_files: str
    samples_per_file: int
    time_steps: int
    distance_scale_m: float
    batch_size: int
    epochs: int
    learning_rate: float
    scheduler_step: int
    scheduler_gamma: float
    model_width: int
    seed: int
    num_workers: int
    io_chunk_size: int
    hdf5_cache_size: int
    checkpoint_interval: int
    resume_path: str
    device: str
    amp: bool
    enhanced_gate_path: str = ""
    enhanced_gate_grd: str = ""


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_supports_grad_scaler(model: torch.nn.Module) -> bool:
    """Return whether every trainable parameter is real-valued.

    PyTorch GradScaler cannot unscale complex gradients.  U-FNO contains
    ComplexFloat Fourier weights, so mixed precision must be disabled for
    this model even when ``--amp`` was requested.
    """
    return not any(parameter.is_complex() for parameter in model.parameters())


def total_file_size_bytes(paths: Iterable[str | Path]) -> int:
    """Return the total size of unique regular files."""
    unique_paths = {Path(path).resolve() for path in paths}
    return sum(path.stat().st_size for path in unique_paths if path.is_file())


def read_linux_process_tree_rss_bytes(
    root_process_id: int,
    *,
    proc_root: str | Path = "/proc",
    excluded_process_ids: Iterable[int] = (),
) -> int | None:
    """Sum resident memory for a Linux process and its current descendants."""
    root = Path(proc_root)
    if not (root / str(root_process_id)).is_dir():
        return None

    pending = [int(root_process_id)]
    visited: set[int] = set()
    excluded = {int(process_id) for process_id in excluded_process_ids}
    total_rss_bytes = 0
    while pending:
        process_id = pending.pop()
        if process_id in visited or process_id in excluded:
            continue
        visited.add(process_id)
        process_directory = root / str(process_id)
        try:
            status_lines = (process_directory / "status").read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            for line in status_lines:
                if line.startswith("VmRSS:"):
                    total_rss_bytes += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            # Processes can terminate between discovery and sampling.
            pass
        try:
            children_text = (
                process_directory / "task" / str(process_id) / "children"
            ).read_text(encoding="utf-8")
            for value in children_text.split():
                try:
                    pending.append(int(value))
                except ValueError:
                    continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return total_rss_bytes


def _monitor_process_tree_memory(
    root_process_id: int,
    interval_seconds: float,
    stop_event,
    result_queue,
) -> None:
    peak_bytes: int | None = None
    while not stop_event.is_set():
        current_bytes = read_linux_process_tree_rss_bytes(
            root_process_id,
            excluded_process_ids=(os.getpid(),),
        )
        if current_bytes is not None:
            peak_bytes = max(peak_bytes or 0, current_bytes)
        stop_event.wait(interval_seconds)
    result_queue.put(peak_bytes)


class PeakHostMemoryMonitor:
    """Poll aggregate RSS in a separate process, avoiding thread/fork hazards."""

    def __init__(self, *, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes: int | None = None
        self._context = multiprocessing.get_context("spawn")
        self._stop_event = self._context.Event()
        self._result_queue = self._context.Queue(maxsize=1)
        self._process: multiprocessing.Process | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("memory monitor has already been started")
        root_process_id = os.getpid()
        self._process = self._context.Process(
            target=_monitor_process_tree_memory,
            args=(
                root_process_id,
                self.interval_seconds,
                self._stop_event,
                self._result_queue,
            ),
            name="peak-host-memory-monitor",
            daemon=True,
        )
        self._process.start()

    def stop(self) -> int | None:
        if self._process is None:
            return self.peak_bytes
        self._stop_event.set()
        self._process.join(timeout=max(2.0, 4.0 * self.interval_seconds))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if not self._result_queue.empty():
            self.peak_bytes = self._result_queue.get_nowait()
        self._result_queue.close()
        return self.peak_bytes


def collect_hardware_information(device: torch.device) -> dict[str, object]:
    """Collect portable hardware/software identifiers for resource reporting."""
    cpu_model = platform.processor()
    host_total_memory_bytes: int | None = None
    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", maxsplit=1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.startswith("MemTotal:"):
                host_total_memory_bytes = int(line.split()[1]) * 1024
                break
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass

    information: dict[str, object] = {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "host_total_memory_bytes": host_total_memory_bytes,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        information.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return information


def shuffle_gate_maps_consistently(
    gate_0: torch.Tensor,
    gate_1: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle the fine gate once and derive the coarse gate from that result."""
    if gate_0.ndim != 5 or gate_0.shape[0] != 1 or gate_0.shape[-2:] != (1, 1):
        raise ValueError("gate_0 must have shape (1, H, W, 1, 1)")
    if gate_1.ndim != 5 or gate_1.shape[0] != 1 or gate_1.shape[-2:] != (1, 1):
        raise ValueError("gate_1 must have shape (1, H/2, W/2, 1, 1)")

    height, width = gate_0.shape[1:3]
    coarse_height, coarse_width = gate_1.shape[1:3]
    if height != 2 * coarse_height or width != 2 * coarse_width:
        raise ValueError("gate_1 spatial shape must be half of gate_0")

    flat_gate = gate_0[0, :, :, 0, 0].reshape(-1)
    permutation_device = flat_gate.device
    if generator is not None:
        generator_device = getattr(generator, "device", torch.device("cpu"))
        permutation_device = torch.device(generator_device)
    permutation = torch.randperm(
        flat_gate.numel(),
        generator=generator,
        device=permutation_device,
    )
    if permutation.device != flat_gate.device:
        permutation = permutation.to(flat_gate.device)
    shuffled_fine = flat_gate[permutation].reshape(height, width)
    shuffled_coarse = shuffled_fine.reshape(
        coarse_height, 2, coarse_width, 2
    ).mean(dim=(1, 3))
    return (
        shuffled_fine[None, :, :, None, None],
        shuffled_coarse[None, :, :, None, None],
    )


def _build_model(variant: ExperimentVariant, model_width: int) -> torch.nn.Module:
    if str(CODE_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(CODE_DIRECTORY))
    if variant.use_gated_model:
        from ablation_common.wave3d1_gated import Uno3D_T10_Gated as ModelClass
    else:
        from wave3d1 import Uno3D_T10 as ModelClass

    # Uno3D_T10 concatenates five internal coordinate channels in forward().
    model_arguments = {
        "in_width": variant.input_channels + 5,
        "width": model_width,
    }
    return ModelClass(**model_arguments, factor=1)


def _build_loader(
    dataset: AblationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    io_chunk_size: int,
) -> DataLoader:
    worker_generator = torch.Generator().manual_seed(seed + 1)
    sampler = None
    if shuffle:
        sampler = FileChunkRandomSampler(
            file_count=len(dataset.hdf5_paths),
            samples_per_file=dataset.samples_per_file,
            chunk_size=io_chunk_size,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=worker_generator,
        persistent_workers=num_workers > 0,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: ExperimentVariant,
    terrain_basis: torch.Tensor,
) -> tuple[float, float, list[float]]:
    model.eval()
    local_total = 0.0
    global_total = 0.0
    sample_count = 0
    per_sample_global: list[float] = []
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            features = expand_static_features(features, targets.shape[-2])
            predictions = model(features)

            if variant is ExperimentVariant.LOCAL_COMPONENTS:
                local_targets = rotate_global_to_local(targets, terrain_basis)
                local_loss = relative_l2_loss(predictions, local_targets)
                global_predictions = rotate_local_to_global(predictions, terrain_basis)
                global_targets = targets
            else:
                local_loss = relative_l2_loss(predictions, targets)
                global_predictions = predictions
                global_targets = targets
            global_loss = relative_l2_loss(global_predictions, global_targets)

            batch_count = features.shape[0]
            local_total += float(local_loss) * batch_count
            global_total += float(global_loss) * batch_count
            sample_count += batch_count

            differences = (global_predictions - global_targets).reshape(batch_count, -1)
            target_flat = global_targets.reshape(batch_count, -1)
            errors = torch.linalg.vector_norm(differences, dim=1) / torch.clamp(
                torch.linalg.vector_norm(target_flat, dim=1), min=1e-12
            )
            per_sample_global.extend(errors.detach().cpu().tolist())

    if sample_count == 0:
        raise ValueError("evaluation loader is empty")
    return local_total / sample_count, global_total / sample_count, per_sample_global


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_training(config: TrainingConfig, output_directory: str | Path) -> None:
    run_wall_start = time.perf_counter()
    variant = ExperimentVariant(config.variant)
    if config.epochs <= 0 or config.batch_size <= 0 or config.model_width <= 0:
        raise ValueError("epochs, batch_size, and model_width must be positive")
    if config.io_chunk_size <= 0 or config.hdf5_cache_size <= 0:
        raise ValueError("io_chunk_size and hdf5_cache_size must be positive")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if config.time_steps != 50:
        raise ValueError("Uno3D_T10 requires exactly 50 time steps")
    set_reproducible_seed(config.seed)
    host_memory_monitor = PeakHostMemoryMonitor()
    host_memory_monitor.start()

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_json(output_path / "config.json", asdict(config))

    station_grid = load_station_grid(config.stations_path)
    train_paths = list_hdf5_paths(config.data_directory, parse_file_indices(config.train_files))
    validation_paths = list_hdf5_paths(
        config.data_directory, parse_file_indices(config.validation_files)
    )
    test_indices = parse_file_indices(config.test_files)
    test_paths = list_hdf5_paths(config.data_directory, test_indices) if test_indices else []
    all_dataset_paths = tuple({*train_paths, *validation_paths, *test_paths})
    dataset_hdf5_disk_bytes = total_file_size_bytes(all_dataset_paths)
    auxiliary_input_paths = [Path(config.stations_path)]
    if config.enhanced_gate_path:
        auxiliary_input_paths.append(Path(config.enhanced_gate_path))
    if config.enhanced_gate_grd:
        auxiliary_input_paths.append(Path(config.enhanced_gate_grd))
    auxiliary_input_disk_bytes = total_file_size_bytes(auxiliary_input_paths)

    shared_terrain_basis = (
        build_terrain_basis(station_grid)
        if variant.uses_terrain_basis
        else np.broadcast_to(
            np.eye(3, dtype=np.float64),
            station_grid.shape[:2] + (3, 3),
        )
    )
    dataset_arguments = {
        "station_grid": station_grid,
        "variant": variant,
        "time_steps": config.time_steps,
        "samples_per_file": config.samples_per_file,
        "distance_scale_m": config.distance_scale_m,
        "terrain_basis": shared_terrain_basis,
        "hdf5_cache_size": config.hdf5_cache_size,
    }
    train_dataset = AblationDataset(hdf5_paths=train_paths, **dataset_arguments)
    validation_dataset = AblationDataset(hdf5_paths=validation_paths, **dataset_arguments)
    test_dataset = AblationDataset(hdf5_paths=test_paths, **dataset_arguments) if test_paths else None

    first_features, first_target = train_dataset[0]
    expected_channels = variant.input_channels
    if first_features.shape[-1] != expected_channels:
        raise RuntimeError(
            f"variant {variant.value} expected {expected_channels} input channels, "
            f"got {first_features.shape[-1]}"
        )
    print(
        f"variant={variant.value} input={tuple(first_features.shape)} "
        f"target={tuple(first_target.shape)} train_samples={len(train_dataset)} "
        f"validation_samples={len(validation_dataset)}"
    )
    # The shape probe opens one HDF5 file in the parent process. Close it
    # before Linux DataLoader workers fork so every worker owns its handles.
    train_dataset.close()

    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
        io_chunk_size=config.io_chunk_size,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
        io_chunk_size=config.io_chunk_size,
    )
    test_loader = (
        _build_loader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            pin_memory=pin_memory,
            io_chunk_size=config.io_chunk_size,
        )
        if test_dataset is not None
        else None
    )

    model = _build_model(variant, config.model_width).to(device)

    # Inject terrain gate maps for the gated model variant.
    if variant.use_gated_model:
        if config.enhanced_gate_path:
            # Load pre-computed enhanced gate from .npy.
            gate_npy = np.load(config.enhanced_gate_path, allow_pickle=True).item()
            gate_maps = {
                "gate_0": np.asarray(gate_npy["gate_0"], dtype=np.float32),
                "gate_1": np.asarray(gate_npy["gate_1"], dtype=np.float32),
            }
            print(f"enhanced_gate loaded from {config.enhanced_gate_path}")
        elif config.enhanced_gate_grd:
            # Load GRD and compute enhanced gate on the fly.
            dem = load_grd_dem(config.enhanced_gate_grd)
            from .geometry import compute_enhanced_terrain_gate_maps
            gate_maps = compute_enhanced_terrain_gate_maps(
                dem, output_grid_shape=station_grid.shape[:2],
            )
            print(f"enhanced_gate computed from GRD: {config.enhanced_gate_grd}")
        else:
            gate_maps = compute_terrain_gate_maps(station_grid)

        gate_0 = torch.tensor(gate_maps["gate_0"], dtype=torch.float32, device=device)
        gate_1 = torch.tensor(gate_maps["gate_1"], dtype=torch.float32, device=device)

        if variant is ExperimentVariant.TERRAIN_GATE_SHUFFLED:
            # Shuffle both gate maps spatially to break the correlation between
            # terrain complexity and spatial location, serving as a negative
            # control that tests whether the gate's spatial structure matters.
            gate_0, gate_1 = shuffle_gate_maps_consistently(gate_0, gate_1)

            print(f"terrain_gate SHUFFLED gate_0={tuple(gate_maps['gate_0'].shape)} "
                  f"gate_1={tuple(gate_maps['gate_1'].shape)}")

        if not hasattr(model, "set_gate_maps"):
            raise RuntimeError("gated variant requested but model does not support set_gate_maps")
        model.set_gate_maps(gate_0, gate_1)
        if variant is not ExperimentVariant.TERRAIN_GATE_SHUFFLED:
            print(
                f"terrain_gate gate_0={tuple(gate_maps['gate_0'].shape)} "
                f"gate_1={tuple(gate_maps['gate_1'].shape)} "
                f"gate_range=[{float(gate_maps['gate_0'].min()):.4f}, "
                f"{float(gate_maps['gate_0'].max()):.4f}]"
            )

    # Fused Adam is intentionally avoided: this model owns complex Fourier
    # parameters, which are unsupported by fused Adam in several PyTorch releases.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    use_amp = (
        config.amp
        and device.type == "cuda"
        and model_supports_grad_scaler(model)
    )
    if config.amp and device.type == "cuda" and not use_amp:
        print(
            "amp_disabled=complex model parameters are unsupported by "
            "PyTorch GradScaler; using float32"
        )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    terrain_basis = torch.tensor(
        train_dataset.terrain_basis, dtype=torch.float32, device=device
    )

    history_path = output_path / "history.csv"
    best_validation_loss = float("inf")
    best_checkpoint: dict | None = None
    best_validation_per_sample: np.ndarray | None = None
    total_compute_seconds = 0.0
    epoch_seconds_this_run: list[float] = []
    train_seconds_this_run: list[float] = []
    start_epoch = 1
    if config.resume_path:
        resume_path = Path(config.resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        if resume_checkpoint.get("variant") != variant.value:
            raise ValueError("resume checkpoint variant does not match this experiment")
        if resume_checkpoint.get("input_channels") != variant.input_channels:
            raise ValueError("resume checkpoint input channels do not match this experiment")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint["amp_scaler_state_dict"])
        random.setstate(resume_checkpoint["python_rng_state"])
        np.random.set_state(resume_checkpoint["numpy_rng_state"])
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        cuda_rng_states = resume_checkpoint.get("cuda_rng_state_all", [])
        if torch.cuda.is_available() and cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if isinstance(train_loader.sampler, FileChunkRandomSampler):
            sampler_state = resume_checkpoint.get("sampler_generator_state")
            if sampler_state is not None:
                train_loader.sampler.generator.set_state(sampler_state)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if start_epoch > config.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but requested epochs={config.epochs}"
            )
        best_validation_loss = float(
            resume_checkpoint["best_validation_global_relative_l2"]
        )
        total_compute_seconds = float(resume_checkpoint.get("total_compute_seconds", 0.0))
        best_path = output_path / "best.pt"
        best_per_sample_path = output_path / "best_validation_per_sample_relative_l2.txt"
        if not best_path.is_file() or not best_per_sample_path.is_file():
            raise FileNotFoundError("resume requires the existing best.pt and validation errors")
        best_checkpoint = torch.load(best_path, map_location="cpu")
        best_validation_per_sample = np.atleast_1d(np.loadtxt(best_per_sample_path))
        print(f"resuming_from={resume_path} start_epoch={start_epoch}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history_mode = "a" if start_epoch > 1 and history_path.is_file() else "w"
    with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        if history_mode == "w":
            writer.writerow(
                [
                    "epoch",
                    "train_relative_l2",
                    "validation_relative_l2",
                    "validation_global_relative_l2",
                    "learning_rate",
                    "train_seconds",
                    "epoch_seconds",
                    "train_samples_per_second",
                ]
            )

        for epoch in range(start_epoch, config.epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            model.train()
            running_loss = torch.zeros((), dtype=torch.float32, device=device)
            sample_count = 0
            for features, targets in train_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                features = expand_static_features(features, targets.shape[-2])
                training_targets = (
                    rotate_global_to_local(targets, terrain_basis)
                    if variant is ExperimentVariant.LOCAL_COMPONENTS
                    else targets
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    predictions = model(features)
                    loss = relative_l2_loss(predictions, training_targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_count = features.shape[0]
                running_loss = running_loss + loss.detach().float() * batch_count
                sample_count += batch_count

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - epoch_start
            scheduler.step()
            train_loss = float(running_loss.item()) / sample_count
            validation_loss, validation_global_loss, validation_per_sample = _evaluate(
                model,
                validation_loader,
                device=device,
                variant=variant,
                terrain_basis=terrain_basis,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start
            total_compute_seconds += epoch_seconds
            epoch_seconds_this_run.append(epoch_seconds)
            train_seconds_this_run.append(train_seconds)
            samples_per_second = sample_count / max(train_seconds, 1e-12)
            learning_rate = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    validation_loss,
                    validation_global_loss,
                    learning_rate,
                    train_seconds,
                    epoch_seconds,
                    samples_per_second,
                ]
            )
            history_file.flush()
            print(
                f"epoch={epoch}/{config.epochs} train={train_loss:.6f} "
                f"val={validation_loss:.6f} val_global={validation_global_loss:.6f} "
                f"train_s={train_seconds:.1f} samples_s={samples_per_second:.2f}"
            )

            checkpoint = {
                "epoch": epoch,
                "variant": variant.value,
                "input_channels": variant.input_channels,
                "model_state_dict": model.state_dict(),
                "validation_global_relative_l2": validation_global_loss,
                "config": asdict(config),
            }
            if validation_global_loss < best_validation_loss:
                best_validation_loss = validation_global_loss
                best_checkpoint = {
                    **checkpoint,
                    "model_state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                }
                best_validation_per_sample = np.asarray(validation_per_sample)
            if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
                sampler_generator_state = (
                    train_loader.sampler.generator.get_state()
                    if isinstance(train_loader.sampler, FileChunkRandomSampler)
                    else None
                )
                last_checkpoint = {
                    **checkpoint,
                    "best_validation_global_relative_l2": best_validation_loss,
                    "total_compute_seconds": total_compute_seconds,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "amp_scaler_state_dict": scaler.state_dict(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    ),
                    "sampler_generator_state": sampler_generator_state,
                }
                torch.save(last_checkpoint, output_path / "last.pt")
                if best_checkpoint is None or best_validation_per_sample is None:
                    raise RuntimeError("best checkpoint was not initialized")
                torch.save(best_checkpoint, output_path / "best.pt")
                np.savetxt(
                    output_path / "best_validation_per_sample_relative_l2.txt",
                    best_validation_per_sample,
                )

    summary = {
        "variant": variant.value,
        "best_validation_global_relative_l2": best_validation_loss,
        "total_compute_seconds": total_compute_seconds,
    }
    if test_loader is not None:
        best_checkpoint = torch.load(output_path / "best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_loss, test_global_loss, test_per_sample = _evaluate(
            model,
            test_loader,
            device=device,
            variant=variant,
            terrain_basis=terrain_basis,
        )
        summary.update(
            {"test_relative_l2": test_loss, "test_global_relative_l2": test_global_loss}
        )
        np.savetxt(output_path / "test_per_sample_relative_l2.txt", np.asarray(test_per_sample))
    peak_host_memory_bytes = host_memory_monitor.stop()
    checkpoint_paths = [output_path / "best.pt", output_path / "last.pt"]
    output_artifact_paths = [
        path
        for path in output_path.rglob("*")
        if path.is_file() and path.name != "summary.json"
    ]
    run_wall_seconds = time.perf_counter() - run_wall_start
    summary.update(
        {
            "dataset_hdf5_disk_bytes": dataset_hdf5_disk_bytes,
            "auxiliary_input_disk_bytes": auxiliary_input_disk_bytes,
            "total_input_disk_bytes": (
                dataset_hdf5_disk_bytes + auxiliary_input_disk_bytes
            ),
            "checkpoint_disk_bytes": total_file_size_bytes(checkpoint_paths),
            "output_artifacts_disk_bytes": total_file_size_bytes(
                output_artifact_paths
            ),
            "peak_host_memory_bytes": peak_host_memory_bytes,
            "peak_host_memory_bytes_this_run": peak_host_memory_bytes,
            "peak_host_memory_gib": (
                peak_host_memory_bytes / 2**30
                if peak_host_memory_bytes is not None
                else None
            ),
            "host_memory_measurement": (
                "peak aggregate RSS of the training process and current descendants"
                if peak_host_memory_bytes is not None
                else "unavailable: Linux /proc was not present"
            ),
            "run_wall_seconds": run_wall_seconds,
            "run_wall_seconds_this_run": run_wall_seconds,
            "start_epoch_this_run": start_epoch,
            "resumed_from_epoch": start_epoch - 1 if start_epoch > 1 else None,
            "epochs_completed_this_run": len(epoch_seconds_this_run),
            "mean_epoch_seconds_this_run": (
                float(np.mean(epoch_seconds_this_run))
                if epoch_seconds_this_run
                else None
            ),
            "total_epoch_seconds_this_run": float(sum(epoch_seconds_this_run)),
            "total_train_seconds_this_run": float(sum(train_seconds_this_run)),
            "resource_metric_scope": {
                "total_compute_seconds": (
                    "cumulative epoch wall time across resumed segments"
                ),
                "run_wall_seconds": "current process invocation only",
                "peak_host_memory": "current process invocation only",
                "peak_gpu_memory": "current process invocation only",
                "output_artifacts_disk_bytes": (
                    "all files in the output directory except summary.json"
                ),
            },
            "hardware": collect_hardware_information(device),
        }
    )
    if device.type == "cuda":
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
        summary.update(
            {
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                "peak_gpu_memory_bytes_this_run": peak_gpu_memory_bytes,
                "peak_gpu_memory_gib": peak_gpu_memory_bytes / 2**30,
            }
        )
    _save_json(output_path / "summary.json", summary)


def build_parser(variant: ExperimentVariant, default_output_directory: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train the {variant.value} U-FNO ablation variant."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing displacement_data*.h5")
    parser.add_argument("--stations", default=str(DEFAULT_STATIONS_PATH))
    parser.add_argument("--train-files", default="1-90")
    parser.add_argument("--validation-files", default="92-93")
    parser.add_argument("--test-files", default="")
    parser.add_argument("--samples-per-file", type=int, default=100)
    parser.add_argument(
        "--time-steps", type=int, default=50, help="Must remain 50 for Uno3D_T10"
    )
    parser.add_argument("--distance-scale-m", type=float, default=50_000.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--scheduler-step", type=int, default=50)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--model-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--io-chunk-size", type=int, default=16)
    parser.add_argument("--hdf5-cache-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--resume", default="", help="Path to a last.pt checkpoint to continue"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    parser.add_argument("--enhanced-gate", default="",
                        help="Path to a .npy file with pre-computed enhanced gate maps")
    parser.add_argument("--enhanced-gate-grd", default="",
                        help="Path to a .grd DEM file to compute enhanced gate on the fly")
    parser.add_argument("--output-dir", default=str(default_output_directory))
    return parser


def main_for_variant(variant: ExperimentVariant, experiment_directory: Path) -> None:
    parser = build_parser(variant, experiment_directory / "outputs")
    arguments = parser.parse_args()
    config = TrainingConfig(
        variant=variant.value,
        data_directory=arguments.data_dir,
        stations_path=arguments.stations,
        train_files=arguments.train_files,
        validation_files=arguments.validation_files,
        test_files=arguments.test_files,
        samples_per_file=arguments.samples_per_file,
        time_steps=arguments.time_steps,
        distance_scale_m=arguments.distance_scale_m,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        scheduler_step=arguments.scheduler_step,
        scheduler_gamma=arguments.scheduler_gamma,
        model_width=arguments.model_width,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        io_chunk_size=arguments.io_chunk_size,
        hdf5_cache_size=arguments.hdf5_cache_size,
        checkpoint_interval=arguments.checkpoint_interval,
        resume_path=arguments.resume,
        device=arguments.device,
        amp=arguments.amp,
        enhanced_gate_path=arguments.enhanced_gate,
        enhanced_gate_grd=arguments.enhanced_gate_grd,
    )
    run_training(config, arguments.output_dir)
