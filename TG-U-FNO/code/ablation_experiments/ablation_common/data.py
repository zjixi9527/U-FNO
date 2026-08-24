"""Lazy HDF5 dataset for the U-FNO ablation variants."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (
    build_geometry_features,
    build_terrain_basis,
    normalize_source_coordinates,
    recover_source_grid_position,
    source_grid_to_physical,
)


class ExperimentVariant(str, Enum):
    BASELINE = "baseline"
    DEM_DISTANCE = "dem_distance"
    TERRAIN_DIRECTION = "terrain_direction"
    LOCAL_COMPONENTS = "local_components"
    DIRECT_COORDINATES = "direct_coordinates"
    GEOMETRY_ONLY = "geometry_only"
    TERRAIN_GATE = "terrain_gate"
    TERRAIN_GATE_SHUFFLED = "terrain_gate_shuffled"

    @property
    def input_channels(self) -> int:
        return {
            ExperimentVariant.BASELINE: 1,
            ExperimentVariant.DEM_DISTANCE: 2,
            ExperimentVariant.TERRAIN_DIRECTION: 5,
            ExperimentVariant.LOCAL_COMPONENTS: 5,
            ExperimentVariant.DIRECT_COORDINATES: 3,
            ExperimentVariant.GEOMETRY_ONLY: 4,
            ExperimentVariant.TERRAIN_GATE: 5,
            ExperimentVariant.TERRAIN_GATE_SHUFFLED: 5,
        }[self]

    @property
    def uses_terrain_basis(self) -> bool:
        return self in {
            ExperimentVariant.TERRAIN_DIRECTION,
            ExperimentVariant.LOCAL_COMPONENTS,
            ExperimentVariant.GEOMETRY_ONLY,
            ExperimentVariant.TERRAIN_GATE,
            ExperimentVariant.TERRAIN_GATE_SHUFFLED,
        }

    @property
    def use_gated_model(self) -> bool:
        return self in {
            ExperimentVariant.TERRAIN_GATE,
            ExperimentVariant.TERRAIN_GATE_SHUFFLED,
        }

    @property
    def use_fno_model(self) -> bool:
        return False


def load_station_grid(
    station_path: str | Path,
    *,
    grid_shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Parse SPECFEM ``STATIONS`` into ``[easting, northing, vertical]``.

    This project uses ``USE_SOURCES_RECEIVERS_Z = .true.`` semantics: the
    sixth field is the receiver's absolute vertical coordinate in metres.
    """
    path = Path(station_path)
    if not path.is_file():
        raise FileNotFoundError(f"station file does not exist: {path}")

    coordinates: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as station_file:
        for line_number, line in enumerate(station_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 6:
                raise ValueError(f"invalid STATIONS line {line_number}: {stripped}")
            expected_station_name = f"CZ{line_number}"
            if fields[0] != expected_station_name:
                raise ValueError(
                    "STATIONS order mismatch on line "
                    f"{line_number}: expected {expected_station_name}, found {fields[0]}"
                )
            try:
                coordinates.append((float(fields[3]), float(fields[2]), float(fields[5])))
            except ValueError as error:
                raise ValueError(f"non-numeric STATIONS coordinates on line {line_number}") from error

    expected_count = int(np.prod(grid_shape))
    if len(coordinates) != expected_count:
        raise ValueError(
            f"station count mismatch: expected {expected_count}, found {len(coordinates)}"
        )
    return np.asarray(coordinates, dtype=np.float64).reshape(*grid_shape, 3)


def list_hdf5_paths(data_directory: str | Path, file_indices: Iterable[int]) -> list[Path]:
    directory = Path(data_directory)
    paths = [directory / f"displacement_data{index}.h5" for index in file_indices]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(f"missing HDF5 files: {preview}{suffix}")
    return paths


class AblationDataset(Dataset):
    """Read existing merged HDF5 data and construct ablation inputs on demand."""

    def __init__(
        self,
        *,
        hdf5_paths: Iterable[str | Path],
        station_grid: np.ndarray,
        variant: ExperimentVariant | str,
        time_steps: int = 50,
        samples_per_file: int = 100,
        distance_scale_m: float = 50_000.0,
        terrain_basis: np.ndarray | None = None,
        hdf5_cache_size: int = 8,
    ) -> None:
        self.hdf5_paths = tuple(Path(path) for path in hdf5_paths)
        if not self.hdf5_paths:
            raise ValueError("hdf5_paths must not be empty")
        if time_steps <= 0 or samples_per_file <= 0:
            raise ValueError("time_steps and samples_per_file must be positive")
        if distance_scale_m <= 0.0:
            raise ValueError("distance_scale_m must be positive")
        if hdf5_cache_size <= 0:
            raise ValueError("hdf5_cache_size must be positive")

        self.station_grid = np.asarray(station_grid, dtype=np.float64)
        if self.station_grid.ndim != 3 or self.station_grid.shape[-1] != 3:
            raise ValueError("station_grid must have shape (H, W, 3)")
        self.variant = ExperimentVariant(variant)
        self.time_steps = int(time_steps)
        self.samples_per_file = int(samples_per_file)
        self.distance_scale_m = float(distance_scale_m)
        self.hdf5_cache_size = int(hdf5_cache_size)
        if terrain_basis is not None:
            self.terrain_basis = np.asarray(terrain_basis, dtype=np.float64)
        elif self.variant.uses_terrain_basis:
            self.terrain_basis = build_terrain_basis(self.station_grid)
        else:
            self.terrain_basis = np.broadcast_to(
                np.eye(3, dtype=np.float64),
                self.station_grid.shape[:2] + (3, 3),
            )
        expected_basis_shape = self.station_grid.shape[:2] + (3, 3)
        if self.terrain_basis.shape != expected_basis_shape:
            raise ValueError(f"terrain_basis must have shape {expected_basis_shape}")
        self._hdf5_handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self._source_xyz_cache: dict[int, np.ndarray] = {}

    def __getstate__(self) -> dict:
        """Avoid copying process-owned HDF5 handles into DataLoader workers."""
        state = self.__dict__.copy()
        state["_hdf5_handles"] = OrderedDict()
        return state

    def _get_hdf5_handle(self, path: Path) -> h5py.File:
        handle = self._hdf5_handles.get(path)
        if handle is not None and handle.id.valid:
            self._hdf5_handles.move_to_end(path)
            return handle
        if handle is not None:
            self._hdf5_handles.pop(path)

        handle = h5py.File(path, "r")
        self._hdf5_handles[path] = handle
        if len(self._hdf5_handles) > self.hdf5_cache_size:
            _, oldest_handle = self._hdf5_handles.popitem(last=False)
            oldest_handle.close()
        return handle

    def close(self) -> None:
        for handle in self._hdf5_handles.values():
            handle.close()
        self._hdf5_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.hdf5_paths) * self.samples_per_file

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        file_index, local_index = divmod(index, self.samples_per_file)
        sample_number = local_index + 1
        source_key = f"source{sample_number}"
        displacement_key = f"displacement{sample_number}"
        hdf5_path = self.hdf5_paths[file_index]

        handle = self._get_hdf5_handle(hdf5_path)
        if source_key not in handle:
            raise KeyError(f"{source_key} is missing from {hdf5_path}")
        if displacement_key not in handle:
            raise KeyError(f"{displacement_key} is missing from {hdf5_path}")
        source = np.asarray(handle[source_key], dtype=np.float32)
        displacement = np.asarray(handle[displacement_key], dtype=np.float32)

        spatial_shape = self.station_grid.shape[:2]
        if source.shape != spatial_shape:
            raise ValueError(
                f"{source_key} shape {source.shape} does not match station grid {spatial_shape}"
            )
        if displacement.ndim != 4 or displacement.shape[1:3] != spatial_shape:
            raise ValueError(
                f"{displacement_key} must have shape (T, {spatial_shape[0]}, "
                f"{spatial_shape[1]}, 3), got {displacement.shape}"
            )
        if displacement.shape[0] < self.time_steps or displacement.shape[-1] != 3:
            raise ValueError(
                f"{displacement_key} does not contain {self.time_steps} steps and 3 components"
            )

        target = np.transpose(displacement[: self.time_steps], (1, 2, 0, 3))
        source_feature = source[..., None]
        feature_fields: list[np.ndarray] = []
        if self.variant in {
            ExperimentVariant.BASELINE,
            ExperimentVariant.DEM_DISTANCE,
            ExperimentVariant.TERRAIN_DIRECTION,
            ExperimentVariant.LOCAL_COMPONENTS,
            ExperimentVariant.TERRAIN_GATE,
            ExperimentVariant.TERRAIN_GATE_SHUFFLED,
        }:
            feature_fields.append(source_feature)

        if self.variant not in {
            ExperimentVariant.BASELINE,
        }:
            source_xyz = self._source_xyz_cache.get(index)
            if source_xyz is None:
                horizontal_index, depth_index = recover_source_grid_position(source)
                source_xyz = source_grid_to_physical(horizontal_index, depth_index)
                self._source_xyz_cache[index] = source_xyz

            if self.variant is ExperimentVariant.DIRECT_COORDINATES:
                normalized_coordinates = normalize_source_coordinates(source_xyz)
                feature_fields.append(
                    np.broadcast_to(normalized_coordinates, spatial_shape + (3,))
                )

            if self.variant in {
                ExperimentVariant.DEM_DISTANCE,
                ExperimentVariant.TERRAIN_DIRECTION,
                ExperimentVariant.LOCAL_COMPONENTS,
                ExperimentVariant.GEOMETRY_ONLY,
                ExperimentVariant.TERRAIN_GATE,
                ExperimentVariant.TERRAIN_GATE_SHUFFLED,
            }:
                geometry = build_geometry_features(
                    self.station_grid,
                    source_xyz,
                    distance_scale_m=self.distance_scale_m,
                    terrain_basis=self.terrain_basis,
                )
                distance_feature = geometry["distance"][..., None]
                feature_fields.append(distance_feature)
                if self.variant in {
                    ExperimentVariant.TERRAIN_DIRECTION,
                    ExperimentVariant.LOCAL_COMPONENTS,
                    ExperimentVariant.GEOMETRY_ONLY,
                    ExperimentVariant.TERRAIN_GATE,
                    ExperimentVariant.TERRAIN_GATE_SHUFFLED,
                }:
                    feature_fields.append(geometry["terrain_direction"])

        features = np.concatenate(feature_fields, axis=-1).astype(np.float32, copy=False)
        target = np.asarray(target, dtype=np.float32)
        return torch.from_numpy(features.copy()), torch.from_numpy(target.copy())
