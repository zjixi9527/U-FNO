"""Geometry transforms shared by all ablation experiments.

The global vector-component order follows SPECFEM's Cartesian output:
``CXX, CXY, CXZ`` = ``easting, northing, vertical``.  The first two
spatial array axes still follow station-file order and must not be confused
with vector-component order.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


DEFAULT_GRID_SIZE = 64.0
DEFAULT_EASTING_ORIGIN_M = 720_000.0
DEFAULT_FAULT_HORIZONTAL_SPAN_M = 20_000.0
DEFAULT_SURFACE_DEPTH_M = -3_000.0
DEFAULT_FAULT_DEPTH_SPAN_M = 10_000.0
DEFAULT_FAULT_NORTHING_INTERCEPT_M = 4_693_850.0
DEFAULT_FAULT_NORTHING_SLOPE = -0.745
DEFAULT_FAULT_DIP_OFFSET_DEGREES = 10.0


def _default_source_coordinate_bounds() -> tuple[np.ndarray, np.ndarray]:
    corner_coordinates = np.stack(
        [
            source_grid_to_physical(horizontal_index, depth_index)
            for horizontal_index in (0.0, DEFAULT_GRID_SIZE)
            for depth_index in (0.0, DEFAULT_GRID_SIZE)
        ]
    )
    return corner_coordinates.min(axis=0), corner_coordinates.max(axis=0)


def _normalize_vectors(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, epsilon)


def recover_source_grid_position(
    source_tensor: np.ndarray,
    *,
    radius: float = 3.0,
) -> tuple[float, float]:
    """Recover the continuous source center from its radial cone encoding."""
    source = np.asarray(source_tensor, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"source_tensor must be 2-D, got shape {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("source_tensor contains non-finite values")
    if np.any(source < 0.0):
        raise ValueError("source_tensor must be non-negative")

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a finite positive number")
    total_weight = float(source.sum())
    if total_weight <= 0.0:
        raise ValueError("source_tensor must contain at least one positive value")

    row_indices, column_indices = np.indices(source.shape, dtype=np.float64)
    initial_position = np.array(
        [
            np.sum(row_indices * source) / total_weight,
            np.sum(column_indices * source) / total_weight,
        ],
        dtype=np.float64,
    )

    active_rows, active_columns = np.nonzero(source > 0.0)
    active_values = source[active_rows, active_columns]
    if active_values.size < 3:
        return float(initial_position[0]), float(initial_position[1])

    encoded_distances = radius * (1.0 - active_values)

    def residuals(center: np.ndarray) -> np.ndarray:
        reconstructed = np.sqrt(
            (active_rows - center[0]) ** 2 + (active_columns - center[1]) ** 2
        )
        return reconstructed - encoded_distances

    solution = least_squares(
        residuals,
        initial_position,
        bounds=(
            np.array([-radius, -radius]),
            np.array([source.shape[0] - 1 + radius, source.shape[1] - 1 + radius]),
        ),
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError("could not recover a finite source position from source_tensor")
    return float(solution.x[0]), float(solution.x[1])


def source_grid_to_physical(
    fault_horizontal_index: float,
    fault_depth_index: float,
    *,
    grid_size: float = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Reproduce the source-grid to physical-coordinate map in ``main1.py``.

    Returns coordinates in ``[easting, northing, vertical]`` order so they
    align with ``CXX, CXY, CXZ`` and the parsed ``STATIONS`` coordinates.
    """
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")

    easting = (
        float(fault_horizontal_index)
        / grid_size
        * DEFAULT_FAULT_HORIZONTAL_SPAN_M
        + DEFAULT_EASTING_ORIGIN_M
    )
    vertical = (
        -float(fault_depth_index)
        / grid_size
        * DEFAULT_FAULT_DEPTH_SPAN_M
        + DEFAULT_SURFACE_DEPTH_M
    )
    northing = (
        DEFAULT_FAULT_NORTHING_SLOPE * easting
        + DEFAULT_FAULT_NORTHING_INTERCEPT_M
        + (-DEFAULT_SURFACE_DEPTH_M + vertical)
        / DEFAULT_FAULT_DEPTH_SPAN_M
        * math.tan(math.radians(DEFAULT_FAULT_DIP_OFFSET_DEGREES))
    )
    return np.array([easting, northing, vertical], dtype=np.float64)


def normalize_source_coordinates(source_xyz: np.ndarray) -> np.ndarray:
    """Map physical source coordinates to ``[-1, 1]`` using fixed domain bounds.

    The bounds come from the four corners of the source plane used by the
    database builder.  They are independent of the train/validation/test
    samples, so this representation cannot leak split-specific statistics.
    """
    source = np.asarray(source_xyz, dtype=np.float64)
    if source.shape != (3,) or not np.isfinite(source).all():
        raise ValueError("source_xyz must contain three finite coordinates")

    physical_minimum, physical_maximum = _default_source_coordinate_bounds()
    normalized = 2.0 * (source - physical_minimum) / (
        physical_maximum - physical_minimum
    ) - 1.0
    return normalized.astype(np.float32)


def build_terrain_basis(
    station_grid: np.ndarray,
    *,
    neighborhood_size: int = 9,
) -> np.ndarray:
    """Build orthonormal local tangent/tangent/normal bases from station DEM.

    ``station_grid`` has shape ``(H, W, 3)`` and component order
    ``[easting, northing, elevation]``.  Rows in the returned matrix are
    ``[t_easting, t_northing, normal]`` expressed in the global basis.
    """
    stations = np.asarray(station_grid, dtype=np.float64)
    if stations.ndim != 3 or stations.shape[-1] != 3:
        raise ValueError(f"station_grid must have shape (H, W, 3), got {stations.shape}")
    if min(stations.shape[:2]) < 2:
        raise ValueError("station_grid must contain at least 2 points on each axis")
    if not np.isfinite(stations).all():
        raise ValueError("station_grid contains non-finite values")
    if neighborhood_size < 3:
        raise ValueError("neighborhood_size must be at least 3")

    horizontal = stations[..., :2].reshape(-1, 2)
    elevation = stations[..., 2].reshape(-1)
    neighbor_count = min(neighborhood_size, horizontal.shape[0])
    _, neighbor_indices = cKDTree(horizontal).query(horizontal, k=neighbor_count)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    slopes = np.empty((horizontal.shape[0], 2), dtype=np.float64)
    for point_index, neighbors in enumerate(neighbor_indices):
        offsets = horizontal[neighbors] - horizontal[point_index]
        elevation_offsets = elevation[neighbors] - elevation[point_index]
        coefficients, _, rank, _ = np.linalg.lstsq(offsets, elevation_offsets, rcond=None)
        if rank < 2:
            raise ValueError("local station geometry is degenerate and cannot define a terrain plane")
        slopes[point_index] = coefficients

    slope_easting = slopes[:, 0].reshape(stations.shape[:2])
    slope_northing = slopes[:, 1].reshape(stations.shape[:2])
    tangent_easting = _normalize_vectors(
        np.stack(
            [np.ones_like(slope_easting), np.zeros_like(slope_easting), slope_easting],
            axis=-1,
        )
    )
    normal = _normalize_vectors(
        np.stack(
            [-slope_easting, -slope_northing, np.ones_like(slope_easting)], axis=-1
        )
    )
    tangent_northing = _normalize_vectors(np.cross(normal, tangent_easting))

    return np.stack([tangent_easting, tangent_northing, normal], axis=-2)


def build_geometry_features(
    station_grid: np.ndarray,
    source_xyz: np.ndarray,
    *,
    distance_scale_m: float,
    terrain_basis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create normalized distance and global/terrain-relative direction fields."""
    if not np.isfinite(distance_scale_m) or distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be a finite positive number")

    stations = np.asarray(station_grid, dtype=np.float64)
    source = np.asarray(source_xyz, dtype=np.float64)
    if stations.ndim != 3 or stations.shape[-1] != 3:
        raise ValueError(f"station_grid must have shape (H, W, 3), got {stations.shape}")
    if source.shape != (3,) or not np.isfinite(source).all():
        raise ValueError("source_xyz must contain three finite coordinates")

    delta = stations - source
    distance_m = np.linalg.norm(delta, axis=-1)
    global_direction = delta / np.maximum(distance_m[..., None], 1e-12)
    basis = build_terrain_basis(stations) if terrain_basis is None else np.asarray(terrain_basis)
    if basis.shape != stations.shape[:2] + (3, 3):
        raise ValueError("terrain_basis shape must be (H, W, 3, 3)")
    terrain_direction = np.einsum("hwkc,hwc->hwk", basis, global_direction)

    return {
        "distance": (distance_m / distance_scale_m).astype(np.float32),
        "distance_m": distance_m.astype(np.float32),
        "global_direction": global_direction.astype(np.float32),
        "terrain_direction": terrain_direction.astype(np.float32),
    }


def compute_terrain_gate_maps(
    station_grid: np.ndarray,
    *,
    smooth_sigma: float = 3.0,
    scaling: str = "adaptive",
) -> dict[str, np.ndarray]:
    """Generate topography-gated masks from DEM for each U-Net skip level.

    The gate value at each surface point encodes local topographic complexity
    (gradient magnitude + residual after smoothing).  Steeper / more rugged
    terrain produces gate values closer to 1, sending more high-frequency
    detail through the skip connections.  Flatter terrain produces values
    closer to 0.

    Parameters
    ----------
    station_grid:
        ``(H, W, 3)`` array of ``[easting, northing, elevation]``.
    smooth_sigma:
        Gaussian-smoothing sigma used to isolate high-wavenumber residuals.
    scaling:
        ``"adaptive"`` — normalise by the raw maximum so values live in
        ``[0, 1]`` but preserve the relative spatial contrast;
        ``"sigmoid"`` — soft centering via a sigmoid transform.

    Returns
    -------
    dict with keys ``gate_0`` (shape ``(1, H, W, 1, 1)``) and ``gate_1``
    (shape ``(1, H//2, W//2, 1, 1)``).  The extra dimensions align with
    the ``Uno3D_T10_Gated`` forward signature.
    """
    stations = np.asarray(station_grid, dtype=np.float64)
    if stations.ndim != 3 or stations.shape[-1] != 3:
        raise ValueError(f"station_grid must have shape (H, W, 3), got {stations.shape}")
    if smooth_sigma <= 0.0:
        raise ValueError("smooth_sigma must be positive")
    if scaling not in ("adaptive", "sigmoid"):
        raise ValueError(f"unknown scaling method: {scaling}")

    from scipy.ndimage import gaussian_filter

    elevation = stations[..., 2]
    gy, gx = np.gradient(elevation)
    grad_mag = np.sqrt(gx**2 + gy**2)
    smooth = gaussian_filter(elevation, sigma=smooth_sigma)
    residual = np.abs(elevation - smooth)
    gate_raw = grad_mag + residual

    if scaling == "adaptive":
        gate_raw = gate_raw / (gate_raw.max() + 1e-12)
    else:
        mean_val = float(gate_raw.mean())
        std_val = max(float(gate_raw.std()), 1e-12)
        gate_raw = 1.0 / (1.0 + np.exp(-(gate_raw - mean_val) / std_val))

    H, W = gate_raw.shape
    gate_0 = gate_raw[None, :, :, None, None].astype(np.float32)
    # Downsample by factor 2 via block averaging for the deeper skip level.
    gate_1 = (gate_raw.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)))
    gate_1 = gate_1[None, :, :, None, None].astype(np.float32)

    return {"gate_0": gate_0, "gate_1": gate_1}


def _gaussian_filter_2d(data: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter via separable convolution — scipy.ndimage alternative."""
    if sigma <= 0:
        return data.copy()
    k = max(int(4 * sigma + 0.5), 3)
    if k % 2 == 0:
        k += 1
    x = np.arange(-k // 2 + 1, k // 2 + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return scipy_signal.fftconvolve(
        data, kernel[:, None], mode="constant"
    )


def _laplace_2d(data: np.ndarray) -> np.ndarray:
    """Laplace operator (sum of second derivatives) — scipy.ndimage.laplace alternative."""
    dy, dx = np.gradient(data)
    dyy, _ = np.gradient(dy)
    _, dxx = np.gradient(dx)
    return dyy + dxx


def _zoom_2d(data: np.ndarray, zoom_factors: tuple[float, float], order: int = 3) -> np.ndarray:
    """Bilinear/bicubic resampling — scipy.ndimage.zoom alternative."""
    H, W = data.shape
    new_h = max(int(round(H * zoom_factors[0])), 2)
    new_w = max(int(round(W * zoom_factors[1])), 2)
    ny, nx = np.ogrid[:new_h, :new_w]
    fy = (ny + 0.5) / new_h * H - 0.5
    fx = (nx + 0.5) / new_w * W - 0.5

    if order <= 1:
        return _bilinear_2d(data, fy, fx)
    else:
        return _bicubic_2d(data, fy, fx)


def _bilinear_2d(data: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of 2D grid at (fy, fx) coordinates."""
    H, W = data.shape
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, H - 2)
    x0 = np.clip(np.floor(fx).astype(np.int32), 0, W - 2)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (fy - y0).astype(np.float64)
    wx = (fx - x0).astype(np.float64)
    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx
    return wa * data[y0, x0] + wb * data[y0, x1] + wc * data[y1, x0] + wd * data[y1, x1]


def _bicubic_2d(data: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Bicubic interpolation of 2D grid at (fy, fx) coordinates."""
    H, W = data.shape
    result = np.zeros_like(fy, dtype=np.float64)
    # Process in chunks to avoid memory blow-up for large grids.
    chunk_size = 1024
    total = fy.size
    flat_fy = fy.ravel()
    flat_fx = fx.ravel()
    for i in range(0, total, chunk_size):
        end = min(i + chunk_size, total)
        fy_c = flat_fy[i:end, None]  # (N, 1)
        fx_c = flat_fx[i:end, None]  # (N, 1)

        y0 = np.clip(np.floor(fy_c).astype(np.int32), 0, H - 2)  # (N, 1)
        x0 = np.clip(np.floor(fx_c).astype(np.int32), 0, W - 2)  # (N, 1)

        # Bicubic kernel coefficients (order=3, cubic B-spline)
        def _cubic(t: np.ndarray) -> np.ndarray:
            at = np.abs(t)
            return np.where(
                at < 1,
                (1.5 * at - 2.5) * at * at + 1,
                np.where(at < 2, (-0.5 * at + 2.5) * at * at - 4 * at + 2, 0),
            )

        y_vals = np.arange(y0[0, 0] - 1, y0[0, 0] + 4)  # 5 rows
        x_vals = np.arange(x0[0, 0] - 1, x0[0, 0] + 4)  # 5 cols

        y_clipped = np.clip(y_vals, 0, H - 1)[:, None]
        x_clipped = np.clip(x_vals, 0, W - 1)[None, :]

        # Interpolated data values (N, 1, 1) ← (H_patch, W_patch)
        patch = data[y_clipped, x_clipped]  # (N, 5, 5)

        cy = np.clip(fy_c + 0.5 - y0, -2, 3)  # (N, 1)
        cx = np.clip(fx_c + 0.5 - x0, -2, 3)  # (N, 1)

        cy_coeffs = _cubic(cy)  # (N, 1)
        cx_coeffs = _cubic(cx)  # (N, 1)

        # 2D cubic: outer product then sum
        kernel_2d = cy_coeffs @ cx_coeffs.T  # (N, 5, 5)
        result[i:end] = np.sum(kernel_2d * patch, axis=(1, 2)).reshape(-1)

    return result.reshape(fy.shape)


def compute_enhanced_terrain_gate_maps(
    dem_elevation: np.ndarray,
    *,
    output_grid_shape: tuple[int, int] = (64, 64),
) -> dict[str, np.ndarray]:
    """Compute a simple slope gate from a high-resolution DEM.

    The gate uses only the DEM gradient magnitude ``|∇h|``, the most direct
    indicator of local topographic slope.  It is robustly scaled by its 90th
    percentile, clipped to ``[0, 1]``, and resized to the station grid.

    Parameters
    ----------
    dem_elevation:
        ``(H_dem, W_dem)`` array of elevations in metres from a high-resolution
        DEM (e.g. 301×301 @ 100 m).  Must be at least as fine as the output
        grid in each dimension.
    output_grid_shape:
        ``(H_out, W_out)`` — the target station-grid resolution (typically
        64×64).

    Returns
    -------
    dict with keys ``gate_0`` (shape ``(1, H_out, W_out, 1, 1)``) and
    ``gate_1`` (shape ``(1, H_out//2, W_out//2, 1, 1)``).
    """
    dem = np.asarray(dem_elevation, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"dem_elevation must be 2-D, got shape {dem.shape}")
    if dem.shape[0] < output_grid_shape[0] or dem.shape[1] < output_grid_shape[1]:
        raise ValueError(
            f"DEM ({dem.shape}) must be at least as large as output grid ({output_grid_shape}) "
            "in each dimension"
        )
    H_out, W_out = output_grid_shape

    # Local slope magnitude is the sole terrain indicator.
    gy, gx = np.gradient(dem)
    grad_mag = np.sqrt(gx**2 + gy**2)
    p90 = float(np.percentile(grad_mag, 90))
    slope_gate = (
        np.clip(grad_mag / p90, 0.0, 1.0)
        if p90 > 1e-12
        else np.zeros_like(grad_mag)
    )

    # Downsample from DEM resolution to output station-grid resolution
    # via bicubic interpolation.
    zoom_y = H_out / dem.shape[0]
    zoom_x = W_out / dem.shape[1]
    gate_0_raw = np.clip(
        _zoom_2d(slope_gate, (zoom_y, zoom_x), order=1),
        0.0,
        1.0,
    ).astype(np.float32)

    gate_0 = gate_0_raw[None, :, :, None, None]
    gate_1 = (gate_0_raw.reshape(H_out // 2, 2, W_out // 2, 2).mean(axis=(1, 3)))
    gate_1 = gate_1[None, :, :, None, None].astype(np.float32)

    return {"gate_0": gate_0, "gate_1": gate_1}


def load_grd_dem(
    grd_path: str,
) -> np.ndarray:
    """Load a Surfer ASCII DSAA GRD file and return the elevation grid.

    Returns
    -------
    ``(nx, ny)`` array of elevations in metres.  Axis 0 runs west-to-east
    and axis 1 runs south-to-north, matching the project's STATIONS grid
    ordering.  Surfer DSAA stores these axes in the opposite array order,
    so the parsed ``(ny, nx)`` grid is transposed once.
    """
    with open(grd_path, "r", encoding="ascii") as fh:
        lines = fh.readlines()

    if len(lines) < 6 or lines[0].strip() != "DSAA":
        raise ValueError("GRD file must be a Surfer ASCII DSAA grid")

    dimensions = lines[1].split()
    if len(dimensions) != 2:
        raise ValueError("DSAA grid-size line must contain nx and ny")
    nx, ny = (int(value) for value in dimensions)
    if nx < 2 or ny < 2:
        raise ValueError(f"DSAA grid dimensions must be at least 2x2, got {nx}x{ny}")

    elevation_values = np.fromstring(" ".join(lines[5:]), sep=" ", dtype=np.float64)
    expected_values = nx * ny
    if elevation_values.size != expected_values:
        raise ValueError(
            f"DSAA data section contains {elevation_values.size} values, "
            f"expected {expected_values} for a {nx}x{ny} grid"
        )
    if not np.isfinite(elevation_values).all():
        raise ValueError("DSAA elevation data contains non-finite values")

    return elevation_values.reshape((ny, nx)).T


def global_to_local(wavefield: np.ndarray, terrain_basis: np.ndarray) -> np.ndarray:
    """Rotate ``(..., time, global_component)`` wavefields to local components."""
    values = np.asarray(wavefield)
    basis = np.asarray(terrain_basis)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"wavefield must have shape (H, W, T, 3), got {values.shape}")
    if basis.shape != values.shape[:2] + (3, 3):
        raise ValueError("terrain_basis is incompatible with wavefield spatial shape")
    return np.einsum("hwkc,hwtc->hwtk", basis, values)


def local_to_global(wavefield: np.ndarray, terrain_basis: np.ndarray) -> np.ndarray:
    """Rotate local tangent/tangent/normal components back to global components."""
    values = np.asarray(wavefield)
    basis = np.asarray(terrain_basis)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"wavefield must have shape (H, W, T, 3), got {values.shape}")
    if basis.shape != values.shape[:2] + (3, 3):
        raise ValueError("terrain_basis is incompatible with wavefield spatial shape")
    return np.einsum("hwkc,hwtk->hwtc", basis, values)
