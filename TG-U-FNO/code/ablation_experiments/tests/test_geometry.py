import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ablation_common.geometry import (
    build_geometry_features,
    build_terrain_basis,
    compute_enhanced_terrain_gate_maps,
    compute_terrain_gate_maps,
    global_to_local,
    local_to_global,
    load_grd_dem,
    normalize_source_coordinates,
    recover_source_grid_position,
    source_grid_to_physical,
)


class GeometryTests(unittest.TestCase):
    def test_load_grd_dem_reads_ascii_dsaa_grid(self):
        with TemporaryDirectory() as temporary_directory:
            grd_path = Path(temporary_directory) / "terrain.grd"
            grd_path.write_text(
                "DSAA\n"
                "3 2\n"
                "100.0 300.0\n"
                "400.0 500.0\n"
                "1.0 6.0\n"
                "1.0 2.0 3.0\n"
                "4.0 5.0 6.0\n",
                encoding="ascii",
            )

            elevation = load_grd_dem(str(grd_path))

        np.testing.assert_allclose(
            elevation,
            np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        )

    def test_high_resolution_gate_uses_only_gradient_magnitude(self):
        rows, columns = np.indices((16, 16), dtype=np.float64)
        planar_dem = 3.0 * rows + 4.0 * columns + 1200.0

        gate_maps = compute_enhanced_terrain_gate_maps(
            planar_dem,
            output_grid_shape=(8, 8),
        )

        np.testing.assert_allclose(
            gate_maps["gate_0"],
            np.ones((1, 8, 8, 1, 1), dtype=np.float32),
            atol=1e-6,
        )

    def test_recovers_subgrid_source_center_from_radial_encoding(self):
        rows, columns = np.indices((64, 64), dtype=np.float64)
        source_row = 17.25
        source_column = 41.5
        distance = np.sqrt((rows - source_row) ** 2 + (columns - source_column) ** 2)
        source_tensor = np.maximum(0.0, 1.0 - distance / 3.0)

        recovered_row, recovered_column = recover_source_grid_position(source_tensor)

        self.assertAlmostEqual(recovered_row, source_row, delta=0.25)
        self.assertAlmostEqual(recovered_column, source_column, delta=0.25)

    def test_recovers_source_center_when_cone_is_truncated_by_boundary(self):
        rows, columns = np.indices((64, 64), dtype=np.float64)
        source_row = 0.35
        source_column = 63.6
        distance = np.sqrt((rows - source_row) ** 2 + (columns - source_column) ** 2)
        source_tensor = np.maximum(0.0, 1.0 - distance / 3.0)

        recovered_row, recovered_column = recover_source_grid_position(source_tensor)

        self.assertAlmostEqual(recovered_row, source_row, delta=0.1)
        self.assertAlmostEqual(recovered_column, source_column, delta=0.1)

    def test_source_grid_mapping_matches_database_builder_endpoints(self):
        source_xyz = source_grid_to_physical(0.0, 0.0)

        expected_easting = 720000.0
        expected_depth = -3000.0
        expected_northing = -0.745 * expected_easting + 4693850.0
        np.testing.assert_allclose(
            source_xyz,
            np.array([expected_easting, expected_northing, expected_depth]),
        )

        deep_source_xyz = source_grid_to_physical(64.0, 64.0)
        deep_easting = 740000.0
        deep_depth = -13000.0
        deep_northing = (
            -0.745 * deep_easting
            + 4693850.0
            + (3000.0 + deep_depth) / 10000.0 * np.tan(np.radians(10.0))
        )
        np.testing.assert_allclose(
            deep_source_xyz,
            np.array([deep_easting, deep_northing, deep_depth]),
            rtol=0.0,
            atol=1e-9,
        )

    def test_source_coordinate_normalization_uses_fixed_physical_bounds(self):
        corner_coordinates = np.stack(
            [
                source_grid_to_physical(row, column)
                for row in (0.0, 64.0)
                for column in (0.0, 64.0)
            ]
        )
        physical_minimum = corner_coordinates.min(axis=0)
        physical_maximum = corner_coordinates.max(axis=0)

        np.testing.assert_allclose(
            normalize_source_coordinates(physical_minimum),
            np.full(3, -1.0),
        )
        np.testing.assert_allclose(
            normalize_source_coordinates(physical_maximum),
            np.full(3, 1.0),
        )

    def test_source_coordinate_normalization_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "three"):
            normalize_source_coordinates(np.zeros(2))

    def test_source_plane_center_normalizes_to_origin(self):
        source_center = source_grid_to_physical(32.0, 32.0)

        normalized = normalize_source_coordinates(source_center)

        np.testing.assert_allclose(normalized, np.zeros(3), atol=1e-7)

    def test_direct_coordinate_normalization_locks_axis_order_and_sign(self):
        cases = (
            ((0.0, 0.0), (-1.0, 1.0, 1.0)),
            ((32.0, 32.0), (0.0, 0.0, 0.0)),
            ((64.0, 64.0), (1.0, -1.0, -1.0)),
        )

        for grid_position, expected in cases:
            with self.subTest(grid_position=grid_position):
                normalized = normalize_source_coordinates(
                    source_grid_to_physical(*grid_position)
                )
                np.testing.assert_allclose(
                    normalized,
                    np.asarray(expected, dtype=np.float32),
                    rtol=0.0,
                    atol=1e-6,
                )
                self.assertEqual(normalized.dtype, np.float32)

    def test_source_coordinate_normalization_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_source_coordinates(np.array([720_000.0, np.nan, -3_000.0]))

    def test_flat_surface_basis_is_global_basis(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3.0, 4),
            np.linspace(0.0, 2.0, 3),
        )
        elevation = np.zeros_like(northing)
        stations = np.stack([easting, northing, elevation], axis=-1)

        basis = build_terrain_basis(stations)

        expected = np.broadcast_to(np.eye(3), basis.shape)
        np.testing.assert_allclose(basis, expected, atol=1e-7)

    def test_planar_surface_normal_matches_analytical_normal(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3000.0, 4),
            np.linspace(0.0, 2000.0, 3),
        )
        slope_northing = 0.1
        slope_easting = -0.05
        elevation = slope_northing * northing + slope_easting * easting
        stations = np.stack([easting, northing, elevation], axis=-1)

        basis = build_terrain_basis(stations)

        expected_normal = np.array([-slope_easting, -slope_northing, 1.0])
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_allclose(
            basis[..., 2, :],
            np.broadcast_to(expected_normal, basis[..., 2, :].shape),
            atol=1e-10,
        )

    def test_local_global_rotation_round_trip(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3000.0, 4),
            np.linspace(0.0, 2000.0, 3),
        )
        elevation = 0.1 * northing - 0.05 * easting
        stations = np.stack([easting, northing, elevation], axis=-1)
        basis = build_terrain_basis(stations)
        wavefield = np.arange(3 * 4 * 5 * 3, dtype=np.float64).reshape(3, 4, 5, 3)

        local = global_to_local(wavefield, basis)
        restored = local_to_global(local, basis)

        np.testing.assert_allclose(restored, wavefield, atol=1e-9)

    def test_geometry_features_are_finite_and_use_fixed_scale(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3000.0, 4),
            np.linspace(0.0, 2000.0, 3),
        )
        elevation = 0.05 * northing
        stations = np.stack([easting, northing, elevation], axis=-1)
        source_xyz = np.array([1000.0, -1000.0, -3000.0])

        features = build_geometry_features(stations, source_xyz, distance_scale_m=10000.0)

        self.assertEqual(features["distance"].shape, (3, 4))
        self.assertEqual(features["terrain_direction"].shape, (3, 4, 3))
        self.assertTrue(np.isfinite(features["distance"]).all())
        self.assertTrue(np.isfinite(features["terrain_direction"]).all())
        expected_distance = np.linalg.norm(stations - source_xyz, axis=-1) / 10000.0
        np.testing.assert_allclose(features["distance"], expected_distance)

    def test_invalid_distance_scale_is_rejected(self):
        stations = np.zeros((2, 2, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "distance_scale_m"):
            build_geometry_features(stations, np.zeros(3), distance_scale_m=0.0)

    def test_terrain_gate_maps_have_correct_shapes_and_range(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3000.0, 12),
            np.linspace(0.0, 2000.0, 8),
        )
        elevation = 100.0 * np.sin(2 * np.pi * northing / 3000.0) + 50.0 * np.cos(
            2 * np.pi * easting / 2000.0
        )
        stations = np.stack([easting, northing, elevation], axis=-1)

        gate_maps = compute_terrain_gate_maps(stations)

        self.assertEqual(gate_maps["gate_0"].shape, (1, 8, 12, 1, 1))
        self.assertEqual(gate_maps["gate_1"].shape, (1, 4, 6, 1, 1))
        self.assertTrue(np.isfinite(gate_maps["gate_0"]).all())
        self.assertTrue(np.isfinite(gate_maps["gate_1"]).all())
        self.assertGreaterEqual(gate_maps["gate_0"].min(), 0.0)
        self.assertLessEqual(gate_maps["gate_0"].max(), 1.0)
        self.assertGreaterEqual(gate_maps["gate_1"].min(), 0.0)
        self.assertLessEqual(gate_maps["gate_1"].max(), 1.0)

    def test_terrain_gate_flat_surface_is_uniform(self):
        northing, easting = np.meshgrid(
            np.linspace(0.0, 3000.0, 8),
            np.linspace(0.0, 2000.0, 6),
        )
        elevation = np.full_like(northing, 1000.0)
        stations = np.stack([easting, northing, elevation], axis=-1)

        gate_maps = compute_terrain_gate_maps(stations)

        # On a flat surface gradient is zero → gate should be nearly zero.
        self.assertAlmostEqual(float(gate_maps["gate_0"].max()), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(gate_maps["gate_1"].max()), 0.0, delta=1e-6)

    def test_terrain_gate_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "station_grid"):
            compute_terrain_gate_maps(np.zeros((4, 4)))
        with self.assertRaisesRegex(ValueError, "smooth_sigma"):
            compute_terrain_gate_maps(np.zeros((4, 4, 3)), smooth_sigma=0.0)
        with self.assertRaisesRegex(ValueError, "scaling"):
            compute_terrain_gate_maps(np.zeros((4, 4, 3)), scaling="unknown")


if __name__ == "__main__":
    unittest.main()
