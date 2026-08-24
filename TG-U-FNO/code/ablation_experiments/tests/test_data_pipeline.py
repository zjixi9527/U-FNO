import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from torch.utils.data import DataLoader

from ablation_common.data import AblationDataset, ExperimentVariant, load_station_grid
from ablation_common.geometry import (
    normalize_source_coordinates,
    source_grid_to_physical,
)


def write_station_file(path: Path, size: int = 2) -> None:
    lines = []
    station_number = 1
    for row in range(size):
        for column in range(size):
            northing = 4_100_000.0 + column * 500.0
            easting = 720_000.0 + row * 500.0
            elevation = 3000.0 + 10.0 * row + 20.0 * column
            lines.append(
                f"CZ{station_number}\tHD\t{northing}\t{easting}\t0\t{elevation}\n"
            )
            station_number += 1
    path.write_text("".join(lines), encoding="utf-8")


def write_hdf5(path: Path, size: int = 2, time_steps: int = 3) -> None:
    source = np.zeros((size, size), dtype=np.float32)
    source[0, 1] = 1.0
    wavefield = np.arange(time_steps * size * size * 3, dtype=np.float32).reshape(
        time_steps, size, size, 3
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("source1", data=source)
        handle.create_dataset("displacement1", data=wavefield)


class StationLoaderTests(unittest.TestCase):
    def test_station_grid_preserves_database_order_and_elevation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "STATIONS"
            write_station_file(path)

            stations = load_station_grid(path, grid_shape=(2, 2))

        self.assertEqual(stations.shape, (2, 2, 3))
        np.testing.assert_allclose(stations[0, 0], [720_000.0, 4_100_000.0, 3000.0])
        np.testing.assert_allclose(stations[1, 1], [720_500.0, 4_100_500.0, 3030.0])

    def test_station_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "STATIONS"
            write_station_file(path)

            with self.assertRaisesRegex(ValueError, "station count"):
                load_station_grid(path, grid_shape=(3, 3))

    def test_station_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "STATIONS"
            write_station_file(path)
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            lines[0], lines[1] = lines[1], lines[0]
            path.write_text("".join(lines), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "order mismatch"):
                load_station_grid(path, grid_shape=(2, 2))

    def test_real_station_grid_preserves_cz64_to_cz65_boundary(self):
        station_path = Path(__file__).resolve().parents[2] / "数据库构建" / "STATIONS"

        stations = load_station_grid(station_path)

        self.assertEqual(stations.shape, (64, 64, 3))
        lines = station_path.read_text(encoding="utf-8").splitlines()
        cz64 = lines[63].split()
        cz65 = lines[64].split()
        np.testing.assert_allclose(
            stations[0, 63], [float(cz64[3]), float(cz64[2]), float(cz64[5])]
        )
        np.testing.assert_allclose(
            stations[1, 0], [float(cz65[3]), float(cz65[2]), float(cz65[5])]
        )


class AblationDatasetTests(unittest.TestCase):
    def test_variants_produce_expected_input_channels_and_target_shape(self):
        expected_channels = {
            ExperimentVariant.BASELINE: 1,
            ExperimentVariant.DEM_DISTANCE: 2,
            ExperimentVariant.TERRAIN_DIRECTION: 5,
            ExperimentVariant.LOCAL_COMPONENTS: 5,
            ExperimentVariant.DIRECT_COORDINATES: 3,
            ExperimentVariant.GEOMETRY_ONLY: 4,
            ExperimentVariant.TERRAIN_GATE: 5,
            ExperimentVariant.TERRAIN_GATE_SHUFFLED: 5,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            datasets = []

            for variant, channel_count in expected_channels.items():
                with self.subTest(variant=variant):
                    dataset = AblationDataset(
                        hdf5_paths=[data_path],
                        station_grid=stations,
                        variant=variant,
                        time_steps=3,
                        samples_per_file=1,
                        distance_scale_m=50_000.0,
                    )
                    datasets.append(dataset)
                    features, target = dataset[0]
                    self.assertEqual(tuple(features.shape), (2, 2, channel_count))
                    self.assertEqual(tuple(target.shape), (2, 2, 3, 3))
                    self.assertTrue(np.isfinite(features.numpy()).all())
                    self.assertTrue(np.isfinite(target.numpy()).all())
            for dataset in datasets:
                dataset.close()

    def test_direction_and_local_variants_share_identical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            shared = {
                "hdf5_paths": [data_path],
                "station_grid": stations,
                "time_steps": 3,
                "samples_per_file": 1,
                "distance_scale_m": 50_000.0,
            }
            direction_dataset = AblationDataset(
                variant=ExperimentVariant.TERRAIN_DIRECTION, **shared
            )
            local_dataset = AblationDataset(
                variant=ExperimentVariant.LOCAL_COMPONENTS, **shared
            )

            direction_features, direction_target = direction_dataset[0]
            local_features, local_target = local_dataset[0]
            direction_dataset.close()
            local_dataset.close()

        np.testing.assert_array_equal(direction_features.numpy(), local_features.numpy())
        np.testing.assert_array_equal(direction_target.numpy(), local_target.numpy())

    def test_direct_coordinate_variant_broadcasts_normalized_physical_xyz(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            dataset = AblationDataset(
                hdf5_paths=[data_path],
                station_grid=stations,
                variant=ExperimentVariant.DIRECT_COORDINATES,
                time_steps=3,
                samples_per_file=1,
            )
            baseline_dataset = AblationDataset(
                hdf5_paths=[data_path],
                station_grid=stations,
                variant=ExperimentVariant.BASELINE,
                time_steps=3,
                samples_per_file=1,
            )

            features, target = dataset[0]
            _, baseline_target = baseline_dataset[0]
            dataset.close()
            baseline_dataset.close()

        expected = normalize_source_coordinates(source_grid_to_physical(0.0, 1.0))
        self.assertEqual(tuple(features.shape), (2, 2, 3))
        np.testing.assert_allclose(
            features.numpy(),
            np.broadcast_to(expected, (2, 2, 3)),
            atol=1e-6,
        )
        np.testing.assert_array_equal(target.numpy(), baseline_target.numpy())

    def test_direct_coordinates_do_not_depend_on_geometry_scale(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            shared = {
                "hdf5_paths": [data_path],
                "station_grid": stations,
                "variant": ExperimentVariant.DIRECT_COORDINATES,
                "time_steps": 3,
                "samples_per_file": 1,
            }
            from ablation_common import data as data_module

            with patch.object(data_module, "build_terrain_basis") as terrain_builder, patch.object(
                data_module, "build_geometry_features"
            ) as geometry_builder:
                first_dataset = AblationDataset(distance_scale_m=1.0, **shared)
                second_dataset = AblationDataset(distance_scale_m=100_000.0, **shared)
                first_features, _ = first_dataset[0]
                second_features, _ = second_dataset[0]

            first_dataset.close()
            second_dataset.close()

        terrain_builder.assert_not_called()
        geometry_builder.assert_not_called()
        np.testing.assert_array_equal(first_features.numpy(), second_features.numpy())

    def test_geometry_only_excludes_cone_and_matches_direction_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            shared = {
                "hdf5_paths": [data_path],
                "station_grid": stations,
                "time_steps": 3,
                "samples_per_file": 1,
                "distance_scale_m": 50_000.0,
            }
            direction_dataset = AblationDataset(
                variant=ExperimentVariant.TERRAIN_DIRECTION, **shared
            )
            geometry_dataset = AblationDataset(
                variant=ExperimentVariant.GEOMETRY_ONLY, **shared
            )

            direction_features, direction_target = direction_dataset[0]
            geometry_features, geometry_target = geometry_dataset[0]
            direction_dataset.close()
            geometry_dataset.close()

        self.assertEqual(tuple(geometry_features.shape), (2, 2, 4))
        np.testing.assert_array_equal(
            geometry_features.numpy(), direction_features.numpy()[..., 1:]
        )
        np.testing.assert_array_equal(geometry_target.numpy(), direction_target.numpy())

    def test_geometry_only_distance_scale_changes_only_distance_channel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            shared = {
                "hdf5_paths": [data_path],
                "station_grid": stations,
                "variant": ExperimentVariant.GEOMETRY_ONLY,
                "time_steps": 3,
                "samples_per_file": 1,
            }
            dataset_50k = AblationDataset(distance_scale_m=50_000.0, **shared)
            dataset_25k = AblationDataset(distance_scale_m=25_000.0, **shared)
            features_50k, _ = dataset_50k[0]
            features_25k, _ = dataset_25k[0]
            dataset_50k.close()
            dataset_25k.close()

        np.testing.assert_allclose(
            features_25k.numpy()[..., 0],
            2.0 * features_50k.numpy()[..., 0],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            features_25k.numpy()[..., 1:], features_50k.numpy()[..., 1:]
        )

    def test_reuses_open_hdf5_handle_and_cached_source_position(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            write_hdf5(data_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            dataset = AblationDataset(
                hdf5_paths=[data_path],
                station_grid=stations,
                variant=ExperimentVariant.TERRAIN_DIRECTION,
                time_steps=3,
                samples_per_file=1,
            )

            from ablation_common import data as data_module

            with patch.object(data_module.h5py, "File", wraps=h5py.File) as file_opener, patch.object(
                data_module,
                "recover_source_grid_position",
                wraps=data_module.recover_source_grid_position,
            ) as source_recovery:
                dataset[0]
                dataset[0]

            dataset.close()

        self.assertEqual(file_opener.call_count, 1)
        self.assertEqual(source_recovery.call_count, 1)

    def test_dataset_index_validates_missing_hdf5_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            data_path = directory / "displacement_data1.h5"
            write_station_file(station_path)
            with h5py.File(data_path, "w") as handle:
                handle.create_dataset("source1", data=np.ones((2, 2)))
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            dataset = AblationDataset(
                hdf5_paths=[data_path],
                station_grid=stations,
                variant=ExperimentVariant.BASELINE,
                time_steps=3,
                samples_per_file=1,
            )

            with self.assertRaisesRegex(KeyError, "displacement1"):
                dataset[0]
            dataset.close()

    def test_multiple_workers_read_hdf5_without_inheriting_handles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            station_path = directory / "STATIONS"
            write_station_file(station_path)
            stations = load_station_grid(station_path, grid_shape=(2, 2))
            paths = []
            for file_number in (1, 2):
                data_path = directory / f"displacement_data{file_number}.h5"
                write_hdf5(data_path)
                paths.append(data_path)
            dataset = AblationDataset(
                hdf5_paths=paths,
                station_grid=stations,
                variant=ExperimentVariant.DEM_DISTANCE,
                time_steps=3,
                samples_per_file=1,
            )
            loader = DataLoader(dataset, batch_size=1, num_workers=2)

            batches = list(loader)

            del loader
            dataset.close()

        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0][0].shape), (1, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
