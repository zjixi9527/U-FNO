import json
import unittest
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
import torch

from ablation_common.training import (
    FileChunkRandomSampler,
    expand_static_features,
    parse_file_indices,
    relative_l2_loss,
    rotate_global_to_local,
    rotate_local_to_global,
    run_training,
    shuffle_gate_maps_consistently,
    main_for_variant,
    model_supports_grad_scaler,
    read_linux_process_tree_rss_bytes,
    total_file_size_bytes,
    TrainingConfig,
)
from ablation_common.data import ExperimentVariant


class TrainingUtilityTests(unittest.TestCase):
    def test_amp_is_disabled_for_models_with_complex_parameters(self):
        real_model = torch.nn.Linear(2, 1)
        complex_model = torch.nn.Module()
        complex_model.register_parameter(
            "fourier_weight",
            torch.nn.Parameter(torch.ones(2, dtype=torch.cfloat)),
        )

        self.assertTrue(model_supports_grad_scaler(real_model))
        self.assertFalse(model_supports_grad_scaler(complex_model))

    def test_total_file_size_counts_unique_files(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.bin"
            second = directory / "second.bin"
            first.write_bytes(b"1234")
            second.write_bytes(b"123456")

            total = total_file_size_bytes([first, second, first])

        self.assertEqual(total, 10)

    def test_linux_process_tree_rss_sums_parent_and_children(self):
        with TemporaryDirectory() as temporary_directory:
            proc_root = Path(temporary_directory)
            for process_id, rss_kib, children in (
                (100, 10, "101 102"),
                (101, 20, ""),
                (102, 30, "103"),
                (103, 40, ""),
            ):
                process_directory = proc_root / str(process_id)
                task_directory = process_directory / "task" / str(process_id)
                task_directory.mkdir(parents=True)
                (process_directory / "status").write_text(
                    f"Name:\ttest\nVmRSS:\t{rss_kib} kB\n",
                    encoding="utf-8",
                )
                (task_directory / "children").write_text(children, encoding="utf-8")

            rss_bytes = read_linux_process_tree_rss_bytes(
                100,
                proc_root=proc_root,
            )

        self.assertEqual(rss_bytes, 100 * 1024)

    def test_static_features_expand_across_time_without_materializing_host_copy(self):
        static = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)

        expanded = expand_static_features(static, time_steps=5)

        self.assertEqual(tuple(expanded.shape), (1, 2, 3, 5, 4))
        self.assertEqual(expanded.stride(-2), 0)
        for time_index in range(5):
            torch.testing.assert_close(expanded[..., time_index, :], static)

    def test_file_chunk_sampler_is_complete_and_file_local(self):
        generator = torch.Generator().manual_seed(7)
        sampler = FileChunkRandomSampler(
            file_count=3,
            samples_per_file=4,
            chunk_size=2,
            generator=generator,
        )

        indices = list(sampler)

        self.assertEqual(sorted(indices), list(range(12)))
        for offset in range(0, len(indices), 2):
            file_indices = {index // 4 for index in indices[offset : offset + 2]}
            self.assertEqual(len(file_indices), 1)

    def test_parse_file_indices_supports_ranges_and_values(self):
        self.assertEqual(parse_file_indices("1-3,7,10-11"), [1, 2, 3, 7, 10, 11])

    def test_parse_file_indices_rejects_descending_range(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_file_indices("5-2")

    def test_relative_l2_is_zero_for_identical_tensors(self):
        target = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 1, 3)
        self.assertEqual(relative_l2_loss(target, target).item(), 0.0)

    def test_shuffled_gate_levels_share_one_spatial_permutation(self):
        gate_0 = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8, 1, 1)
        gate_1 = torch.zeros(1, 4, 4, 1, 1)
        generator = torch.Generator().manual_seed(17)

        shuffled_0, shuffled_1 = shuffle_gate_maps_consistently(
            gate_0,
            gate_1,
            generator=generator,
        )

        self.assertCountEqual(
            shuffled_0.flatten().tolist(),
            gate_0.flatten().tolist(),
        )
        expected_1 = (
            shuffled_0[0, :, :, 0, 0]
            .reshape(4, 2, 4, 2)
            .mean(dim=(1, 3))
        )
        torch.testing.assert_close(shuffled_1[0, :, :, 0, 0], expected_1)

    def test_shuffled_gate_supports_cpu_generator_for_cuda_gate(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for cross-device generator coverage")

        cpu_gate_0 = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8, 1, 1)
        cpu_gate_1 = torch.zeros(1, 4, 4, 1, 1)
        cpu_reference = shuffle_gate_maps_consistently(
            cpu_gate_0,
            cpu_gate_1,
            generator=torch.Generator().manual_seed(23),
        )

        cuda_reference = shuffle_gate_maps_consistently(
            cpu_gate_0.cuda(),
            cpu_gate_1.cuda(),
            generator=torch.Generator().manual_seed(23),
        )

        self.assertEqual(cuda_reference[0].device.type, "cuda")
        self.assertEqual(cuda_reference[1].device.type, "cuda")
        torch.testing.assert_close(cuda_reference[0].cpu(), cpu_reference[0])
        torch.testing.assert_close(cuda_reference[1].cpu(), cpu_reference[1])

    def test_rotate_local_to_global_matches_numpy_basis_transform(self):
        basis = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        basis[..., 0, 0] = 0.0
        basis[..., 0, 1] = 1.0
        basis[..., 1, 0] = -1.0
        basis[..., 1, 1] = 0.0
        local = torch.arange(24, dtype=torch.float32).reshape(1, 2, 2, 2, 3)

        global_values = rotate_local_to_global(local, torch.tensor(basis, dtype=torch.float32))
        expected = torch.einsum(
            "hwkc,bhwtk->bhwtc", torch.tensor(basis, dtype=torch.float32), local
        )

        torch.testing.assert_close(global_values, expected)

    def test_global_local_rotation_round_trip_on_gpu_training_shape(self):
        basis = torch.eye(3).expand(2, 2, 3, 3).clone()
        basis[..., 0, 0] = 0.0
        basis[..., 0, 1] = 1.0
        basis[..., 1, 0] = -1.0
        basis[..., 1, 1] = 0.0
        global_values = torch.arange(72, dtype=torch.float32).reshape(1, 2, 2, 6, 3)

        local_values = rotate_global_to_local(global_values, basis)
        restored = rotate_local_to_global(local_values, basis)

        torch.testing.assert_close(restored, global_values)


class TinyWavefieldModel(torch.nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.projection = torch.nn.Linear(input_channels, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def write_full_station_grid(path: Path) -> None:
    lines = []
    station_number = 1
    for row in range(64):
        for column in range(64):
            northing = 4_100_000.0 + column * 500.0
            easting = 720_000.0 + row * 500.0
            elevation = 3000.0 + 0.1 * column + 0.2 * row
            lines.append(
                f"CZ{station_number}\tHD\t{northing}\t{easting}\t0\t{elevation}\n"
            )
            station_number += 1
    path.write_text("".join(lines), encoding="utf-8")


def write_training_hdf5(path: Path) -> None:
    source = np.zeros((64, 64), dtype=np.float32)
    source[20, 30] = 1.0
    displacement = np.ones((50, 64, 64, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("source1", data=source)
        handle.create_dataset("displacement1", data=displacement)


class TrainingIntegrationTests(unittest.TestCase):
    def test_all_variant_entrypoints_construct_complete_training_config(self):
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            for variant in ExperimentVariant:
                with self.subTest(variant=variant.value), patch.object(
                    sys,
                    "argv",
                    [
                        "train.py",
                        "--data-dir",
                        "unused",
                        "--output-dir",
                        str(output_root / variant.value),
                    ],
                ), patch(
                    "ablation_common.training.run_training"
                ) as training_runner:
                    main_for_variant(variant, output_root / variant.value)

                config = training_runner.call_args.args[0]
                self.assertEqual(config.variant, variant.value)
                self.assertEqual(config.batch_size, 1)
                self.assertEqual(config.epochs, 200)
                self.assertEqual(config.learning_rate, 0.005)

    def test_new_source_encodings_complete_one_epoch_with_expected_channels(self):
        variants = {
            "direct_coordinates": 3,
            "geometry_only": 4,
        }
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data_directory = directory / "data"
            data_directory.mkdir()
            stations_path = directory / "STATIONS"
            write_full_station_grid(stations_path)
            for file_index in (1, 2):
                write_training_hdf5(data_directory / f"displacement_data{file_index}.h5")

            for variant, channel_count in variants.items():
                with self.subTest(variant=variant):
                    output_directory = directory / variant
                    config = TrainingConfig(
                        variant=variant,
                        data_directory=str(data_directory),
                        stations_path=str(stations_path),
                        train_files="1",
                        validation_files="2",
                        test_files="",
                        samples_per_file=1,
                        time_steps=50,
                        distance_scale_m=50_000.0,
                        batch_size=1,
                        epochs=1,
                        learning_rate=0.001,
                        scheduler_step=1,
                        scheduler_gamma=0.5,
                        model_width=1,
                        seed=7,
                        num_workers=0,
                        io_chunk_size=1,
                        hdf5_cache_size=2,
                        checkpoint_interval=10,
                        resume_path="",
                        device="cpu",
                        amp=False,
                    )

                    with patch(
                        "ablation_common.training._build_model",
                        return_value=TinyWavefieldModel(input_channels=channel_count),
                    ):
                        run_training(config, output_directory)

                    checkpoint = torch.load(
                        output_directory / "best.pt", map_location="cpu"
                    )
                    self.assertEqual(checkpoint["variant"], variant)
                    self.assertEqual(checkpoint["input_channels"], channel_count)

    def test_one_epoch_local_component_run_writes_reproducible_artifacts(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data_directory = directory / "data"
            output_directory = directory / "outputs"
            data_directory.mkdir()
            stations_path = directory / "STATIONS"
            write_full_station_grid(stations_path)
            for file_index in (1, 2, 3):
                write_training_hdf5(data_directory / f"displacement_data{file_index}.h5")

            config = TrainingConfig(
                variant="local_components",
                data_directory=str(data_directory),
                stations_path=str(stations_path),
                train_files="1",
                validation_files="2",
                test_files="3",
                samples_per_file=1,
                time_steps=50,
                distance_scale_m=50_000.0,
                batch_size=1,
                epochs=1,
                learning_rate=0.001,
                scheduler_step=1,
                scheduler_gamma=0.5,
                model_width=1,
                seed=7,
                num_workers=0,
                io_chunk_size=1,
                hdf5_cache_size=2,
                checkpoint_interval=10,
                resume_path="",
                device="cpu",
                amp=False,
            )

            with patch(
                "ablation_common.training._build_model",
                return_value=TinyWavefieldModel(input_channels=5),
            ), patch(
                "ablation_common.training.torch.load", wraps=torch.load
            ) as checkpoint_loader:
                run_training(config, output_directory)

            self.assertEqual(checkpoint_loader.call_count, 1)
            self.assertEqual(Path(checkpoint_loader.call_args.args[0]).name, "best.pt")

            expected_files = {
                "best.pt",
                "last.pt",
                "config.json",
                "history.csv",
                "summary.json",
                "best_validation_per_sample_relative_l2.txt",
                "test_per_sample_relative_l2.txt",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_directory.iterdir()}))
            checkpoint = torch.load(output_directory / "best.pt", map_location="cpu")
            self.assertEqual(checkpoint["variant"], "local_components")
            self.assertEqual(checkpoint["input_channels"], 5)
            self.assertNotIn("optimizer_state_dict", checkpoint)
            last_checkpoint = torch.load(output_directory / "last.pt", map_location="cpu")
            self.assertIn("optimizer_state_dict", last_checkpoint)
            self.assertIn("scheduler_state_dict", last_checkpoint)
            self.assertIn("amp_scaler_state_dict", last_checkpoint)
            self.assertIn("torch_rng_state", last_checkpoint)
            self.assertIn("sampler_generator_state", last_checkpoint)
            history_header = (output_directory / "history.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("epoch_seconds", history_header)
            self.assertIn("train_samples_per_second", history_header)

            resume_config = replace(
                config,
                epochs=2,
                resume_path=str(output_directory / "last.pt"),
            )
            with patch(
                "ablation_common.training._build_model",
                return_value=TinyWavefieldModel(input_channels=5),
            ):
                run_training(resume_config, output_directory)

            history_lines = (output_directory / "history.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(history_lines), 3)
            resumed_checkpoint = torch.load(output_directory / "last.pt", map_location="cpu")
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            expected_resource_fields = {
                "dataset_hdf5_disk_bytes",
                "auxiliary_input_disk_bytes",
                "total_input_disk_bytes",
                "checkpoint_disk_bytes",
                "output_artifacts_disk_bytes",
                "peak_host_memory_bytes",
                "run_wall_seconds",
                "mean_epoch_seconds_this_run",
                "epochs_completed_this_run",
                "hardware",
            }
            self.assertTrue(expected_resource_fields.issubset(summary))
            self.assertGreater(summary["dataset_hdf5_disk_bytes"], 0)
            self.assertGreater(summary["checkpoint_disk_bytes"], 0)
            self.assertGreaterEqual(summary["run_wall_seconds"], 0.0)
            self.assertEqual(summary["epochs_completed_this_run"], 1)
            self.assertIn("torch_version", summary["hardware"])


if __name__ == "__main__":
    unittest.main()
