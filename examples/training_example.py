import sys
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch


def load_training_module(repo_root: Path):
    """
    Dynamically load U-FNO-wave3d1.py as a Python module.
    This avoids problems caused by the hyphen in the filename.
    """
    script_path = repo_root / "U-FNO-wave3d1.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find training script: {script_path}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = importlib.util.spec_from_file_location("ufno_train_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_h5_schema_example(output_h5: Path):
    """
    Write a very small HDF5 example showing the expected field names
    used by the training code.

    Note:
    - The real training code expects source1...source100 and
      displacement1...displacement100 inside files named like
      displacement_data1.h5, displacement_data2.h5, etc.
    - Here we only write source1 and displacement1 to demonstrate
      the data format without creating a huge file.
    """
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    source = np.zeros((64, 64), dtype=np.float32)
    displacement = np.zeros((50, 64, 64, 3), dtype=np.float32)

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("source1", data=source)
        f.create_dataset("displacement1", data=displacement)


def run_minimal_forward_pass(train_module, output_dir: Path):
    """
    Instantiate the model defined in U-FNO-wave3d1.py and run one
    synthetic forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_module.Uno3D_T10(in_width=6, width=4, factor=1).to(device)
    model.eval()

    # Minimal synthetic input matching the expected input layout:
    # (batch, x, y, t, c) = (1, 64, 64, 50, 1)
    x = torch.zeros((1, 64, 64, 50, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        y = model(x)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "demo_output.npy", y.cpu().numpy())

    summary_lines = [
        "U-FNO training example summary",
        f"Device: {device}",
        f"Input shape: {tuple(x.shape)}",
        f"Output shape: {tuple(y.shape)}",
        "A minimal forward pass through Uno3D_T10 completed successfully.",
        "",
        "Training data naming convention used by U-FNO-wave3d1.py:",
        "- files: displacement_data1.h5, displacement_data2.h5, ...",
        "- datasets inside each file: source1...source100, displacement1...displacement100",
    ]

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Minimal forward pass completed.")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Saved output to: {output_dir / 'demo_output.npy'}")
    print(f"Saved summary to: {output_dir / 'summary.txt'}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = Path(__file__).resolve().parent
    output_dir = example_dir / "outputs"

    print("=" * 72)
    print("U-FNO Training-Aware Example")
    print("=" * 72)

    train_module = load_training_module(repo_root)

    # 1) Save a tiny HDF5 example showing the expected data schema
    schema_example_path = output_dir / "example_training_sample.h5"
    write_h5_schema_example(schema_example_path)
    print(f"Saved HDF5 schema example to: {schema_example_path}")

    # 2) Run one minimal model forward pass
    run_minimal_forward_pass(train_module, output_dir)

    print("=" * 72)
    print("Done.")
    print("This example does NOT perform full training.")
    print("It only demonstrates the model entry point and the expected HDF5 schema.")
    print("=" * 72)


if __name__ == "__main__":
    main()
