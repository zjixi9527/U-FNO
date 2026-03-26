import os
import sys
import platform
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_python_environment() -> None:
    print_header("Environment Check")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")

    if torch is None:
        print("PyTorch        : NOT INSTALLED")
    else:
        print(f"PyTorch        : {torch.__version__}")
        print(f"CUDA available : {torch.cuda.is_available()}")


def check_required_packages() -> None:
    print_header("Required Package Check")

    required_packages = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("h5py", "h5py"),
        ("matplotlib", "matplotlib"),
        ("torch", "torch"),
    ]

    for package_name, _ in required_packages:
        try:
            __import__(package_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[MISSING] {package_name}")


def check_repository_files(repo_root: Path) -> None:
    print_header("Repository File Check")

    required_files = [
        "README.md",
        "requirements.txt",
        "FNO_2D.py",
        "fno-wave3d.py",
        "fno-predict.py",
        "U-FNO-wave3d1.py",
        "u-fno-predict.py",
    ]

    for rel_path in required_files:
        file_path = repo_root / rel_path
        if file_path.exists():
            print(f"[OK] {rel_path}")
        else:
            print(f"[MISSING] {rel_path}")


def generate_demo_files(output_dir: Path) -> None:
    print_header("Generate Demo Input/Output")

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(2026)

    # Create a small synthetic input array
    demo_input = np.random.randn(1, 16, 16, 4).astype(np.float32)

    # Create a simple demo output array
    demo_output = demo_input.mean(axis=-1, keepdims=True)

    input_path = output_dir / "demo_input.npy"
    output_path = output_dir / "demo_output.npy"

    np.save(input_path, demo_input)
    np.save(output_path, demo_output)

    print(f"Saved demo input : {input_path}")
    print(f"Saved demo output: {output_path}")
    print(f"Demo input shape : {demo_input.shape}")
    print(f"Demo output shape: {demo_output.shape}")


def main() -> None:
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent
    output_dir = current_file.parent / "outputs"

    print_header("U-FNO Repository Quick Test")

    check_python_environment()
    check_required_packages()
    check_repository_files(repo_root)
    generate_demo_files(output_dir)

    print_header("Quick Test Completed")
    print("The repository structure and Python environment were checked successfully.")
    print("This quick test does not reproduce the full experiments in the paper.")
    print("For full training/inference, please prepare the dataset according to the ScienceDB description.")


if __name__ == "__main__":
    main()
