# Rapid Prediction of Three-Dimensional Seismic Wavefields with Random Sources Using a U-Shaped Fourier Neural Operator

This repository provides the source code for the manuscript:

**Rapid Prediction of Three-Dimensional Seismic Wavefields with Random Sources Using a U-Shaped Fourier Neural Operator**

## 1. Overview

Three-dimensional seismic wavefield simulation in complex sites is computationally expensive, especially when multiple earthquake source locations need to be considered.  
This repository contains the implementation of a U-shaped Fourier Neural Operator (U-FNO) framework for rapid prediction of three-dimensional seismic wavefields.

The repository includes:
- model definition files,
- training scripts,
- prediction scripts,
- dependency specification,
- and a minimal quick test for repository verification.

## 2. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── FNO_2D.py
├── fno-wave3d.py
├── fno-predict.py
├── U-FNO-wave3d1.py
├── u-fno-predict.py
├── model/
├── data-3d/
├── examples/
│   ├── README.md
│   └── quick_test.py
```

## 3. Environment Requirements

Recommended environment:
- Python 3.10+
- PyTorch
- NumPy
- SciPy
- h5py
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 4. Data Availability

The full training and validation dataset is not included in this GitHub repository because of its large size and GitHub storage limitations.

The dataset construction workflow is based on **data-main**, which is used for earthquake source parameter modification and case submission.  
The site model parameters, dataset construction information, and part of the reference data are available at:

**Science Data Bank:**  
https://www.scidb.cn/s/AfyEVb

Please note:
- This GitHub repository mainly provides the source code used in the study.
- The large-scale dataset is hosted externally.
- Users should prepare the required data files locally before running full training or prediction.
- 
## 5. Representative Result Visualization

Representative comparison between the U-FNO prediction and the spectral element reference solution for source location 1 \((7, 8, 5)\) km. From top to bottom: U-FNO prediction, spectral element solution, and  error.

![Representative comparison between the U-FNO prediction and the spectral element reference solution for source location 1](https://github.com/zjixi9527/U-FNO/blob/main/1.gif)

## 6. Test

A minimal quick test is provided to verify that the repository is correctly downloaded and that the Python environment is properly configured.

Run:

```bash
python examples/quick_test.py
```

This quick test checks:
- Python and platform information,
- whether PyTorch is installed,
- whether the required Python packages are available,
- and whether the core repository files exist.

Expected result:
- the script prints environment information,
- reports required packages as available,
- and confirms that the main repository files are present.

## 7. Main Scripts

### 7.1 Training scripts
- `U-FNO-wave3d1.py`: main training script for the proposed U-FNO model.
- `fno-wave3d.py`: training script for the baseline FNO model.

### 7.2 Prediction scripts
- `u-fno-predict.py`: prediction/inference script for the trained U-FNO model.
- `fno-predict.py`: prediction/inference script for the trained FNO model.

### 7.3 Model definition
- `FNO_2D.py`: neural operator model components used by the project.

### 7.4 Additional folders
- `model/`: model-related files or saved checkpoints.
- `data-3d/`: data folder used by the scripts.
- `examples/`: minimal example and quick test files.

## 8. How to Use

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the quick test

```bash
python examples/quick_test.py
```

### Step 3: Prepare data

Download or organize the required dataset according to the information provided at:

https://www.scidb.cn/s/AfyEVb

Place the data in your local working directory and modify the input/output paths in the scripts if necessary.

### Step 4: Train the model

Example:

```bash
python U-FNO-wave3d1.py
```

### Step 5: Run prediction

Example:

```bash
python u-fno-predict.py
```

## 9. Notes on Paths and Data

Some scripts may require users to modify:
- input data paths,
- output directories,
- model checkpoint paths.

Before running full experiments, please update these paths according to your local environment.

## 10. Reproducibility Note

This repository is intended to provide public access to the source code used in the study.  
Because the full dataset is large, the repository hosts the code and a lightweight verification example, while the dataset construction details and part of the data are provided through Science Data Bank.


## 10. License

This repository is publicly available for academic research and verification purposes.

A formal open-source license can be added in future revisions if needed.

