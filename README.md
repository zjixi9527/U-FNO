# U-FNO: Rapid Prediction of Three-Dimensional Seismic Wavefields with Random Sources Using a U-Shaped Fourier Neural Operator

This repository provides the source code associated with the paper:

**Rapid Prediction of Three-Dimensional Seismic Wavefields with Random Sources Using a U-Shaped Fourier Neural Operator**

## Overview

Three-dimensional seismic wavefield simulation in complex sites is computationally expensive, especially when many earthquake source locations need to be considered. This repository contains the implementation of a U-shaped Fourier Neural Operator (U-FNO) framework for rapid prediction of three-dimensional seismic wavefields.

The repository includes model definition files, training scripts, prediction scripts, and a minimal quick test/example for basic verification.
## Data

The full training/validation dataset is not included in this GitHub repository because of its large size and GitHub storage limitations.

Information on dataset construction is available at:
https://www.scidb.cn/s/AfyEVb
## Visual Comparison

Representative comparison between the U-FNO prediction and the spectral element reference solution.  
From top to bottom: **U-FNO prediction**, **spectral element solution**, and **absolute error**.

!([U-FNO vs spectral element solution](https://github.com/zjixi9527/U-FNO/blob/main/assets/gits/1.gif))

## Repository Contents

A typical repository structure is as follows:

```text
.
├── README.md
├── requirements.txt
├── FNO_2D.py
├── fno-wave3d.py
├── fno-predict.py
├── U-FNO-wave3d1.py
├── u-fno-predict.py
├── examples/
│   ├── training_example.py
│   └── README.md
└── data-3d/
