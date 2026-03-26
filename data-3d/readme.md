# Database Construction for 3D Seismic Wavefield Modeling with Random Sources

## Overview

This repository provides a description of the database construction workflow used for three-dimensional seismic wavefield modeling with random sources. The database was developed to support the training, testing, and evaluation of the U-shaped Fourier Neural Operator (U-FNO) framework.

The main purpose of this database is to organize seismic source parameters, simulation case settings, site model parameters, and representative output data in a reproducible manner.

## Database Construction Workflow

The database was constructed through the following workflow:

1. **Source parameter modification and case submission**  
   The script `data-main` was used to modify seismic source parameters and submit different simulation cases. This script serves as the main entry point for generating multiple source-condition combinations and managing the corresponding simulation tasks.

2. **Site model parameter definition**  
   The site model parameters were not stored directly in this repository. Instead, they are hosted externally in ScienceDB and can be accessed through the following link:  
   [https://www.scidb.cn/s/AfyEVb](https://www.scidb.cn/s/AfyEVb)

3. **Reference data storage**  
   In addition to the site model parameters, part of the reference data used in this study is also stored in the same ScienceDB record for convenient access and verification.

## Data Contents

The database includes or references the following types of information:

- seismic source parameter settings,
- simulation case configurations,
- site model parameters,
- representative reference data for comparison and validation.

## Notes on Reproducibility

To reproduce or extend the database construction process:

- use `data-main` to modify source parameters,
- submit the desired simulation cases according to the study design,
- download the site model parameters and reference data from ScienceDB,
- organize the generated outputs following the same data structure adopted in this repository.

