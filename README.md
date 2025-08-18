 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![Continuous integration](https://github.com/openosmia/snicar-fx/workflows/CI/badge.svg)](https://github.com/openosmia/snicar-fx/actions)
 [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
 [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
# SNICAR-fx: A flexible and light-weight version of the SNICAR model

SNICAR-fx started as a fork of the biosnicar-py repository, a python translation of SNICAR (SNow, ICe and Aerosols Radiative model), to then evolve into its own standalone version. SNICAR-fx solves the 1-D unpolarized radiative transfer equation for a column of snow and/or ice, of which single scattering properties are calculated with geometric optics. Each layer can be represented as grains of ice in air of various shapes or bubbles of air in ice, along with a specific water content and concentrations of light absorbing particles. The incoming irradiance can be direct with a prescribed SZA, or diffuse. Fresnel boundary layers can be incorporated to account for the change in refractive index between air and ice by using the two-stream Delta-Eddingon solver developed by Briegleb et al. 2007 and improved by Whicker et al. 2022 to include spectrally-dependent Fresnel reflectance coefficients. A multi-stream delta-M solver employing the advanced matrix operator method from Liu and Weng 2013 is also available for homogeneous interfaces (i.e. no Fresnel layers) and direct irradiance. 

What makes it different from SNICAR and BioSNICAR?
- flexible spectral range: between 200 and 5000nm with resolution >= 1nm
- light weight: no dependence on large optical properties databases
- speed: up to 50x faster than SNICAR-ADv4 depending on the model set-up
- directional capability: multi-stream solver based on the advanced matrix operator method (! no fresnel layers and diffuse irradiance for now)
- additional features: ice refractive index of Cooper et al. 2021, liquid water content in snow and ice, empirical optical properties of glacial microbes
- extensive unit tests

Future developments: 
- include non spherical air bubbles in ice
- include high-resolution (1nm) LAPs and irradiance files to avoid interpolation in handling of resolution
- make solar irradiance input more flexible to handle inputs from other atmospheric models
- include diffuse irradiance and Fresnel layers into the multi-stream solver

## How to use

### Installation

Install Python first.
Create an environment.
Activate environment.
Install the package:

```
pip install -e .
```

### Running the code

Set your inputs in the yaml file, then run 'outputs snicarfx.run()'


# Permissions

This code is provided with no conditions.

# Citations

## Equations and model formulation
Original SNICAR equations (Two-stream Delta-Eddington formulation):
Joseph et al. 1976 - https://doi.org/10.1175/1520-0469(1976)033<2452:TDEAFR>2.0.CO;2
Wiscombe and Warren 1980 - https://doi.org/10.1175/1520-0469(1980)037<2712:AMFTSA>2.0.CO;2
Flanner et al. 2021 - https://doi.org/10.5194/gmd-14-7673-2021

Adding-doubling solver with Fresnel layers:
Briegleb and Light 2007 - https://doi.org/10.5065/D6B27S71
Whicker et al. 2022 - https://doi.org/10.5194/tc-16-1197-2022
Whicker et al. 2022 - https://doi.org/10.5194/tc-16-1197-2022

Multi-stream advanced matrix operator method and adding solver: 
Liu and Weng 2013 - 10.1109/JSTARS.2013.2247026

Air bubble single scattering properties: 
Kokhanovsky 2002 (asymmetry parameter) - 10.1088/1464-4258/5/1/307

Ice grain single scattering properties (similar as in TARTES):
Kokhanovsky and Macke 1997 - https://doi.org/10.1364/ao.36.008785
Kokhanovsky 2004 - https://link.springer.com/book/9783540211846
Picard and Libois 2024 (TARTES) - https://doi.org/10.5194/gmd-17-8927-2024
Robledano et al. 2023 - https://doi.org/10.1038/s41467-023-39671-3

## Data

Ice refractive indices:
Cooper et al. 2021 - https://doi.org/10.5194/tc-15-1931-2021
Picard et al. 2016 - https://doi.org/10.5194/tc-10-2655-2016
Warren and Brandt 2008 -  https://doi.org/10.1029/2007JD009744

Spectral irradiances (SWNB2 model runs):
Flanner et al. 2021 - https://doi.org/10.5194/gmd-14-7673-2021

Optical properties: 
Chevrollier et al. 2022 (ice algae) - https://doi.org/10.1017/jog.2022.64
Chevrollier et al. 2025 (snow algae) - https://doi.org/10.5194/tc-19-1527-2025
Flanner et al. 2012 (black carbon) - https://doi.org/10.5194/acp-12-4699-2012
Skiles et al. 2017 (Colorado mineral dust) - https://doi.org/10.1017/jog.2016.126
Kirchstetter et al. 2004 (brown carbon) - https://doi.org/10.1029/2004JD004999
Flanner et al. 2014 (volcanic ashes) - https://doi.org/10.1002/2014JD021977
Balkanski et al. 2007 (Saharan dust) - https://doi.org/10.5194/acp-7-81-2007
