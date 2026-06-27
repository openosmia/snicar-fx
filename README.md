 [![License: ACSL](https://img.shields.io/badge/License-ACSL-red.svg)](https://anticapitalist.software/)
 [![Continuous integration](https://github.com/openosmia/snicar-fx/workflows/CI/badge.svg)](https://github.com/openosmia/snicar-fx/actions)
 [![codecov](https://codecov.io/gh/openosmia/snicar-fx/graph/badge.svg?token=GS6DG4A1CU)](https://codecov.io/gh/openosmia/snicar-fx)
 [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
 [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Validated with Pydantic](https://img.shields.io/badge/Validated%20with-Pydantic-4FC08D?style=flat&logo=pydantic)](https://docs.pydantic.dev/)

# SNICAR-fx

SNICAR-fx provides a flexible and modular software to simulate radiance, reflectance and albedo at the top-of-atmosphere (TOA) and bottom-of-atmosphere (BOA) for snow and ice surfaces. The model is designed to be user-friendly and can be used for a variety of applications, such as the development of albedo parametrisations in climate models or the interpretation of ground, airborne or spaceborne spectral data. 

Feel free to email us at openosmia@proton.me for any question or guidance with the model. Any input, bug report or contribution is welcome! 🌟

## Table of Contents
1. [Main features](#main-features)
2. [Installation and usage](#installation-and-usage)
3. [Model description](#model-description)
   - [Brief history](#brief-history)
   - [Input parameters](#input-parameters)
   - [Functionality](#functionality)
   - [Limitations](#limitations)
4. [References](#references)
   - [Equations and model formulation](#equations-and-model-formulation)
   - [Data](#data)
   - [Software](#software)
5. [Citation](#citation)

## Main features 

- **flexible spectral resolution**: user-defined resolution in the solar spectrum range and built-in spectral response functions for various satellite instruments
- **versatile land-atmosphere coupling**: unified vertical profile of atmosphere/land layers or simple boundary condition
- **angular resolution**: multi-stream solvers for simulations over polar and azimuthal angles 
- **specialized snow and ice features**: liquid water content and empirical optical properties of diverse particles (dust, cryoconite, algae)
- **user-friendly**: a few lines of code for a model run, and a dedicated API for batch runs
- **modular by design**: adaptable to various user needs and new modules

## Installation and usage 

It is recommended to install `snicar-fx` via Github, with `conda` and `pip`.

Clone the repository and move into the `snicar-fx` directory
```bash
git clone https://github.com/openosmia/snicar-fx
cd snicar-fx
```

Create the `conda` environment with all required dependencies
```bash
conda env create -f environment.yml
```

Activate the new `conda` environment named `snicarfx`
```bash
conda activate snicarfx
```

Install the `snicarfx` software in editable (-e) mode, so that there is not need to re-install the package after local modifications. `conda` has already installed the dependencies at the previous step, so here `pip` only sets up the associated paths
```bash
pip install -e .
```

In order to run land-atmosphere coupled simulations, atmospheric gas absorption cross sections will be required. You can produce a file that is readily compatible with SNICAR-fx using [HAPI2LIBIS](https://github.com/amikko/hapi2libis), or fetch the SNICAR-fx default file available in the SNICAR-fx data archive via [pooch](https://github.com/fatiando/pooch) by running the following command in your terminal
```bash
snicarfx-download-data
```

You're all set! 🌟 
The model runs with a few lines of code from parameters set in a `yaml` input file. Examples of input files for different user cases are available in `examples/`, along with example scripts to run the model. 

## Model description

#### Brief history

SNICAR-fx originally started from the need of a snow/ice radiative transfer
model that would enable a flexible spectral range and resolution while
being fast, open-source and light-weight. The project began as an upgrade 
of the [biosnicar-py](https://github.com/jmcook1186/biosnicar-py) software,
a python translation of
[SNICAR-ADv4](https://github.com/chloewhicker/SNICAR-ADv4). 
Since then, it evolved into a distinct model with 
its own functionalities, strengths and assumptions borrowing from various
established models such as [DISORT v.4.0.99](http://www.rtatmocn.com/disort/), 
[LibRadTran](https://www.libradtran.org/doku.php?id=start) 
or the [CRTM](https://github.com/JCSDA/crtm). 
Today, SNICAR-fx is a flexible and 
modular software for simulations of top-of-atmosphere (TOA) and/or bottom-of-atmosphere (BOA) 
directional radiance and/or reflectance from snow/ice surfaces. Its development
currently targets melting glacier surfaces containing various
light absorbing particles, in particular glacial microbes. 

#### Functionality
 
SNICAR-fx solves the **1-D unpolarized** radiative transfer equation in a
column of homogeneous layers, with a choice of various solvers, briefly
described below:

- A fast two-stream Delta-Eddingon solver ([Briegleb and Light 2007](https://doi.org/10.5065/D6B27S71),
[Whicker et al. 2022](https://doi.org/10.5194/tc-16-1197-2022), similar as in 
[SNICAR-ADv4](https://github.com/mflanner/SNICAR-ADv4)),
to model snow/ice hemispherical albedo, with the possibility of including Fresnel
boundary layers to account for a change in refractive index at the air/ice interface. The solver takes
the incoming surface irradiance (direct + diffuse) as boundary, which is pre-computed
with [LibRadTran](https://www.libradtran.org/doku.php?id=start).
- [PythonicDISORT](https://github.com/LDEO-CREW/Pythonic-DISORT), a python implementation of [DISORT v.4.0.99](http://www.rtatmocn.com/disort/), to model the directional radiance/reflectance (polar + azimuth angle). This solver can be used for coupled and uncoupled simulation to generate TOA and BOA radiance/reflectance/albedo. For coupled simulations, the solver boundary is the
incoming top-of-atmosphere irradiance from [Coddington et al. 2023](https://doi.org/10.1029/2022EA002637). The
addition of Fresnel boundary layers is not possible with this solver. Both the delta-M and delta-M+ methods are available to truncate the phrase function, including IMS-TMS corrections for the Delta-M method.
- A multi-stream adding-doubling solver employing the
advanced matrix operator method
(ADA; [Liu and Weng 2006](https://doi.org/10.1175/JAS3808.1), 
[Liu and Weng 2013](https://doi.org/10.1109/JSTARS.2013.2247026), implemented in the 
[CRTM](https://github.com/JCSDA/crtm)), to model the directional radiance/reflectance 
(polar + azimuth angle). This solver offers similar functionality as the other multi-stream solver except the incoming irradiance is assumed 100% direct, the output is returned over computational polar angles (no interpolation), and no correction for the delta-M method is available.

Each solver is fed the optical properties of snow/ice (+ atmospheric) layers. 
- Snow/ice: Both snow and ice layers are described by their thickness, density, specific surface area, liquid water content and concentrations of various externally-mixed light absorbing particles (see References). Snow layers are modelled as porous media made of perfectly
spherical grains or with an empirically-defined optical shape ([Robledano et al. 2023](https://doi.org/10.1038/s41467-023-39671-3)), similarly to
the model [TARTES](https://github.com/ghislainp/tartes), while ice layers are represented by a continuous medium of ice with spherical air inclusions,
similar to the model [SNICAR-ADv4](https://github.com/chloewhicker/SNICAR-ADv4). The phase function of snow/ice is modelled with the Henyey–Greenstein function by default.
- Atmosphere: atmospheric layers are described by their concentrations of various atmospheric gases and of aerosols. Standard [AFGL atmospheric profiles](https://apps.dtic.mil/sti/citations/ADA175173) are used to prescribe baselines gas concentrations, which can be changed by scaling with column-integrated concentrations via the input file. Molecular scattering is computed following [Bodhaine et al. 1999](https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2) similarly to the default in [LibRadTran]([https://doi.org/10.1175/JAS3808.1](https://www.libradtran.org/doku.php?id=start)). Gaseous absorption is computed using [HAPI2LIBIS](https://doi.org/10.5194/gmd-18-7529-2025) from the [HITRAN database](https://hitran.org/), and aerosol properties are modelled using a mixture of [OPAC](https://geisa.aeris-data.fr/opac/) aerosols.

#### Input parameters
 
The input file contains 5 blocks: 
- SOLVER (computational parameters)
- SPECTRAL (spectral resolution)
- SOLAR (solar geometry)
- LAND (snow/ice properties) 
- ATMOSPHERE (atmospheric properties)

Each block contains various parameters, and not all of them are mandatory. Examples of user cases and associated input parameters are provided in `examples/`, and a full description of each parameter is accessible via: 

```python
from snicarfx import Config
Config.print_help()
```


#### Limitations

Currently, SNICAR-fx does not (indicative non-exhaustive list):
- model radiative transfer outside the solar spectral range.
- model the effect of polarization.
- account for the Earth surface curvature in high latitudes.
- model the effect of clouds in the atmosphere.
- model the effect of Fresnel boundaries in the case of multi-stream / coupled simulations.
- allow for a flexible snow / ice phase function (Henyey–Greenstein as default).
- model the albedo under fully diffuse (cloudy) atmosphere 

## References

#### Equations and model formulation
<details>
<summary>Multi-stream advanced matrix operator method and adding solver (similar as in <a href="https://github.com/JCSDA/crtm">CRTM</a>)</summary> 

- Liu, Q. and Weng, F. (2006). Advanced Doubling–Adding Method for Radiative Transfer in Planetary Atmospheres. 
Journal of the Atmospheric Sciences, 63(12), 3459-3465.
[DOI](https://doi.org/10.1175/JAS3808.1)

- Liu, Q. and Weng, F. (2013). Using advanced matrix operator (AMOM) in community radiative transfer model. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 6(3), 1211-1218. 
[DOI](https://doi.org/10.1109/JSTARS.2013.2247026)

</details>

<details>
<summary>Discrete ordinate method and DISORT solver (i.e. <a href="http://www.rtatmocn.com/disort/">DISORT v4.0.99</a>)</summary> 

- Stamnes, K., Tsay, S. C., Wiscombe, W. and Laszlo, I. (2000). DISORT, a general-purpose Fortran program for discrete-ordinate-method radiative transfer in scattering and emitting layered media: documentation of methodology.

- Stamnes, K., Tsay, S. C., Wiscombe, W. and Jayaweera, K. (1988). Numerically stable algorithm for discrete-ordinate-method radiative transfer in multiple scattering and emitting layered media. Applied optics, 27(12), 2502-2509. [DOI](https://doi.org/10.1364/AO.27.002502)

- Laszlo, I., Stamnes, K., Wiscombe, W. J. and Tsay, S. C. (2016). The discrete ordinate algorithm, DISORT for radiative transfer. In Light Scattering Reviews, Volume 11: Light Scattering and Radiative Transfer (pp. 3-65). Berlin, Heidelberg: Springer Berlin Heidelberg. 
[DOI](https://doi.org/10.1007/978-3-662-49538-4_1)

- Lin, Z., Stamnes, S., Jin, Z., Laszlo, I., Tsay, S. C., Wiscombe, W. J. and Stamnes, K. (2015). Improved discrete ordinate solutions in the presence of an anisotropically reflecting lower boundary: Upgrades of the DISORT computational tool. Journal of Quantitative Spectroscopy and Radiative Transfer, 157, 119-134.
[DOI](https://doi.org/10.1016/j.jqsrt.2015.02.014)

- S. Chandrasekhar (1960). Radiative Transfer.

- Ho, D. X. (2024). PythonicDISORT: A Python reimplementation of theDiscrete Ordinate Radiative Transfer package DISORT. Journal of Open Source Software, 9(103). [DOI](https://doi.org/10.21105/joss.06442)

</details>

<details>
<summary>Two-stream Delta-Eddington solver (original SNICAR formulation, similar as in <a href="https://github.com/mflanner/SNICARv3">SNICAR-ADv3</a>, <a href="https://github.com/chloewhicker/SNICAR-ADv4">SNICAR-ADv4</a>, <a href="github.com/jmcook1186/biosnicar-py">biosnicar-py</a>)</summary> 

- Joseph, J. H., Wiscombe, W. J. and Weinman, J. A. (1976). The delta-Eddington approximation for radiative flux 
transfer. Journal of Atmospheric Sciences, 33(12), 2452-2459. 
[DOI](https://doi.org/10.1175/1520-0469(1976)033<2452:TDEAFR>2.0.CO;2)

- Wiscombe, W. J. and Warren, S. G. (1980). A model for the spectral albedo of snow. I: Pure snow. Journal of 
Atmospheric Sciences, 37(12), 2712-2733.
[DOI](https://doi.org/10.1175/1520-0469(1980)037<2712:AMFTSA>2.0.CO;2)

- Briegleb, P. and Light, B. (2007). A Delta-Eddington mutiple scattering parameterization for solar radiation 
in the sea ice component of the community climate system model. 
[DOI](https://doi.org/10.5065/D6B27S71)

- Whicker, C. A., Flanner, M. G., Dang, C., Zender, C. S., Cook, J. M. and Gardner, A. S. (2021). SNICAR-ADv4: 
a physically based radiative transfer model to represent the spectral albedo of glacier ice. The Cryosphere, 
2021, 1-36. [DOI](https://doi.org/10.5194/tc-16-1197-2022)
</details>

<details>
<summary>Phase function truncation method and correction</summary> 

<br>

- Wiscombe, W. J. (1977). The delta–M method: Rapid yet accurate radiative flux calculations for strongly asymmetric phase functions. 
Journal of Atmospheric Sciences, 34(9), 1408-1422.
[DOI](https://doi.org/10.1175/1520-0469(1977)034<1408:TDMRYA>2.0.CO;2)

- Lin, Z., Chen, N., Fan, Y., Li, W., Stamnes, K. and Stamnes, S. (2018). New treatment of strongly anisotropic scattering phase functions: the Delta-M+ method. Journal of the Atmospheric Sciences, 75(1), 327-336. [DOI](https://doi.org/10.1175/JAS-D-17-0233.1)

- Nakajima, T. and Tanaka, M. (1988). Algorithms for radiative intensity calculations in moderately thick atmospheres using a truncation approximation. Journal of Quantitative Spectroscopy and Radiative Transfer, 40(1), 51-69. [DOI](https://doi.org/10.1016/0022-4073(88)90031-3)

</details>

<details>
<summary>Atmospheric molecular scattering (similar as in <a href="https://www.libradtran.org/doku.php?id=start">LibRadTran</a>)</summary>

<br>

- Bodhaine, B. A., Wood, N. B., Dutton, E. G. and Slusser, J. R. (1999). On Rayleigh optical depth calculations. Journal of Atmospheric and Oceanic Technology, 16(11), 1854-1861. [DOI](https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2)

</details>

<details>
<summary>Air bubble asymmetry parameter</summary> 

<br>

- Kokhanovsky, A. A. (2002). Optical properties of bubbles. Journal of Optics A: Pure and Applied Optics, 
5(1), 47. [DOI](https://doi.org/10.1088/1464-4258/5/1/307)

</details>


<details>
<summary>Ice grain single scattering properties (similar as in <a href="https://github.com/ghislainp/tartes">TARTES</a>)</summary>

- Kokhanovsky, A. A. (2025). Snow optics. Springer. 
[DOI](https://doi.org/10.1007/978-3-031-85979-3)</summary> 

- Kokhanovsky, A. A. and Macke, A. (1997). Integral light-scattering and absorption characteristics of large, nonspherical particles. 
Applied optics, 36(33), 8785-8790. [DOI](https://doi.org/10.1364/AO.36.008785)

- Kokhanovsky, A., Brell, M., Segl, K. and Chabrillat, S. (2024). SNOWTRAN: a fast radiative transfer model for polar 
hyperspectral remote sensing applications. Remote Sensing, 16(2), 334. 
[DOI](https://doi.org/10.3390/rs16020334)

- Picard, G. and Libois, Q. (2024). Simulation of snow albedo and solar irradiance profile with the Two-streAm 
Radiative TransfEr in Snow (TARTES) v2.0 model, Geosci. Model Dev., 17, 8927–8953. [DOI](https://doi.org/10.5194/gmd-17-8927-2024)

- Robledano, A., Picard, G., Dumont, M., Flin, F., Arnaud, L. and Libois, Q. (2023). Unraveling the optical shape of snow.
  Nature Communications, 14(1), 3955. [DOI](https://doi.org/10.1038/s41467-023-39671-3)
  
</details>
</details>

#### Data

<details>
<summary>Ice refractive indices</summary> 

- Cooper, M. G., Smith, L. C., Rennermalm, A. K., Tedesco, M., Muthyala, R., Leidman, S. Z., ... and Fayne, J. 
V. (2021). Spectral attenuation coefficients from measurements of light transmission in bare ice on the 
Greenland Ice Sheet. The Cryosphere, 15(4), 1931-1953. 
[DOI](https://doi.org/10.5194/tc-15-1931-2021)

- Picard, G., Libois, Q. and Arnaud, L. (2016). Refinement of the ice absorption spectrum in the visible using 
radiance profile measurements in Antarctic snow. The Cryosphere, 10(6), 2655-2672. 
[DOI](https://doi.org/10.5194/tc-10-2655-2016)

- Warren, S. G. and Brandt, R. E. (2008). Optical constants of ice from the ultraviolet to the microwave: A 
revised compilation. Journal of Geophysical Research: Atmospheres, 113(D14). 
[DOI](https://doi.org/10.1029/2007JD009744)
</details>

<details>
<summary>Top-of-atmosphere solar irradiance</summary> 

<br>

- Coddington, O. M., Richard, E. C., Harber, D., Pilewskie, P., Woods, T. N., Snow, M., ... and Sun, K. (2023).
  Version 2 of the TSIS‐1 Hybrid solar reference spectrum and extension to the full spectrum. Earth and Space Science,
  10(3), e2022EA002637. [DOI](https://doi.org/10.1029/2022EA002637)

</details>

<details>
<summary>Atmospheric aerosols</summary> 

<br>

- Hess, Michael, Peter Koepke, and Ingrid Schult. "Optical properties of aerosols and clouds: The software package OPAC."
  Bulletin of the American meteorological society 79.5 (1998): 831-844.
[DOI](https://doi.org/10.1175/1520-0477(1998)079<0831:OPOAAC>2.0.CO;2)

</details>

<details>
<summary>Absorption and profile of atmospheric gases</summary> 

- Anderson, G. P., Clough, S. A., Kneizys, F. X., Chetwynd, J. H. and Shettle, E. P. (1986).
  AFGL atmospheric constituent profiles. Environ. Res. Pap, 954, 1-46.
  
- Gordon, I. E., Rothman, L. S., Hargreaves, R. J., Gomez, F. M., Bertin, T., Hill, C., ... and Zobov, N. F. (2026).
  The HITRAN2024 molecular spectroscopic database. Journal of Quantitative Spectroscopy and Radiative Transfer, 109807.
  [DOI](https://doi.org/10.1016/j.jqsrt.2026.109807)

- Kukkurainen, A., Mikkonen, A., Arola, A., Lipponen, A., Kolehmainen, V. and Sabater, N. (2025).
  HAPI2LIBIS (v1. 0): a new tool for flexible high-resolution radiative transfer computations with libRadtran
  (version 2.0. 5). Geoscientific Model Development, 18(20), 7529-7544. [DOI](https://doi.org/10.5194/gmd-18-7529-2025)

</details>

<details>
<summary>Light absorbing particles</summary> 

- **Ice algae**: Chevrollier, L. A., Cook, J. M., Halbach, L., Jakobsen, H., Benning, L. G., Anesio, A. M. and 
Tranter, M. (2023). Light absorption and albedo reduction by pigmented microalgae on snow and ice. Journal 
of Glaciology, 69(274), 333-341. [DOI](https://doi.org/10.1017/jog.2022.64)

- **Red snow algae**: Chevrollier, L. A., Wehrlé, A., Cook, J. M., Pirk, N., Benning, L. G., Anesio, A. M. and Tranter, 
M. (2025). Separating the albedo-reducing effect of different light-absorbing particles on snow using deep 
learning. The Cryosphere, 19(4), 1527-1538. [DOI](https://doi.org/10.5194/tc-19-1527-2025)

- **Southwestern KN mineral dust**: Chevrollier, L. A., Wehrlé, A., Cook, J. M., Blukis, R., Stevens, I. S., Benning, L. G., Anesio, A. M. and Tranter, M. (2026). Surface processes darkening the southwestern ice sheet of Kalaallit Nunaat (Greenland), in press.

- **Dark cryoconite**: Chevrollier, L. A., Wehrlé, A., Cook, J. M., Blukis, R., Stevens, I. S., Benning, L. G., Anesio, A. M. and Tranter, M. (2026). Surface processes darkening the southwestern ice sheet of Kalaallit Nunaat (Greenland), in press.

- **Black carbon**: Flanner, M. G., Liu, X., Zhou, C., Penner, J. E. and Jiao, C. (2012). Enhanced solar energy 
absorption by internally-mixed black carbon in snow grains. Atmospheric Chemistry and Physics, 12(10), 
4699-4721. [DOI](https://doi.org/10.5194/acp-12-4699-2012)

- **Colorado mineral dust**: Skiles, S. M., Painter, T. and Okin, G. S. (2017). A method to retrieve the spectral 
complex refractive index and single scattering optical properties of dust deposited in mountain snow. 
Journal of Glaciology, 63(237), 133-147. 
[DOI](https://doi.org/10.1017/jog.2016.126)

- **Brown carbon**: Kirchstetter, T. W., Novakov, T. and Hobbs, P. V. (2004). Evidence that the spectral dependence 
of light absorption by aerosols is affected by organic carbon. Journal of Geophysical Research: Atmospheres, 
109(D21). [DOI](https://doi.org/10.1029/2004JD004999)

- **Volcanic ashes**: Flanner, M. G., Gardner, A. S., Eckhardt, S., Stohl, A. and Perket, J. (2014). Aerosol 
radiative forcing from the 2010 Eyjafjallajökull volcanic eruptions. Journal of Geophysical Research: 
Atmospheres, 119(15), 9481-9491. 
[DOI](https://doi.org/10.1002/2014JD021977)

- **Sahara dust**: Balkanski, Y., Schulz, M., Claquin, T. and Guibert, S. (2007). Reevaluation of Mineral aerosol 
radiative forcings suggests a better agreement with satellite and AERONET data. Atmospheric Chemistry and 
Physics, 7(1), 81-95. [DOI](https://doi.org/10.5194/acp-7-81-2007)

- **Greenland dust**: Polashenski, C. M., Dibb, J. E., Flanner, M. G., Chen, J. Y., Courville, Z. R., Lai, A. M., Schauer, J. J., Shafer, M. M., and Bergin, M. (2015). Neither dust nor black carbon causing apparent albedo decline in Greenland's dry snow zone: Implications for MODIS C5 surface reflectance, Geophys. Res. Lett., 42, 9319–9327. [DOI](https://doi.org/10.1002/2015GL065912)

- **Greenland dust Cook**: Cook, J. M., Tedstone, A. J., Williamson, C., McCutcheon, J., Hodson, A. J., Dayal, A., Skiles, M., Hofer, S., Bryant, R., McAree, O., McGonigle, A., Ryan, J., Anesio, A. M., Irvine-Fynn, T. D. L., Hubbard, A., Hanna, E., Flanner, M., Mayanna, S., Benning, L. G., van As, D., Yallop, M., McQuaid, J. B., Gribbin, T., and Tranter, M. (2020). Glacier algae accelerate melt rates on the south-western Greenland Ice Sheet, The Cryosphere, 14, 309–330. [DOI](https://doi.org/10.5194/tc-14-309-2020).

</details>
</details>

#### Software
<details>
<summary>Python libraries</summary> 

- **Numpy**: Harris, C. R., Millman, K. J., Van Der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... and Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357-362. [DOI](https://doi.org/10.1038/s41586-020-2649-2)

- **Pandas**: The pandas development team. (2026). pandas-dev/pandas: Pandas (v3.0.3). Zenodo. [DOI](https://doi.org/10.5281/zenodo.20127038)

- **Matplotlib**: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in science & engineering, 9(3), 90-95. [DOI](https://doi.org/10.1109/MCSE.2007.55)

- **Netcdf4**: [Github repository](https://github.com/Unidata/netcdf4-python)

- **Scipy**: Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272. [DOI](https://doi.org/10.1038/s41592-019-0686-2).

 - **Xarray**: Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. [DOI](https://doi.org/10.5334/jors.148)

- **Pyyaml**: [Github repository](https://github.com/yaml/pyyaml)

- **Pydantic**: Colvin, S., Jolibois, E., Ramezani, H., Garcia Badaracco, A., Dorsey, T., Montague, D., Matveenko, S., Trylesinski, M., Runkle, S., Hewitt, D., Hall, A. and Plot, V. (2026). Pydantic Validation (Version 2.14.0a1) [Computer software]. [DOI](https://doi.org/10.5281/zenodo.8180180)

- **PythonicDISORT** : Ho, D. X. (2024). PythonicDISORT: A Python reimplementation of theDiscrete Ordinate Radiative Transfer package DISORT. Journal of Open Source Software, 9(103).[DOI](https://doi.org/10.21105/joss.06442)
  
- **Pooch** : Uieda, L., Soler, S. R., Rampin, R., van Kemenade, H., Turk, M., Shapero, D., Banihirwe, A. and Leeman, J. (2020). Pooch: A friend to fetch your data files. Journal of Open Source Software, 5(45), 1943. [DOI](https://doi.org/10.21105/joss.01943)

</details>

## Citation

If you use SNICAR-fx for research, please include a reference to the overall software as well as the appropriate solver. For the citation of specific features or datasets, please see [References](#references).

- Software: Zenodo archive (for reproducibility) and/or Chevrollier et al. 2026 (for application and validation)
- Solver: [Ho et al. 2024](https://doi.org/10.21105/joss.06442) and [Stamnes et al 2000](https://web.gps.caltech.edu/~vijay/Papers/RT%20Models/DISORT%20Report.pdf) for 'multi-stream-disort', [Liu and Weng 2013](https://doi.org/10.1175/JAS3808.1) for 'multi-stream-ada', [Briegleb and Light 2007](https://doi.org/10.5065/D6B27S71) and [Whicker et al. 2022](https://doi.org/10.5194/tc-16-1197-2022) for 'two-stream-ad'
