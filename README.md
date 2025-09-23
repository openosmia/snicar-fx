 [![License: ACSL](https://img.shields.io/badge/License-ACSL-red.svg)](https://anticapitalist.software/)
 [![Continuous integration](https://github.com/openosmia/snicar-fx/workflows/CI/badge.svg)](https://github.com/openosmia/snicar-fx/actions)
 [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
 [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Validated with Pydantic](https://img.shields.io/badge/Validated%20with-Pydantic-4FC08D?style=flat&logo=pydantic)](https://docs.pydantic.dev/)

# SNICAR-fx

### *A flexible, light-weight, fast and feature-enhanced adaptation of the SNICAR model*

SNICAR-fx started with the idea of upgrading the
[biosnicar-py](https://github.com/jmcook1186/biosnicar-py) software,
a python translation of
[SNICAR-ADv4](https://github.com/chloewhicker/SNICAR-ADv4) (SNow, ICe
and Aerosols Radiative model), to a version that would be more portable
and enable a flexible spectral range and resolution. Since then, it 
evolved into a distinct model with expanded functionality and its own
strengths and assumptions.


## What makes SNICAR-fx different from SNICAR-ADv4?

- **flexible spectral grid**: between 200 and 5000nm with resolution >= 1nm
- **light weight**: no dependence on large optical properties databases
- **fast**: up to 50x faster (single-threaded) depending on the model
  set-up (nb layers in particular)
- **directional capability**: two-stream and multi-stream solver (! no fresnel layers and diffuse irradiance for now)
- **specialized snow and ice features**: liquid water content and empirical optical properties of glacial microbes

## Brief description

SNICAR-fx solves the 1-D unpolarized radiative transfer equation for a
column of homogeneous layers of snow and/or ice. Each layer can be
represented as a bulk medium made of ice grains or
air bubbles, of which size should be larger than the wavelength 
because SNICAR-fx employs geometric optics assumptions to model 
single scattering properties, in contrast to SNICAR-ADv4 which uses
Mie theory. Each layer has a specific surface area, water content 
and concentrations of various light absorbing particles, and the
incoming irradiance can be direct with a prescribed
SZA, or diffuse. Fresnel boundary layers can be incorporated between
layers to account for the change in refractive index between air and ice 
when using the two-stream Delta-Eddingon solver (
[Briegleb and Light 2007](https://doi.org/10.5065/D6B27S71),
[Whicker et al. 2022](https://doi.org/10.5194/tc-16-1197-2022)). 
A multi-stream delta-M solver employing the
advanced matrix operator method is also available 
([Liu and Weng 2006](https://doi.org/10.1175/JAS3808.1), 
[Liu and Weng 2013](https://doi.org/10.1109/JSTARS.2013.2247026)),
but does not support Fresnel layers and diffuse irradiance for
now. 

SNICAR-fx is currently developed with a focus on melting environments
and the radiative forcing of light absorbing particles. More specifically, 
the development currently targets melting weathering crust environments and
the software recently incorporated an empirical ice refractive index 
([Cooper et al. 2021](https://doi.org/10.5194/tc-15-1931-2021)) as well 
as empirical optical properties of various glacial microbes and the option 
to include liquid water within the column.

## How to use

### Installation

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

Install the `snicarfx` software. Using `pip` together with `conda` is usually a bad idea, but here conda installs all the dependencies and pip only sets up the associated paths, that's all!
```bash
pip install -e .
```

### Running the code

Example scripts are provided in `/examples`. So far only single runs using the two-stream solver are provided, but code for batch runs is coming!


## References

### Equations and model formulation
<details>
<summary>Original SNICAR equations (Two-stream Delta-Eddington formulation)</summary>

- Joseph, J. H., Wiscombe, W. J., & Weinman, J. A. (1976). The delta-Eddington approximation for radiative flux 
transfer. Journal of Atmospheric Sciences, 33(12), 2452-2459. 
[DOI](https://doi.org/10.1175/1520-0469(1976)033<2452:TDEAFR>2.0.CO;2)

- Wiscombe, W. J., & Warren, S. G. (1980). A model for the spectral albedo of snow. I: Pure snow. Journal of 
Atmospheric Sciences, 37(12), 2712-2733.
[DOI](https://doi.org/10.1175/1520-0469(1980)037<2712:AMFTSA>2.0.CO;2)

- Flanner, M. G., Arnheim, J., Cook, J. M., Dang, C., He, C., Huang, X., ... & Zender, C. S. (2021). SNICAR-AD 
v3: A community tool for modeling spectral snow albedo. Geoscientific Model Development, 2021, 1-49. 
[DOI](https://doi.org/10.5194/gmd-14-7673-2021)
</details>

<details>
<summary>Adding-doubling solver with Fresnel layers</summary>

- Briegleb, P., & Light, B. (2007). A Delta-Eddington mutiple scattering parameterization for solar radiation 
in the sea ice component of the community climate system model. 
[DOI](https://doi.org/10.5065/D6B27S71)

- Whicker, C. A., Flanner, M. G., Dang, C., Zender, C. S., Cook, J. M., & Gardner, A. S. (2021). SNICAR-ADv4: 
a physically based radiative transfer model to represent the spectral albedo of glacier ice. The Cryosphere, 
2021, 1-36. [DOI](https://doi.org/10.5194/tc-16-1197-2022)
</details>

<details>
<summary>Multi-stream advanced matrix operator method and adding solver</summary> 

<br>

- Liu, Q., & Weng, F. (2013). Using advanced matrix operator (AMOM) in community radiative transfer model. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 6(3), 1211-1218. 
[DOI](https://doi.org/10.1109/JSTARS.2013.2247026)
</details>

<details>
<summary>Air bubble asymmetry parameter</summary> 
 
<br>

- Kokhanovsky, A. A. (2002). Optical properties of bubbles. Journal of Optics A: Pure and Applied Optics, 
5(1), 47. [DOI](https://doi.org/10.1088/1464-4258/5/1/307)
</details>

<details>
<summary>Ice grain single scattering properties (similar as in <a href="https://github.com/ghislainp/tartes">TARTES</a>)</summary>

- Kokhanovsky, A. A. (2021). Snow optics. Springer. 
[DOI](https://doi.org/10.1007/978-3-031-85979-3)</summary> 

- Kokhanovsky, A. A., & Macke, A. (1997). Integral light-scattering and absorption characteristics of large, nonspherical particles. 
Applied optics, 36(33), 8785-8790. [DOI](https://doi.org/10.1364/AO.36.008785)

- Kokhanovsky, A., Brell, M., Segl, K., & Chabrillat, S. (2024). SNOWTRAN: a fast radiative transfer model for polar 
hyperspectral remote sensing applications. Remote Sensing, 16(2), 334. 
[DOI](https://doi.org/10.3390/rs16020334)

- Picard, G. and Libois, Q. (2024). Simulation of snow albedo and solar irradiance profile with the Two-streAm 
Radiative TransfEr in Snow (TARTES) v2.0 model, Geosci. Model Dev., 17, 8927–8953. [DOI](https://doi.org/10.5194/gmd-17-8927-2024)
</details>

### Data

<details>
<summary>Ice refractive indices</summary> 

- Cooper, M. G., Smith, L. C., Rennermalm, A. K., Tedesco, M., Muthyala, R., Leidman, S. Z., ... & Fayne, J. 
V. (2021). Spectral attenuation coefficients from measurements of light transmission in bare ice on the 
Greenland Ice Sheet. The Cryosphere, 15(4), 1931-1953. 
[DOI](https://doi.org/10.5194/tc-15-1931-2021)

- Picard, G., Libois, Q., & Arnaud, L. (2016). Refinement of the ice absorption spectrum in the visible using 
radiance profile measurements in Antarctic snow. The Cryosphere, 10(6), 2655-2672. 
[DOI](https://doi.org/10.5194/tc-10-2655-2016)

- Warren, S. G., & Brandt, R. E. (2008). Optical constants of ice from the ultraviolet to the microwave: A 
revised compilation. Journal of Geophysical Research: Atmospheres, 113(D14). 
[DOI](https://doi.org/10.1029/2007JD009744)
</details>

<details>
<summary>Spectral irradiances (SWNB2 model runs)</summary> 

<br>

- Flanner, M. G., Arnheim, J., Cook, J. M., Dang, C., He, C., Huang, X., ... & Zender, C. S. (2021). SNICAR-AD
v3: A community tool for modeling spectral snow albedo. Geoscientific Model Development, 2021, 1-49.
[DOI](https://doi.org/10.5194/gmd-14-7673-2021)
</details>

<details>
<summary>Light absorbing particles</summary> 

- **Black carbon**: Flanner, M. G., Liu, X., Zhou, C., Penner, J. E., & Jiao, C. (2012). Enhanced solar energy 
absorption by internally-mixed black carbon in snow grains. Atmospheric Chemistry and Physics, 12(10), 
4699-4721. [DOI](https://doi.org/10.5194/acp-12-4699-2012)

- **Ice algae**: Chevrollier, L. A., Cook, J. M., Halbach, L., Jakobsen, H., Benning, L. G., Anesio, A. M., & 
Tranter, M. (2023). Light absorption and albedo reduction by pigmented microalgae on snow and ice. Journal 
of Glaciology, 69(274), 333-341. [DOI](https://doi.org/10.1017/jog.2022.64)

- **Colorado mineral dust**: Skiles, S. M., Painter, T., & Okin, G. S. (2017). A method to retrieve the spectral 
complex refractive index and single scattering optical properties of dust deposited in mountain snow. 
Journal of Glaciology, 63(237), 133-147. 
[DOI](https://doi.org/10.1017/jog.2016.126)

- **Brown carbon**: Kirchstetter, T. W., Novakov, T., & Hobbs, P. V. (2004). Evidence that the spectral dependence 
of light absorption by aerosols is affected by organic carbon. Journal of Geophysical Research: Atmospheres, 
109(D21). [DOI](https://doi.org/10.1029/2004JD004999)

- **Snow algae**: Chevrollier, L. A., Wehrlé, A., Cook, J. M., Pirk, N., Benning, L. G., Anesio, A. M., & Tranter, 
M. (2025). Separating the albedo-reducing effect of different light-absorbing particles on snow using deep 
learning. The Cryosphere, 19(4), 1527-1538. 
[DOI](https://doi.org/10.5194/tc-19-1527-2025)

- **Volcanic ashes**: Flanner, M. G., Gardner, A. S., Eckhardt, S., Stohl, A., & Perket, J. (2014). Aerosol 
radiative forcing from the 2010 Eyjafjallajökull volcanic eruptions. Journal of Geophysical Research: 
Atmospheres, 119(15), 9481-9491. 
[DOI](https://doi.org/10.1002/2014JD021977)

- **Greenland dust**: Cook, J. M., Tedstone, A. J., Williamson, C., McCutcheon, J., Hodson, A. J., Dayal, A., ... 
& Tranter, M. (2020). Glacier algae accelerate melt rates on the south-western Greenland Ice Sheet. The 
Cryosphere, 14(1), 309-330. [DOI](https://doi.org/10.5194/tc-14-309-2020)

- **Sahara dust**: Balkanski, Y., Schulz, M., Claquin, T., & Guibert, S. (2007). Reevaluation of Mineral aerosol 
radiative forcings suggests a better agreement with satellite and AERONET data. Atmospheric Chemistry and 
Physics, 7(1), 81-95. [DOI](https://doi.org/10.5194/acp-7-81-2007)

</details>
