# SNICAR-fx: A flexible and light-weight version of the SNICAR model

SNICAR-fx started as a fork of the biosnicar-py repository, a python translation of the SNICAR (SNow, ICe and Aerosols Radiative transfer) model, to then evolve into its own standalone version. SNICAR-fx solves the 1-D unpolarized radiative transfer equation for a column of snow and/or ice using an adding-doubling solver, originally developed by Briegleb et al. 2007 and improved by Whicker et al. 2022 to include spectrally-dependent Fresnel reflectance coefficients. 

What makes it different from SNICAR and BioSNICAR?
- flexible spectral range (between 200 and 5000nm with resolution >= 1nm)
- light weight: no dependence on large optical properties databases
- as-much-as possible vectorixation of the code
- optical properties for snow/ice use geometric optics approximations instead of Mie theory
- Liquid water in snow and ice 

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

Original SNICAR model:
Flanner, M. et al. (2007): Present-day climate forcing and response from black carbon in snow, J. Geophys. Res., 112, D11202, https://doi.org/10.1029/2006JD008003
Toon, O. B., McKay, C. P., Ackerman, T. P., and Santhanam, K. (1989), Rapid calculation of radiative heating rates and photodissociation rates in inhomogeneous multiple scattering atmospheres, J. Geophys. Res., 94( D13), 16287– 16301, doi:10.1029/JD094iD13p16287.

Adding doubling solver:
Briegleb, B. P., and B. Light. "A Delta-Eddington multiple scattering parameterization for solar radiation in the sea ice component of the Community Climate System Model." NCAR technical note (2007).
Dang, C., Zender, C., Flanner M. 2019. Intercomparison and improvement of two-stream shortwave radiative transfer schemes in Earth system models for a unified treatment of cryospheric surfaces. The Cryosphere, 13, 2325–2343, https://doi.org/10.5194/tc-13-2325-2019
Flanner, M. G. et al., SNICAR-ADv3: a community tool for modeling spectral snow albedo, Geosci. Model Dev., 14, 7673–7704, https://doi.org/10.5194/gmd-14-7673-2021, 2021.

Fresnel equations:
Whicker, C. A., et al. "SNICAR-ADv4: a physically based radiative transfer model to represent the spectral albedo of glacier ice." The Cryosphere 16.4 (2022): 1197-1220.

Biosnicar:
Cook, J. M. et al. (2020): Glacier algae accelerate melt rates on the western Greenland Ice Sheet, The Cryosphere Discuss., https://doi.org/10.5194/tc-2019-58, in review, 2019.

Snow and ice algae optical properties:
Chevrollier, L-A., et al. "Light absorption and albedo reduction by pigmented microalgae on snow and ice." Journal of Glaciology 69.274 (2023): 333-341.

Other optical properties:
Balkanski, Y., Schulz, M., Claquin, T., & Guibert, S. (2007). Reevaluation of Mineral aerosol radiative forcings suggests a better agreement with satellite and AERONET data. Atmospheric Chemistry and Physics, 7(1), 81-95.

Flanner, M et al. (2009) Springtime warming and reduced snow cover from
carbonaceous particles. Atmospheric Chemistry and Physics, 9: 2481-2497, 2009.

Flanner, M. G., Gardner, A. S., Eckhardt, S., Stohl, A., & Perket, J. (2014). Aerosol radiative forcing from the 2010 Eyjafjallajökull volcanic eruptions. Journal of Geophysical Research: Atmospheres, 119(15), 9481-9491.

Kirchstetter, T. W., Novakov, T., & Hobbs, P. V. (2004). Evidence that the spectral dependence of light absorption by aerosols is affected by organic carbon. Journal of Geophysical Research: Atmospheres, 109(D21).
Polashenski et al. (2015): Neither dust nor black carbon causing apparent albedo decline in Greenland's dry snow zone: Implications for MODIS C5 surface reflectance, Geophys. Res. Lett., 42, 9319– 9327, doi:10.1002/2015GL065912, 2015.
Skiles, S. M., Painter, T., & Okin, G. S. (2017). A method to retrieve the spectral complex refractive index and single scattering optical properties of dust deposited in mountain snow. Journal of Glaciology, 63(237), 133-147.

Ice refractive indices:

Picard, G., Libois, Q., & Arnaud, L. (2016). Refinement of the ice absorption spectrum in the visible using radiance profile measurements in Antarctic snow. The Cryosphere, 10(6), 2655-2672.

Warren, S. G., & Brandt, R. E. (2008). Optical constants of ice from the ultraviolet to the microwave: A revised compilation. Journal of Geophysical Research: Atmospheres, 113(D14).
