"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


def use_data_snicaradv4(land, solar):
    """
    Ensures the configuration for the refractive index, fresnel coefficients
    and surface irradiance used in snicar-fx match that of SNICAR-ADv4.

    Parameters
    ----------
    land : LandColumn
        Instance of the LandColumn class

    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    Returns
    ----------
    land : LandColumn
        Instance of the LandColumn class

    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    """

    land.ref_idx_im = xr.open_dataset("./tests/test_data/rfidx_ice.nc").im_Pic16.values
    land.ref_idx_re = xr.open_dataset("./tests/test_data/rfidx_ice.nc").re_Pic16.values
    land.fl_r_dif_b = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fb_ice_Pic16.values
    land.fl_r_dif_a = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fa_ice_Pic16.values

    if solar.sky_conditions == "clear":
        solar.total_irradiance = xr.open_dataset(
            "./tests/test_data/swnb_480bnd_"
            + "mls_clr_"
            + str("SZA" + str(solar.sza).rjust(2, "0"))
            + ".nc"
        )["flx_frc_sfc"].values
        solar.total_irradiance[solar.total_irradiance == 0] = 1e-30

        solar.direct_beam = solar.total_irradiance / np.cos(np.deg2rad(solar.sza))
        solar.diffuse = np.zeros_like(solar.total_irradiance)

    elif solar.sky_conditions == "cloudy":
        solar.total_irradiance = xr.open_dataset(
            "./tests/test_data/swnb_480bnd_mls_cld.nc"
        )["flx_frc_sfc"].values
        solar.total_irradiance[solar.total_irradiance == 0] = 1e-30
        solar.diffuse = solar.total_irradiance / np.pi
        solar.direct_beam = np.zeros_like(solar.total_irradiance)

    return land, solar
