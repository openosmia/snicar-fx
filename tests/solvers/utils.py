"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


def use_data_snicaradv4(land_column, irradiance):
    """
    Ensures the refractive index and fresnel coefficients used to generate
    snicar-fx benchmark data to test snicar-fx against SNICAR-ADv4 correspond
    to the values used in SNICAR-ADv4.

    Parameters
    ----------
    land_column : LandColumn
        Instance of the LandColumn class
    
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    Returns
    ----------
    land_column : LandColumn
        Instance of the LandColumn class
    
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    """

    land_column.ref_idx_im = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).im_Pic16.values
    land_column.ref_idx_re = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).re_Pic16.values
    land_column.fl_r_dif_b = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fb_ice_Pic16.values
    land_column.fl_r_dif_a = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fa_ice_Pic16.values
    
    if irradiance.sky_conditions == "clear":
        irradiance.total_irradiance = xr.open_dataset(
            "./tests/test_data/swnb_480bnd_"
            + "mls_clr_"
            + str("SZA" + str(irradiance.sza).rjust(2, "0"))
            + ".nc"
        )["flx_frc_sfc"].values
        irradiance.total_irradiance[irradiance.total_irradiance == 0] = 1e-30

        irradiance.direct_beam = irradiance.total_irradiance / np.cos(np.deg2rad(irradiance.sza))
        irradiance.diffuse = np.zeros_like(irradiance.total_irradiance)

    elif irradiance.sky_conditions == "cloudy":
        irradiance.total_irradiance = xr.open_dataset(
            "./tests/test_data/swnb_480bnd_mls_cld.nc"
        )["flx_frc_sfc"].values
        irradiance.total_irradiance[irradiance.total_irradiance == 0] = 1e-30
        irradiance.diffuse = irradiance.total_irradiance / np.pi
        irradiance.direct_beam = np.zeros_like(irradiance.total_irradiance)

    return land_column, irradiance
