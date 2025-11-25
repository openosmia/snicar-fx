"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import xarray as xr
import numpy as np


def use_data_snicaradv4(column, irradiance):
    """
    Ensures the refractive index and fresnel coefficients used to generate
    snicar-fx benchmark data to test snicar-fx against SNICAR-ADv4 correspond
    to the values used in SNICAR-ADv4.

    Parameters
    ----------
    column : ColumnProperties
        Instance of the ColumnProperties class

    Returns
    ----------
    column : ColumnProperties
        Updated instance of the ColumnProperties class

    """

    column.ref_idx_im = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).im_Pic16.values
    column.ref_idx_re = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).re_Pic16.values
    column.fl_r_dif_b = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fb_ice_Pic16.values
    column.fl_r_dif_a = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fa_ice_Pic16.values
    
    irradiance.cos_sza = np.cos(np.deg2rad(np.rint(irradiance.sza)))
    
    if irradiance.direct == 1: 
        irradiance.flx_slr = xr.open_dataset(
            './tests/test_data/swnb_480bnd_'
                + "mls_clr_"
                + str("SZA" + str(irradiance.sza).rjust(2, "0"))
                + ".nc"
            )["flx_frc_sfc"].values 
        irradiance.flx_slr[irradiance.flx_slr == 0] = 1e-30
        irradiance.fs = (irradiance.flx_slr 
                          / (irradiance.cos_sza * np.pi)
                          )
        irradiance.fd = np.zeros_like(irradiance.fs)

        
    else: 
        irradiance.flx_slr = xr.open_dataset(
            './tests/test_data/swnb_480bnd_mls_cld.nc'
            )["flx_frc_sfc"].values
        irradiance.flx_slr[irradiance.flx_slr == 0] = 1e-30
        irradiance.fd = irradiance.flx_slr 
        irradiance.fs = np.zeros_like(irradiance.fd)
        

    return column, irradiance
