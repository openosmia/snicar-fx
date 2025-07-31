#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """

import xarray as xr


def match_matlab_config(column):
    """Ensures model config is equal to the Matlab version used to
    generate benchmark data.

    This function resets values in instances of Ice, Illumination and
    ModelConfig to ensure equivalence between BioSNICAR and the Matlab
    code used to generate the benchmark data.  Also ensures all vars
    have correct length, and re-executes the class functions in Ice
    and Illumination that update refractive indices and at-surface
    irradiance.

    Args:
        column: instance of Ice class
        irradiance: instance of SolarIrradiance class
        model_config: instance of ModelConfig class

    Returns:

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

    return column
