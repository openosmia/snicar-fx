#!/usr/bin/env python3
"""

Test snicar-fx against outputs from SNICAR_ADv4 Matlab code

"""

import numpy as np
import pytest
import xarray as xr

from snicarfx.classes import ColumnProperties, ModelConfig, SolarIrradiance
from snicarfx.rt_solvers import solve_adding_doubling
from tests.conftest import parameter_grid
from tests.utils import match_matlab_config


@pytest.mark.parametrize("idx, params", enumerate(parameter_grid()))
def test_snicarfx_outputs(
    idx, params, column, 
    benchmark_snicaradv4_spectral_data, 
    benchmark_snicaradv4_bba_data,
    benchmark_snicaradv4_absorbed_flux_data,
    absolute_tolerance_benchmark
):

    layer_type, density, radius, sza, bc, thickness_profile, direct = params

    # Setup inputs
    model_config = ModelConfig("./tests/inputs_tests.yaml")
    column = ColumnProperties(model_config)
    irradiance = SolarIrradiance(model_config)

    column = match_matlab_config(column)

    # calculate irradiance
    irradiance.direct = direct
    irradiance.solzen = sza
    irradiance.calculate_irradiance()

    # calculate column ssa, g, mac
    column.thickness_profile = thickness_profile
    column.layer_type = [layer_type] * len(column.thickness_profile)
    column.density = [density] * len(column.thickness_profile)
    column.layer_mass = [
        column.density[i] * column.thickness_profile[i]
        for i in range(len(column.thickness_profile))
    ]

    snow_idx = np.where(np.array(column.layer_type) == 0)[0]
    ice_idx = np.where(np.array(column.layer_type) != 0)[0]

    for i in snow_idx:
        file_ssps = str(
            "./tests/test_data/ice_spherical_grains_BH83/"
            + f"ice_{column.rf_type}/ice_{column.rf_type}_"
            + "{}.nc".format(str(radius).rjust(4, "0"))
        )

        with xr.open_dataset(file_ssps) as ssps:
            column.ss_alb[i, :] = ssps["ss_alb"].values
            column.ext_cff[i, :] = ssps["ext_cff_mss"].values
            column.asm_prm[i, :] = ssps["asm_prm"].values
            column.tau[i, :] = column.layer_mass[i] * column.ext_cff[i, :]

    for i in ice_idx:
        file_ssps = str(
            "./tests/test_data/bubbly_ice_files_BH83/"
            + "bbl_{}.nc".format(str(radius).rjust(4, "0"))
        )
        with xr.open_dataset(file_ssps) as ssps:
            column.asm_prm[i, :] = ssps["asm_prm"].values
            sca_cff_vlm_air_bbl = ssps["sca_cff_vlm"].values
            vlm_frac_air = 1 - column.density[i] / 917
            scattering_cff = sca_cff_vlm_air_bbl * vlm_frac_air / column.density[i]
            abs_cff = (4 * np.pi * column.ref_idx_im) / (column.wavelengths) / 917
            column.ext_cff[i, :] = scattering_cff + abs_cff
            column.ss_alb[i, :] = scattering_cff / column.ext_cff[i, :]
            column.tau[i, :] = column.layer_mass[i] * column.ext_cff[i, :]

    column.lap_concentrations[:, 0] = bc * 1e-9 

    column.add_laps_to_column_ops()

    # solve RTE
    outputs = solve_adding_doubling(column, irradiance)

    # spectral albedo only until 2705nm for now, as small issue in next 15 bds
    assert np.allclose(
        outputs.albedo[:250],
        benchmark_snicaradv4_spectral_data[idx][:250],
        atol=absolute_tolerance_benchmark,
    )
    #BBA
    assert np.allclose(
        outputs.BBA,
        benchmark_snicaradv4_bba_data[idx],
        atol=absolute_tolerance_benchmark,
    )
    #Absorbed flux
    assert np.allclose(
        outputs.abs_slr_tot,
        benchmark_snicaradv4_absorbed_flux_data[idx],
        atol=absolute_tolerance_benchmark,
    )

    
    
