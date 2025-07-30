#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test snicar-fx against outputs from SNICAR_ADv4 Matlab code

"""

from snicarfx.solver import adding_doubling_solver
from snicarfx.classes import ColumnProperties, SolarIrradiance, ModelConfig
import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("idx, params", enumerate(parameter_grid))
def test_snicarfx_outputs(params_idx, column, benchmark_data):

    layer_type, density, reff, zen, bc, dz = params

    # Setup inputs
    model_config = ModelConfig("./tests/inputs_tests.yaml")
    column = ColumnProperties(model_config)
    irradiance = SolarIrradiance(model_config)

    irradiance.solzen = zen
    irradiance.calculate_irradiance()

    column.thickness = dz
    column.layer_type = [layer_type] * len(dz)
    column.density = [density] * len(dz)

    column.layer_mass = [
        column.density[i] * column.thickness[i] for i in range(len(column.thickness))
    ]

    snow_idx = np.where(np.array(column.layer_type) == 0)[0]
    ice_idx = np.where(np.array(column.layer_type) != 0)[0]

    for i in snow_idx:
        file_ssps = str(
            "./tests/test_data/ice_spherical_grains_BH83/"
            + f"ice_{column.rf_type}/ice_{column.rf_type}_"
            + "{}.nc".format(str(reff).rjust(4, "0"))
        )

        with xr.open_dataset(file_ssps) as ssps:
            column.ss_alb[i, :] = ssps["ss_alb"].values
            column.ext_cff[i, :] = ssps["ext_cff_mss"].values
            column.asm_prm[i, :] = ssps["asm_prm"].values
            column.tau[i, :] = column.layer_mass[i] * column.ext_cff[i, :]

    for i in ice_idx:
        file_ssps = str(
            "./tests/test_data/bubbly_ice_files_BH83/"
            + "bbl_{}.nc".format(str(reff).rjust(4, "0"))
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

    column.lap_concentrations[:, 0] = [
        bc * 1e-9,
        bc * 1e-9,
        bc * 1e-9,
        bc * 1e-9,
        bc * 1e-9,
    ]

    column.add_laps_to_column_ops()

    outputs = adding_doubling_solver(column, irradiance)

    error = np.abs(outputs.albedo - benchmark_matlab_data[idx])

    assert np.allclose(
        outputs.albedo,
        benchmark_matlab_data[idx],
        atol=absolutex_tolerance_benchmark,
    )
