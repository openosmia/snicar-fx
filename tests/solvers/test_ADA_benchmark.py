#!/usr/bin/env python3
"""
Test snicar-fx two-stream and multi-stream outputs against outputs of the
two-stream SNICAR_ADv4 Matlab code

"""

import numpy as np
import pytest
import xarray as xr

from snicarfx.core import (
    ColumnProperties,
    ModelInputs,
    SolarIrradiance,
    solve_advanced_adding_doubling,
)
from tests.conftest import multistream_ADA_parameter_grid


def test_multistream_outputs(
    idx_ada,
    params_ada,
    column,
    benchmark_ADA_spectral_data,
    absolute_tolerance_benchmark,
):

    w, t_od, g, wvl_idx = params_ada

    # Setup inputs
    model_inputs = ModelInputs("./tests/inputs_tests.yaml")
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs)

    column.ss_alb[:, wvl_idx] = w
    column.tau[:, wvl_idx] = t_od
    column.asm_prm[:, wvl_idx] = g
    column.update_column_ops_with_laps()
    column.update_column_ops_with_laps()

    # solve RTE
    albedo = solve_advanced_adding_doubling(column, irradiance)

    # a given set of parameters (including a given wavelength)
    assert np.allclose(
        albedo[wvl_idx],
        benchmark_ADA_spectral_data.sel(w=w, t_od=t_od, g=g, wvl_idx=wvl_idx)[
            "albedo"
        ].values,
        atol=absolute_tolerance_benchmark,
    )
