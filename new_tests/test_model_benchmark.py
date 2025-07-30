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


@pytest.mark.parametrize("params_idx", range(len(parameter_grid)))
def test_snicarfx_outputs(params_idx, benchmark_data):

    layer_type, density, reff, zen, bc, dz = parameter_grid[params_idx]

    # Setup inputs
    model_config = ModelConfig("./tests/inputs_tests.yaml")
    column = ColumnProperties(model_config)
    irradiance = SolarIrradiance(model_config)

    irradiance.solzen = zen
    irradiance.calculate_irradiance()

    column.thickness = dz
    column.layer_type = [layer_type] * len(dz)
    column.rho = [density] * len(dz)

    snow_idx = np.where(np.array(column.layer_type) == 0)[0]
    ice_idx = np.where(np.array(column.layer_type) != 0)[0]
