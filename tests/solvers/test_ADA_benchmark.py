"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

import numpy as np

from snicarfx.core import (
    ColumnProperties,
    ModelInputs,
    SolarIrradiance,
    solve_advanced_adding_doubling,
)


def test_multistream_outputs(
    params_ada,
    column,
    benchmark_ada_spectral_data,
    absolute_tolerance_benchmark,
):
    """
    Assert that snicar-fx reproduces the spectral albedo modelled using the 
    ADA module of the Community Radiative Transfer Model (CRTM) within a tolerance
    of 1e-5 when the exact same input data is used. Input data is defined via
    the input test file and the `params_ada` parameter.
     
    
    Parameters
    ----------
    params_ada : array
        Sets of parameters used as input for the model.
    column : ColumnProperties
        Instance of the ColumnProperties class
    benchmark_ada_spectral_data : array
        Spectral albedo data generated using the Fortran-based ADA module of 
        CRTM for the parameter grid `params`.
    absolute_tolerance_benchmark: float
        Tolerance value for the error.
    
    """
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
        benchmark_ada_spectral_data.sel(w=w, t_od=t_od, g=g, wvl_idx=wvl_idx)[
            "albedo"
        ].values,
        atol=absolute_tolerance_benchmark,
    )
