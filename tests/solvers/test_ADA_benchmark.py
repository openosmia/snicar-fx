"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np

from snicarfx.core.solvers.multi_stream_solver import solve_multi_stream_rt


def test_multistream_outputs(
    session,
    params_ada,
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
    land_column = session.land_column
    irradiance = session.solar_irradiance
    atmosphere = session.atmosphere_column

    land_column.ss_alb[:, :] = w
    land_column.tau[:, :] = t_od
    land_column.asm_prm[:, :] = g

    # legendre moments
    land_column.legendre_moments = (
        land_column.asm_prm[None, :, :]
        ** np.arange(land_column.n_expansion)[:, None, None]
    )

    # solve RTE
    results = solve_multi_stream_rt(
        land_column,
        atmosphere,
        irradiance,
        session.config.SOLVER.OUTPUT_LEVELS,
        session.config.SOLVER.N_STREAMS,
    )

    # a given set of parameters (including a given wavelength)
    assert np.allclose(
        results["albedo_boa"][wvl_idx],
        benchmark_ada_spectral_data.sel(w=w, t_od=t_od, g=g, wavelength_index=wvl_idx)[
            "albedo"
        ].values,
        atol=absolute_tolerance_benchmark,
    )
