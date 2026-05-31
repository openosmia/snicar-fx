"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np

from snicarfx.core.solvers.multi_stream_solver_ada import solve_multi_stream_rt_ada


def test_multistream_outputs(
    session_multistream_uncoupled,
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
    session_multistream_uncoupled : Session
        Instance of Session class from snicar-fx.
    params_ada : tuple
        Sets of parameters used as input for the model.
    benchmark_ada_spectral_data : array
        Spectral albedo data generated using the Fortran-based ADA module of
        CRTM for the parameter grid `params_ada`.
    absolute_tolerance_benchmark: float
        Tolerance value for the error.

    """

    w, t_od, g, wvl_idx = params_ada

    # Setup inputs
    land = session_multistream_uncoupled.land
    solar = session_multistream_uncoupled.solar
    atmosphere = session_multistream_uncoupled.atmosphere

    land.ss_alb[:, :] = w
    land.tau[:, :] = t_od
    land.asm_prm[:, :] = g

    # legendre moments
    land.legendre_moments = (
        land.asm_prm[None, :, :] ** np.arange(land.n_expansion + 2)[:, None, None]
    )

    # solve RTE
    results = solve_multi_stream_rt_ada(
        land, atmosphere, solar, session_multistream_uncoupled.config
    )

    # a given set of parameters (including a given wavelength)
    assert np.allclose(
        results["albedo_boa"][wvl_idx],
        benchmark_ada_spectral_data.sel(w=w, t_od=t_od, g=g, wavelength_index=wvl_idx)[
            "albedo"
        ].values,
        atol=absolute_tolerance_benchmark,
        rtol=0.0,
    )
