"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np

from snicarfx.core.solvers.multi_stream_solver_disort import (
    solve_multi_stream_rt_disort,
    solve_multi_stream_rt_disort_wrapper,
)


def test_pythonicdisort_outputs(
    session_multistream_coupled,
    multistream_pythonicdisort_params,
    absolute_tolerance_pythonicdisort,
):
    """
    Test that outputs from the backend PythonicDISORT routine
    within SNICAR-fx strictly match those from the high-level
    PythonicDISORT wrapper (configuration with delta-M truncation
                            and no intensity correction).

    Parameters
    ----------
    session_multistream_coupled : Session
        Instance of Session class from snicar-fx.
    multistream_pythonicdisort_params : array
        Sets of parameters used as input for the model.
    absolute_tolerance_pythonicdisort: float
        Tolerance value for the error.

    """

    sza, saa, _azimuth, aod = multistream_pythonicdisort_params

    session_multistream_coupled.solar.sza = sza
    session_multistream_coupled.solar.saa = saa
    session_multistream_coupled.atmosphere.AOD550 = aod

    results_backend = solve_multi_stream_rt_disort(
        session_multistream_coupled.land,
        session_multistream_coupled.atmosphere,
        session_multistream_coupled.solar,
        session_multistream_coupled.config,
    )

    results_wrapper = solve_multi_stream_rt_disort_wrapper(
        session_multistream_coupled.land,
        session_multistream_coupled.atmosphere,
        session_multistream_coupled.solar,
        session_multistream_coupled.config,
        nt_cor=False,
    )

    assert np.allclose(
        results_backend["albedo_toa"],
        results_wrapper["albedo_toa"],
        atol=absolute_tolerance_pythonicdisort,
        rtol=0.0,
    )
    assert np.allclose(
        results_backend["directional_radiance_toa"],
        results_wrapper["directional_radiance_toa"],
        atol=absolute_tolerance_pythonicdisort,
        rtol=0.0,
    )
