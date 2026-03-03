"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
from snicarfx.core.solvers.multi_stream_solver_disort import (
    solve_multi_stream_rt_disort,
    solve_multi_stream_rt_disort_wrapper,
)


def test_pythonicdisort_outputs(session2, multistream_pythonicdisort_params):
    """
    Test that outputs from the backend PythonicDISORT routine
    within SNICAR-fx strictly match those from the high-level
    PythonicDISORT wrapper.

    """

    sza, saa, azimuth, aod = multistream_pythonicdisort_params

    results_backend = solve_multi_stream_rt_disort(
        session2.land_column,
        session2.atmosphere_column,
        session2.solar_irradiance,
        session2.config.SOLVER,
    )

    results_wrapper = solve_multi_stream_rt_disort_wrapper(
        session2.land_column,
        session2.atmosphere_column,
        session2.solar_irradiance,
        session2.config.SOLVER,
    )

    assert np.allclose(
        results_backend["albedo_toa"], results_wrapper["albedo_toa"], atol=1e-10
    )
