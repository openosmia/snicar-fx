"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "SZA_values, SAA_values, azimuth_angle_values, AOD_values, SSA_values,"[
        # SOLAR
        np.arange(42, 72, 10),
        np.arange(50, 250, 50),
        np.arange(20, 60, 10),
        np.linspace(0.1, 1, 5),
    ]
)
def test_multistreams_outputs():
    """
    This function has two goals: (1) test that outputs from the
    backend PythonicDISORT routine within SNICAR-fx strictly match
    those from the high-level PythonicDISORT wrapper and (2) test that
    outputs from the the backend PythonicDISORT routine within
    SNICAR-fx match those from the ADA solver within a loose
    tolerance.
    """
