#!/usr/bin/env python3
"""

Test the SolarIrradiance class.

This basically just checks variable shapes and expected outputs, and
would catch changes in snicar-fx data files and code using them.

"""

import numpy as np


def test_columnproperties_shapes(irradiance, expected_shapes):

    assert len(irradiance.stubs) == 7


def test_solarirradiance_values(
    irradiance,
    expected_mean_Fs,
    expected_mean_flx_slr,
    expected_Fd,
    relative_tolerance_column_properties,
):
    """ """

    assert np.all(~np.isnan(irradiance.Fs))
    assert np.all(~np.isnan(irradiance.flx_slr))
    assert np.all(~np.isnan(irradiance.Fd))

    assert np.isclose(
        np.nanmean(irradiance.Fs),
        expected_mean_Fs,
        rtol=relative_tolerance_column_properties,
    )

    assert np.isclose(
        np.nanmean(irradiance.flx_slr),
        expected_mean_flx_slr,
        rtol=relative_tolerance_column_properties,
    )

    assert np.allclose(
        irradiance.Fd, expected_Fd, rtol=relative_tolerance_column_properties
    )
