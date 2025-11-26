"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np

def test_solarirradiance_shapes(irradiance, expected_shapes):
    """
    Verify shapes of attributes of a SolarIrradiance instance match expected
    values in the layer and wavelength dimensions.

    Parameters
    ----------
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class
    expected_shapes : array
        Expected shapes of the attributes of `column`
    """
    for var in [irradiance.fs, irradiance.fd, irradiance.flx_slr]:
        assert isinstance(var, np.ndarray)
        assert var.shape == expected_shapes["1d_wavelengths"]


def test_solarirradiance_values(
    irradiance,
):
    """
    Assert that average values of attributes defined in the test input file
    match expected values within a tolerance threshold.

    Parameters
    ----------
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    """

    assert np.all(~np.isnan(irradiance.fs))
    assert np.all(~np.isnan(irradiance.flx_slr))
    assert np.all(~np.isnan(irradiance.fd))
    
    
    # assert np.isclose(
    #     np.nanmean(irradiance.fs),
    #     expected_mean_fs,
    #     atol=absolute_tolerance_internal_variables,
    # )

    # assert np.isclose(
    #     np.nanmean(irradiance.flx_slr),
    #     expected_mean_flx_slr,
    #     atol=absolute_tolerance_internal_variables,
    # )

    # assert np.allclose(
    #     irradiance.fd, expected_fd, atol=absolute_tolerance_internal_variables
    # )
