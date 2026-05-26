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
    expected_shapes : dict
        Expected shapes of the attributes of `irradiance`
    """
    for var in [irradiance.direct_beam, irradiance.diffuse, irradiance.total_irradiance]:
        assert isinstance(var, np.ndarray)
        assert var.shape == expected_shapes["1d_wavelengths_solar"]


def test_solarirradiance_values(
    irradiance
):
    """
    Assert that average values of attributes defined in the test input file
    match expected values within a tolerance threshold.

    Parameters
    ----------
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class

    """

    assert np.all(~np.isnan(irradiance.direct_beam))
    assert np.all(~np.isnan(irradiance.total_irradiance))
    assert np.all(~np.isnan(irradiance.diffuse))
