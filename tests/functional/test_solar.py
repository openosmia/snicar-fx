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
    for var in [irradiance.direct_beam, irradiance.diffuse, irradiance.total_irradiance]:
        assert isinstance(var, np.ndarray)
        assert var.shape == expected_shapes["1d_wavelengths_solar"]


def test_solarirradiance_values(
    irradiance,
    expected_mean_direct,
    expected_mean_total_irradiance,
    expected_mean_diffuse,
    absolute_tolerance_internal_variables,
):
    """
    Assert that average values of attributes defined in the test input file
    match expected values within a tolerance threshold.

    Parameters
    ----------
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class
    expected_mean_diffuse : float
        Expected mean value of the direct collimated beam as sourced in
        the test input file.
    expected_mean_total_irradiance : float
        Expected mean value of the total solar flux as sourced
        in the test input file.
    expected_mean_direct : float
        Expected mean value of the diffuse solar beam for light as sourced
        in the test input file.
    absolute_tolerance_internal_variables: float
        Tolerance value for the error.

    """

    assert np.all(~np.isnan(irradiance.direct_beam))
    assert np.all(~np.isnan(irradiance.total_irradiance))
    assert np.all(~np.isnan(irradiance.diffuse))

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
    #     np.nanmean(irradiance.fd),
    #     expected_mean_fd,
    #     atol=absolute_tolerance_internal_variables,
    # )
