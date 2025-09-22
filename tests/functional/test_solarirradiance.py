import numpy as np


def test_solarirradiance_values(
    irradiance,
    expected_mean_fs,
    expected_mean_flx_slr,
    expected_fd,
    absolute_tolerance_internal_variables,
):
    """
    Assert that average values of attributes defined in the test input file
    match expected values within a tolerance threshold.
    
    Parameters
    ----------
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class
    expected_mean_fs : float
        Expected mean value of the direct collimated beam as sourced in 
        the test input file.
    expected_mean_flx_slr : float
        Expected mean value of the total solar flux as sourced 
        in the test input file.
    expected_fd : float
        Expected mean value of the diffuse solar beam for light as sourced 
        in the test input file.
    absolute_tolerance_internal_variables: float
        Tolerance value for the error.
    
    """

    assert np.all(~np.isnan(irradiance.fs))
    assert np.all(~np.isnan(irradiance.flx_slr))
    assert np.all(~np.isnan(irradiance.fd))

    assert np.isclose(
        np.nanmean(irradiance.fs),
        expected_mean_fs,
        atol=absolute_tolerance_internal_variables,
    )

    assert np.isclose(
        np.nanmean(irradiance.flx_slr),
        expected_mean_flx_slr,
        atol=absolute_tolerance_internal_variables,
    )

    assert np.allclose(
        irradiance.fd, expected_fd, atol=absolute_tolerance_internal_variables
    )
