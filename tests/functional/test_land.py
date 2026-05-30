"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np


def test_landcolumn_shapes(land_column, expected_shapes):
    """
    Verify shapes of attributes of a LandColumn instance match expected
    values in the layer and wavelength dimensions.

    Parameters
    ----------
    land_column : LandColumn
        Instance of the LandColumn class
    expected_shapes : dict
        Expected shapes of the attributes of `land_column`
    """
    for var in [
        land_column.asm_prm,
        land_column.ext_cff,
        land_column.tau,
        land_column.ss_alb,
    ]:
        assert isinstance(var, np.ndarray)
        assert var.shape == expected_shapes["2d_layers_wavelengths"]

    assert isinstance(land_column.layer_mass, np.ndarray)
    assert land_column.layer_mass.shape == expected_shapes["1d_layers"]


def test_landcolumn_values(
    land_column,
    expected_mean_ref_idx_re,
    expected_mean_ref_idx_im_water,
    expected_mean_fl_r_dif_a,
    expected_tau,
    absolute_tolerance_internal_variables,
):
    """
    Assert that average values of attributes defined in the test input file
    match expected values within a tolerance threshold.

    Parameters
    ----------
    land_column : LandColumn
        Instance of the LandColumn class
    expected_mean_ref_idx_re : float
        Expected mean value of the real refractive index of ice as sourced in
        the test input file.
    expected_mean_ref_idx_im_water : float
        Expected mean value of the imaginary refractive index of water as sourced
        in the test input file.
    expected_mean_fl_r_dif_a : float
        Expected mean value of the diffuse fresnel coefficient for light from
        above.
    expected_tau : float
        Expected mean optical thickness for the parameters defined in the test
        input file.
    absolute_tolerance_internal_variables: float
        Tolerance value for the error.

    """

    assert np.all(~np.isnan(land_column.ref_idx_re))
    assert np.all(~np.isnan(land_column.ref_idx_im_water))
    assert np.all(~np.isnan(land_column.fl_r_dif_a))

    assert np.isclose(
        np.nanmean(land_column.ref_idx_re),
        expected_mean_ref_idx_re,
        atol=absolute_tolerance_internal_variables,
        rtol=0.0,
    )

    assert np.isclose(
        np.nanmean(land_column.ref_idx_im_water),
        expected_mean_ref_idx_im_water,
        atol=absolute_tolerance_internal_variables,
        rtol=0.0,
    )

    assert np.isclose(
        np.nanmean(land_column.fl_r_dif_a),
        expected_mean_fl_r_dif_a,
        atol=absolute_tolerance_internal_variables,
        rtol=0.0,
    )

    assert np.allclose(
        np.nanmean(land_column.tau),
        expected_tau,
        atol=absolute_tolerance_internal_variables,
        rtol=0.0,
    )
