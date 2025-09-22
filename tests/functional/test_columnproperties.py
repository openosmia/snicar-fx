"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

import numpy as np


def test_columnproperties_shapes(column, expected_shapes):
    """
    Verify shapes of attributes of a ColumnProperties instance match expected
    values in the layer and wavelength dimensions.
    
    Parameters
    ----------
    column : ColumnProperties
        Instance of the ColumnProperties class
    expected_shapes : array
        Expected shapes of the attributes of `column`
    """
    for var in [column.asm_prm, column.ext_cff, column.tau, column.ss_alb]:
        assert isinstance(var, np.ndarray)
        assert var.shape == expected_shapes["2d_layers_wavelengths"]

    assert isinstance(column.layer_mass, np.ndarray)
    assert column.layer_mass.shape == expected_shapes["1d_layers"]


def test_columnproperties_values(
    column,
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
    column : ColumnProperties
        Instance of the ColumnProperties class
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

    assert np.all(~np.isnan(column.ref_idx_re))
    assert np.all(~np.isnan(column.ref_idx_im_water))
    assert np.all(~np.isnan(column.fl_r_dif_a))

    assert np.isclose(
        np.nanmean(column.ref_idx_re),
        expected_mean_ref_idx_re,
        atol=absolute_tolerance_internal_variables,
    )

    assert np.isclose(
        np.nanmean(column.ref_idx_im_water),
        expected_mean_ref_idx_im_water,
        atol=absolute_tolerance_internal_variables,
    )

    assert np.isclose(
        np.nanmean(column.fl_r_dif_a),
        expected_mean_fl_r_dif_a,
        atol=absolute_tolerance_internal_variables,
    )

    assert np.allclose(
        column.tau, expected_tau, atol=absolute_tolerance_internal_variables
    )
