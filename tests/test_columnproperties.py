#!/usr/bin/env python3
"""

Test the ColumnProperties class.

This basically just checks variable shapes and expected outputs, and
would catch changes in snicar-fx data files and code using them.

"""

import numpy as np


def test_columnproperties_shapes(column, expected_shapes):

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
    relative_tolerance_column_properties,
):
    """ """

    assert np.all(~np.isnan(column.ref_idx_re))
    assert np.all(~np.isnan(column.ref_idx_im_water))
    assert np.all(~np.isnan(column.fl_r_dif_a))

    assert np.isclose(
        np.nanmean(column.ref_idx_re),
        expected_mean_ref_idx_re,
        rtol=relative_tolerance_column_properties,
    )

    assert np.isclose(
        np.nanmean(column.ref_idx_im_water),
        expected_mean_ref_idx_im_water,
        rtol=relative_tolerance_column_properties,
    )

    assert np.isclose(
        np.nanmean(column.fl_r_dif_a),
        expected_mean_fl_r_dif_a,
        rtol=relative_tolerance_column_properties,
    )

    assert np.allclose(
        column.tau, expected_tau, rtol=relative_tolerance_column_properties
    )
