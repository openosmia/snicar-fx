"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import pytest


def test_scale_atmospheric_profile(atmosphere_column):
    """
    Test atmospheric profile scaling by simply doubling the
    integrated gas concentration.
    """

    current_integrated_o3 = atmosphere_column.atmosphere_profile["o3(cm-3)"].copy()

    atmosphere_column.integrated_gas_concentrations["O3"] *= 2

    atmosphere_column.scale_atmospheric_profile()

    new_integrated_o3 = atmosphere_column.atmosphere_profile["o3(cm-3)"].copy()

    expected_integrated_o3 = current_integrated_o3 * 2

    assert np.allclose(new_integrated_o3, expected_integrated_o3, rtol=1e-12, atol=0.0)


def test_set_rayleigh_legendre_moments(atmosphere_column):
    """
    Assert that phase coefficients of order > 3 are null
    """

    assert np.all(atmosphere_column.rayleigh_legendre_moments[3:, :, :] == 0.0)


def test_scale_tau_aerosols(atmosphere_column):
    """
    Test aerosol scaling by simply doubling the Aerosol Optical
    Depth (AOD).
    """

    current_tau_aerosols = atmosphere_column.tau_aerosols.copy()

    atmosphere_column.AOD *= 2

    atmosphere_column.scale_tau_aerosols()

    new_tau_aerosols = atmosphere_column.tau_aerosols.copy()

    expected_tau_aerosols = current_tau_aerosols * 2

    assert new_tau_aerosols == pytest.approx(expected_tau_aerosols, rel=1e-12)


def test_atmospheric_properties(atmosphere_column):
    """
    Test that the single scattering albedo is within ]0,1[, that
    the optical depth is within ]0,+inf[, and that the first moment of
    the Legendre expansion of the atmosphere phase function is 1.

    """

    assert np.all((atmosphere_column.ss_alb > 0) & (atmosphere_column.ss_alb < 1))
    assert np.all(atmosphere_column.legendre_moments[0, :, :] == 1.0)
