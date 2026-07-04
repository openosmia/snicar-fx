"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import copy

import numpy as np
import pytest


def test_scale_gas_concentrations(atmosphere):
    """
    Test atmospheric profile scaling by doubling the integrated O3 concentration
    and verifying that the entire profile concentration has doubled.

    Parameters
    ----------
    atmosphere : AtmosphereColumn
        Instance of AtmosphereColumn class from snicar-fx.
    """

    atm = copy.deepcopy(atmosphere)

    current_integrated_o3 = atm.atmosphere_profile["o3(cm-3)"].copy()

    atm.integrated_gas_concentrations["O3"] *= 2

    atm.scale_gas_concentrations()

    new_integrated_o3 = atm.atmosphere_profile["o3(cm-3)"].copy()

    expected_integrated_o3 = current_integrated_o3 * 2

    assert np.allclose(new_integrated_o3, expected_integrated_o3, rtol=1e-12, atol=0.0)


def test_set_rayleigh_legendre_moments(atmosphere):
    """
    Assert that phase coefficients of order > 3 are null

    Parameters
    ----------
    atmosphere : AtmosphereColumn
        Instance of AtmosphereColumn class from snicar-fx.
    """

    assert np.all(atmosphere.rayleigh_legendre_moments[3:, :, :] == 0.0)


def test_scale_tau_aerosols(atmosphere):
    """
    Test aerosol scaling by doubling the Aerosol Optical Depth at 550nm.

    Parameters
    ----------
    atmosphere : AtmosphereColumn
        Instance of AtmosphereColumn class from snicar-fx.
    """

    current_tau_aerosols = atmosphere.tau_aerosols.copy()

    atmosphere.AOD *= 2

    atmosphere.scale_tau_aerosols()

    new_tau_aerosols = atmosphere.tau_aerosols.copy()

    expected_tau_aerosols = current_tau_aerosols * 2

    assert new_tau_aerosols == pytest.approx(expected_tau_aerosols, rel=1e-12)


def test_atmospheric_properties(atmosphere):
    """
    Test that the single scattering albedo is within ]0,1[, that
    the optical depth is within ]0,+inf[, and that the first moment of
    the Legendre expansion of the atmosphere phase function is 1.

    Parameters
    ----------
    atmosphere : AtmosphereColumn
        Instance of AtmosphereColumn class from snicar-fx.
    """

    assert np.all((atmosphere.ss_alb > 0) & (atmosphere.ss_alb < 1))
    assert np.all(atmosphere.legendre_moments[0, :, :] == 1.0)
