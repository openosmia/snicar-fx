#!/usr/bin/env python3
"""
Defines the shared fixtures that are then used throughout the different test files

Let's create them only once per test module.

"""

from itertools import product

import pandas as pd
import pytest
import xarray as xr

from snicarfx.core import ColumnProperties, ModelInputs, SolarIrradiance

TEST_INPUT_FILE = "./tests/inputs_tests.yaml"


@pytest.fixture(scope="module")
def model_inputs():
    """Provides a shared instance of ModelInputs."""
    return ModelInputs(TEST_INPUT_FILE)


@pytest.fixture(scope="module")
def column(model_inputs):
    """Provides a shared ColumnProperties instance using shared ModelInputs."""
    return ColumnProperties(model_inputs)


@pytest.fixture(scope="module")
def irradiance(model_inputs):
    """Provides a SolarIrradiance instance using shared ModelInputs."""
    return SolarIrradiance(model_inputs)


@pytest.fixture(scope="module")
def expected_shapes(column):
    return {
        "1d_layers": (column.nbr_lyr,),
        "1d_wavelengths": (column.nbr_wvl,),
        "2d_layers_wavelengths": (
            column.nbr_lyr,
            column.nbr_wvl,
        ),
    }


@pytest.fixture(scope="module")
def expected_mean_ref_idx_re():
    return 1.3140363224931713


@pytest.fixture(scope="module")
def expected_mean_ref_idx_im_water():
    return 0.02835950902555164


@pytest.fixture(scope="module")
def expected_mean_fl_r_dif_a():
    return 0.07721768897304701


@pytest.fixture(scope="module")
def expected_tau():
    return 325.0


@pytest.fixture(scope="module")
def expected_mean_fs():
    return 0.0010316714047111436


@pytest.fixture(scope="module")
def expected_mean_flx_slr():
    return 0.0020833333333333324


@pytest.fixture(scope="module")
def expected_fd():
    return 0.0


@pytest.fixture(scope="module")
def relative_tolerance_column_properties():
    return 1e-9


def twostream_parameter_grid():
    """
    parameter grid to test snicar-fx against Matlab benchmark data
    """

    layer_type_grid = [0, 1]
    density_grid = [300, 600, 900]
    radii_grid = [200, 600, 1000]
    sza_grid = [30, 50, 70]
    bc_grid = [0, 100, 1000]
    thickness_profiles_grid = [[0.01, 0.01, 0.01], [0.01, 0.1, 1], [0.01, 10, 100]]
    direct_diffuse_grid = [1, 0]

    return list(
        product(
            layer_type_grid,
            density_grid,
            radii_grid,
            sza_grid,
            bc_grid,
            thickness_profiles_grid,
            direct_diffuse_grid,
        )
    )


def multistream_ADA_parameter_grid(ds):
    """
    parameter grid to test snicar-fx against ADA Fortran data.
    Just read the grid from the nc file of the Fortran results.
    """
    return list(
        product(
            ds.w.values,
            ds.t_od.values,
            ds.g.values,
            ds.wvl_idx.values,
        )
    )


@pytest.fixture(scope="module")
def benchmark_ADA_spectral_data():
    return xr.open_dataset("./tests/test_data/benchmark_ADA_spectral_albedo.nc")


def pytest_generate_tests(metafunc):
    """
    pytest hook to parametrize tests that use idx_ada and params_ada
    """
    if {"idx_ada", "params_ada"} <= set(metafunc.fixturenames):
        ds = xr.open_dataset("./tests/test_data/benchmark_ADA_spectral_albedo.nc")
        grid = list(enumerate(multistream_ADA_parameter_grid(ds)))
        metafunc.parametrize("idx_ada,params_ada", grid)


@pytest.fixture(scope="module")
def benchmark_snicaradv4_spectral_data():
    return pd.read_csv(
        "./tests/test_data/benchmark_SNICARADv4_spectral_albedo.csv", header=None
    )


@pytest.fixture(scope="module")
def benchmark_snicaradv4_bba_data():
    return pd.read_csv("./tests/test_data/benchmark_SNICARADv4_BBA.csv", header=None)


@pytest.fixture(scope="module")
def benchmark_snicaradv4_absorbed_flux_data():
    return pd.read_csv(
        "./tests/test_data/benchmark_SNICARADv4_absorbed_flux.csv", header=None
    )


@pytest.fixture(scope="module")
def absolute_tolerance_benchmark():
    return 1e-5
