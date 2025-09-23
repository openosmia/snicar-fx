"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from itertools import product

import pandas as pd
import pytest
import xarray as xr

from snicarfx.core import ColumnProperties, ModelInputs, SolarIrradiance

TEST_INPUT_FILE = "./tests/inputs_tests.yaml"
CORE_INPUT_FILE = "./src/snicarfx/inputs.yaml"


@pytest.fixture(scope="module")
def test_input_file():
    """Fetch path to the test input file."""
    return TEST_INPUT_FILE


@pytest.fixture(scope="module")
def core_input_file():
    """Fetch path to the core input file."""
    return CORE_INPUT_FILE


@pytest.fixture(scope="module")
def model_inputs():
    """Provide a shared instance of ModelInputs."""
    return ModelInputs(TEST_INPUT_FILE)


@pytest.fixture(scope="module")
def column(model_inputs):
    """Provide a shared ColumnProperties instance using shared ModelInputs."""
    return ColumnProperties(model_inputs)


@pytest.fixture(scope="module")
def irradiance(model_inputs):
    """Provide a SolarIrradiance instance using shared ModelInputs."""
    return SolarIrradiance(model_inputs)


@pytest.fixture(scope="module")
def expected_shapes(column):
    """Fetch expected shapes in the layer and wavelength dimensions."""
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
    """Fetch mean real refractive index of ice from test input file."""
    return 1.3140363224931713


@pytest.fixture(scope="module")
def expected_mean_ref_idx_im_water():
    """Fetch mean imaginary refractive index of water from test input file."""
    return 0.02835950902555164


@pytest.fixture(scope="module")
def expected_mean_fl_r_dif_a():
    """
    Fetch mean diffuse fresnel coefficient for light coming from above.
    """
    return 0.07721768897304701


@pytest.fixture(scope="module")
def expected_tau():
    """Fetch mean optical thickness from test input file."""
    return 325.0


@pytest.fixture(scope="module")
def expected_mean_fs():
    """Fetch mean direct collimated solar beam from test input file."""
    return 0.0010316714047111436


@pytest.fixture(scope="module")
def expected_mean_flx_slr():
    """Fetch mean solar flux from test input file."""
    return 0.0020833333333333324


@pytest.fixture(scope="module")
def expected_fd():
    """Fetch mean diffuse solar beam from test input file."""
    return 0.0


def twostream_parameter_grid():
    """Define parameter grid to test snicar-fx against SNICAR-ADv4 outputs."""
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


@pytest.fixture(scope="module")
def benchmark_ada_spectral_data():
    """Read CRTM ADA outputs to test snicar-fx against."""
    return xr.open_dataset("./tests/test_data/benchmark_ADA_spectral_albedo.nc")


def multistream_ada_parameter_grid(ds):
    """Read parameter grid to test snicar-fx against CRTM ADA outputs."""
    return list(
        product(
            ds.w.values,
            ds.t_od.values,
            ds.g.values,
            ds.wvl_idx.values,
        )
    )


def pytest_generate_tests(metafunc):
    """Store parameter grid for the tests against CRTM into `params_ada`."""
    if {"params_ada"} <= set(metafunc.fixturenames):
        ds = xr.open_dataset("./tests/test_data/benchmark_ADA_spectral_albedo.nc")
        grid = list(multistream_ada_parameter_grid(ds))
        metafunc.parametrize("params_ada", grid)


@pytest.fixture(scope="module")
def benchmark_snicaradv4_spectral_data():
    """Read SNICAR_ADv4 spectral albedo outputs to test snicar-fx against."""
    return pd.read_csv(
        "./tests/test_data/benchmark_SNICARADv4_spectral_albedo.csv", header=None
    )


@pytest.fixture(scope="module")
def benchmark_snicaradv4_bba_data():
    """Read SNICAR_ADv4 BBA outputs to test snicar-fx against."""
    return pd.read_csv("./tests/test_data/benchmark_SNICARADv4_BBA.csv", header=None)


@pytest.fixture(scope="module")
def benchmark_snicaradv4_absorbed_flux_data():
    """Read SNICAR_ADv4 absorbed flux outputs to test snicar-fx against."""
    return pd.read_csv(
        "./tests/test_data/benchmark_SNICARADv4_absorbed_flux.csv", header=None
    )


@pytest.fixture(scope="module")
def absolute_tolerance_internal_variables():
    """Set absolute tolerance on error for the internal variables."""
    return 1e-9


@pytest.fixture(scope="module")
def absolute_tolerance_benchmark():
    """
    Set absolute tolerance on error between snicar-fx outputs and CRTM/SNICAR-ADv4.
    """
    return 1e-5
