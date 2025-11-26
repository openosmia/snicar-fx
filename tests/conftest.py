"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from itertools import product

import pytest
import xarray as xr

from snicarfx.core.components.land import LandColumn
from snicarfx.core.components.solar import SolarIrradiance
from snicarfx.core.components.atmosphere import AtmosphereColumn
from snicarfx.core.session.config import Config
from snicarfx.core.session.session import Session

TEST_INPUT_FILE = "./tests/inputs_tests.yaml"
CORE_INPUT_FILE = "./src/snicarfx/inputs.yaml"


@pytest.fixture(scope="module")
def core_input_file():
    """Fetch path to the core input file."""
    return CORE_INPUT_FILE


@pytest.fixture(scope="module")
def test_input_file():
    """Fetch path to the test input file."""
    return TEST_INPUT_FILE


@pytest.fixture(scope="module")
def config():
    """Provide a shared instance of Config."""
    return Config.from_yaml(TEST_INPUT_FILE)


@pytest.fixture(scope="module")
def session():
    """Provide a shared Session instance."""
    return Session(TEST_INPUT_FILE)


@pytest.fixture(scope="module")
def root_dir():
    """Provide a shared package root directory."""
    return Session.get_package_root()


@pytest.fixture(scope="module")
def land_column(config, root_dir):
    """Provide a shared LandColumn instance using shared Config."""
    return LandColumn(config, root_dir)


@pytest.fixture(scope="module")
def atmosphere_column(config, root_dir):
    """Provide a shared AtmosphereColumn instance using shared Config."""
    return AtmosphereColumn(config, root_dir)


@pytest.fixture(scope="module")
def irradiance(config, root_dir):
    """Provide a SolarIrradiance instance using shared Config."""
    return SolarIrradiance(config, root_dir)


@pytest.fixture(scope="module")
def expected_shapes(land_column):
    """Fetch expected shapes in the layer and wavelength dimensions."""
    return {
        "1d_layers": (land_column.nbr_lyr,),
        "1d_wavelengths": (land_column.nbr_wvl,),
        "2d_layers_wavelengths": (
            land_column.nbr_lyr,
            land_column.nbr_wvl,
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
    return 0.0018987214821366344


@pytest.fixture(scope="module")
def expected_mean_fd():
    """Fetch mean diffuse solar beam from test input file."""
    return 0.0001846118511966984


@pytest.fixture(scope="module")
def expected_mean_flx_slr():
    """Fetch mean solar flux from test input file."""
    return 0.0020833333333333324


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


@pytest.fixture(scope="module")
def benchmark_snicaradv4_spectral_data():
    """Read SNICAR_ADv4 spectral albedo outputs to test snicar-fx against."""
    return xr.open_dataset("./tests/test_data/benchmark_SNICARADv4_spectral_albedo.nc")


@pytest.fixture(scope="module")
def benchmark_snicaradv4_bba_data():
    """Read SNICAR_ADv4 BBA outputs to test snicar-fx against."""
    return xr.open_dataset("./tests/test_data/benchmark_SNICARADv4_BBA.nc")


@pytest.fixture(scope="module")
def benchmark_snicaradv4_absorbed_flux_data():
    """Read SNICAR_ADv4 absorbed flux outputs to test snicar-fx against."""
    return xr.open_dataset("./tests/test_data/benchmark_SNICARADv4_absorbed_flux.nc")


def twostream_parameter_grid(ds):
    """Define parameter grid to test snicar-fx against SNICAR-ADv4 outputs."""
    return list(
        product(
            ds.layer_type.values - 1,
            ds.density.values,
            ds.reff.values,
            ds.sza.values,
            ds.bc.values,
            ds.dz_layers.values,
            ds.direct.values,
        )
    )


def pytest_generate_tests(metafunc):
    """Store parameter grid for the tests."""
    if {"params_ada"} <= set(metafunc.fixturenames):
        ds = xr.open_dataset("./tests/test_data/benchmark_ADA_spectral_albedo.nc")
        grid = list(multistream_ada_parameter_grid(ds))
        metafunc.parametrize("params_ada", grid)
    if {"params_2str"} <= set(metafunc.fixturenames):
        ds = xr.open_dataset("./tests/test_data/benchmark_SNICARADv4_BBA.nc")
        grid = list(twostream_parameter_grid(ds))
        metafunc.parametrize("params_2str", grid)


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
