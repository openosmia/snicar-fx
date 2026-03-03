"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from itertools import product

import pytest
import xarray as xr
import numpy as np

from snicarfx.core.components.atmosphere import AtmosphereColumn
from snicarfx.core.components.land import LandColumn
from snicarfx.core.components.solar import SolarIrradiance
from snicarfx.core.session.session import Session

TEST_INPUT_FILE1 = "./tests/inputs_tests.yaml"
TEST_INPUT_FILE2 = "./tests/inputs_tests2.yaml"
CORE_INPUT_FILE = "./src/snicarfx/inputs.yaml"


@pytest.fixture(scope="module")
def core_input_file():
    """Fetch path to the core input file."""
    return CORE_INPUT_FILE


@pytest.fixture(scope="module")
def test_input_file():
    """Fetch path to the test input file."""
    return TEST_INPUT_FILE1


@pytest.fixture(scope="module")
def test_input_file2():
    """Fetch path to the test input file."""
    return TEST_INPUT_FILE2


@pytest.fixture(scope="module")
def config(session):
    """Provide a shared instance of Config."""
    return session.config


@pytest.fixture(scope="module")
def session():
    """Provide a shared Session instance."""
    return Session(TEST_INPUT_FILE1)


@pytest.fixture(scope="module")
def config2(session2):
    """Provide a shared instance of Config."""
    return session2.config


@pytest.fixture(scope="module")
def session2():
    """Provide a shared Session instance."""
    return Session(TEST_INPUT_FILE2)


@pytest.fixture(scope="module")
def land_column(config):
    """Provide a shared LandColumn instance using shared Config."""
    return LandColumn(config)


@pytest.fixture(scope="module")
def atmosphere_column(config):
    """Provide a shared AtmosphereColumn instance using shared Config."""
    return AtmosphereColumn(config)


@pytest.fixture(scope="module")
def irradiance(config):
    """Provide a SolarIrradiance instance using shared Config."""
    return SolarIrradiance(config)


@pytest.fixture(scope="module")
def land_column2(config2):
    """Provide a shared LandColumn instance using shared Config."""
    return LandColumn(config2)


@pytest.fixture(scope="module")
def atmosphere_column2(config2):
    """Provide a shared AtmosphereColumn instance using shared Config."""
    return AtmosphereColumn(config2)


@pytest.fixture(scope="module")
def irradiance2(config2):
    """Provide a SolarIrradiance instance using shared Config."""
    return SolarIrradiance(config2)


@pytest.fixture(scope="module")
def expected_shapes(session):
    """Fetch expected shapes in the layer and wavelength dimensions."""
    return {
        "1d_layers": (session.land_column.nbr_lyr,),
        "1d_wavelengths_solar": (len(session.config._wavelengths_solar),),
        "2d_layers_wavelengths": (
            session.land_column.nbr_lyr,
            session.land_column.nbr_wvl,
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
            ds.wavelength_index.values,
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


@pytest.fixture(
    params=product(
        np.linspace(42, 72, 3),  # SZA
        np.linspace(50, 250, 3),  # SAA
        np.linspace(20, 60, 3),  # Azimuth
        np.linspace(0.1, 1, 3),  # AOD
    ),
    ids=lambda p: f"SZA{p[0]}_SAA{p[1]}_Az{p[2]}_AOD{p[3]:.2f}",
)
def multistream_pythonicdisort_params(request):
    """
    Create sets of parameters to be used in tests of PythonicDISORT.

    """
    return request.param


@pytest.fixture(
    params=[
        # SOLAR
        {"component": "SOLAR", "field": "SZA", "value": 42},
        {"component": "SOLAR", "field": "SAA", "value": 180},
        # SOLVER
        {"component": "SOLVER", "field": "OUTPUT_LEVELS", "value": "TOA"},
        {"component": "SOLVER", "field": "N_FOURIER_MODES", "value": 3},
        {"component": "SOLVER", "field": "AZIMUTH_ANGLES", "value": (10, 170, 5)},
        {"component": "SOLVER", "field": "POLAR_ANGLES", "value": (10, 90, 5)},
        # ATMOSPHERE
        {"component": "ATMOSPHERE", "field": "INTEGRATED_AOD_550", "value": 0.42},
        {
            "component": "ATMOSPHERE",
            "field": "INTEGRATED_GAS_CONCENTRATIONS",
            "value": {"H2O": 15, "NO2": 1e-02, "O3": 0.01},
        },
        # LAND
        {"component": "LAND", "field": "LAYER_TYPE", "value": (0, 0, 0)},
        {"component": "LAND", "field": "GRAIN_SHAPE", "value": (0, 0, 0)},
        {"component": "LAND", "field": "RF_TYPE", "value": "Pic16"},
        {"component": "LAND", "field": "LWC", "value": (0.01, 0.01, 0.01)},
        {"component": "LAND", "field": "THICKNESS", "value": (0.07, 0.04, 0.1)},
        {"component": "LAND", "field": "SPECIFIC_SURFACE_AREA", "value": (1, 2, 3)},
        {"component": "LAND", "field": "DENSITY", "value": (600, 700, 800)},
        {
            "component": "LAND",
            "field": "LIGHT_ABSORBING_PARTICLES",
            "value": {
                "BC1": {"FILE": "bc_ChCB_rn40_dns1270.nc", "CONC": (2, 20, 200)},
                "BC2": {"FILE": "bc_ChCB_rn40_dns1270.nc", "CONC": (3, 30, 300)},
            },
        },
    ],
    ids=lambda p: f"{p['component']}_{p['field']}_{p['value']}",
)
def update_api_params(request):
    """
    Create sets of parameters to be used in tests of the update API.
    """
    return request.param


@pytest.fixture(scope="module")
def absolute_tolerance_internal_variables():
    """Set absolute tolerance on error for the internal variables."""
    return 1e-9


@pytest.fixture(scope="module")
def absolute_tolerance_benchmark():
    """
    Set absolute tolerance on error between snicar-fx outputs and
    CRTM/SNICAR-ADv4.

    """
    return 1e-5


@pytest.fixture(scope="module")
def absolute_tolerance_pythonicdisort():
    """
    Set absolute tolerance on error between backend routine and
    the high-level wrapper of PythonicDISORT

    """
    return 5e-13


@pytest.fixture(scope="module")
def absolute_tolerance_update_api_gs():
    """
    Set absolute tolerance on error for the update API, on
    atmosphere property scaling, specifically.

    """
    return 5e-13


@pytest.fixture(scope="module")
def absolute_tolerance_update_api():
    """
    Set absolute tolerance on error for the update API

    """
    return 8e-16
