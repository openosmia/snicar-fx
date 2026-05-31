"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from itertools import product

import numpy as np
import pytest
import xarray as xr
import glob
from pathlib import Path

from snicarfx import Session
from snicarfx.cli.download import ZENODO_RECORD
from snicarfx.core import AtmosphereColumn, LandColumn, SolarIrradiance

# path to different test input files
TEST_INPUT_FILE_TWOSTREAM = "./tests/input_files/inputs_tests_twostream.yaml"
TEST_INPUT_FILE_MULTISTREAM_UNCOUPLED = (
    "./tests/input_files/inputs_tests_ada_multistream_uncoupled.yaml"
)
TEST_INPUT_FILE_MULTISTREAM_COUPLED = (
    "./tests/input_files/inputs_tests_disort_multistream_coupled.yaml"
)

# API URL to test download of zenodo data archive
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# list all available example scripts to be tested
PACKAGE_ROOT = Session.get_package_root()
EXAMPLES_DIR = PACKAGE_ROOT / "examples"
EXAMPLE_SCRIPTS = sorted(glob.glob(f"{EXAMPLES_DIR}/*/*.py"))


@pytest.fixture(scope="module")
def test_input_file_twostream():
    """Fetch path to the test input file for two-stream configuration."""
    return TEST_INPUT_FILE_TWOSTREAM


@pytest.fixture(scope="module")
def test_input_file_multistream_uncoupled():
    """Fetch path to the test input file for multi-stream uncoupled configuration."""
    return TEST_INPUT_FILE_MULTISTREAM_UNCOUPLED


@pytest.fixture(scope="module")
def test_input_file_multistream_coupled():
    """Fetch path to the test input file for multi-stream coupled configuration."""
    return TEST_INPUT_FILE_MULTISTREAM_COUPLED


@pytest.fixture(scope="module")
def api_url():
    """Fetch path to API URL."""
    return API_URL


@pytest.fixture(scope="module")
def session_twostream():
    """Provide a shared Session instance for two-stream configuration."""
    return Session(TEST_INPUT_FILE_TWOSTREAM)


@pytest.fixture(scope="module")
def config(session_twostream):
    """Provide a shared instance of Config for two-stream configuration."""
    return session_twostream.config


@pytest.fixture(scope="module")
def solar(config):
    """Provide a SolarIrradiance instance using shared two-stream Config."""
    return SolarIrradiance(config)


@pytest.fixture(scope="module")
def land(config):
    """Provide a shared LandColumn instance using shared two-stream Config."""
    return LandColumn(config)


@pytest.fixture(scope="module")
def session_multistream_uncoupled():
    """Provide a shared Session instance for multi-stream uncoupled configuration."""
    return Session(TEST_INPUT_FILE_MULTISTREAM_UNCOUPLED)


@pytest.fixture(scope="module")
def session_multistream_coupled():
    """Provide a shared Session instance for multi-stream coupled configuration."""
    return Session(TEST_INPUT_FILE_MULTISTREAM_COUPLED)


@pytest.fixture(scope="module")
def atmosphere(session_multistream_coupled):
    """Provide a shared AtmosphereColumn instance."""
    return AtmosphereColumn(session_multistream_coupled.config)


@pytest.fixture(scope="module")
def expected_shapes(session_twostream):
    """Fetch expected shapes in the layer and wavelength dimensions."""
    return {
        "1d_layers": (session_twostream.land.nbr_lyr,),
        "1d_wavelengths_solar": (len(session_twostream.config._wavelengths_solar),),
        "2d_layers_wavelengths": (
            session_twostream.land.nbr_lyr,
            session_twostream.land.nbr_wvl,
        ),
    }


@pytest.fixture(scope="module")
def expected_mean_ref_idx_re():
    """Fetch mean real refractive index of ice."""
    return 1.3140363224931713


@pytest.fixture(scope="module")
def expected_mean_ref_idx_im_water():
    """Fetch mean imaginary refractive index of water."""
    return 0.02835950902555164


@pytest.fixture(scope="module")
def expected_mean_fl_r_dif_a():
    """Fetch mean diffuse fresnel coefficient for light coming from above."""
    return 0.07721768897304701


@pytest.fixture(scope="module")
def expected_tau():
    """Fetch mean optical thickness from two-stream input file config."""
    return 325.0


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
    if {"params_twostream"} <= set(metafunc.fixturenames):
        ds = xr.open_dataset("./tests/test_data/benchmark_SNICARADv4_BBA.nc")
        grid = list(twostream_parameter_grid(ds))
        metafunc.parametrize("params_twostream", grid)


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
        {"component": "SOLVER", "field": "POLAR_ANGLES", "value": (10, 89, 5)},
        # ATMOSPHERE
        {"component": "ATMOSPHERE", "field": "INTEGRATED_AOD_550", "value": 0.42},
        {"component": "ATMOSPHERE", "field": "INTEGRATED_AOD_550", "value": 0},
        {
            "component": "ATMOSPHERE",
            "field": "INTEGRATED_GAS_CONCENTRATIONS",
            "value": {"H2O": 15, "NO2": 1e-06, "O3": 0.007},
        },
        {
            "component": "ATMOSPHERE",
            "field": "INTEGRATED_GAS_CONCENTRATIONS",
            "value": {"H2O": 0, "NO2": 0, "O3": 0},
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


@pytest.fixture(
    params=[
        # Only ATMOSPHERE fields
        {"field": "INTEGRATED_AOD_550", "sequence": [0.42, 0.0, 0.11, 0.22]},
        {
            "field": "INTEGRATED_GAS_CONCENTRATIONS",
            "sequence": [
                {"H2O": 15, "NO2": 1e-06, "O3": 0.005, "O2": 100},
                {"H2O": 0, "NO2": 0, "O3": 0, "O2": 0},
                {"H2O": 7, "NO2": 1.2e-6, "O3": 0.002, "O2": 200},
                {"H2O": 24, "NO2": 0.5e-6, "O3": 0.001, "O2": 1000},
            ],
        },
    ],
    ids=lambda p: f"{p['field']}_{p['sequence']}",
)
def update_api_scaling_params(request):
    """
    Create sets of parameters to be used in tests of sequential
    scaling with the update API (occurring in ATMOSPHERE only).

    """
    return request.param


@pytest.fixture(
    params=EXAMPLE_SCRIPTS,
    ids=lambda p: Path(p).parent.name,
)
def example_script_path(request):
    """
    Prepare the list of example scripts to be run by test_examples.
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
def absolute_tolerance_update_api():
    """
    Set absolute tolerance on error for the update API

    """
    return 8e-16
