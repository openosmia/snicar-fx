#!/usr/bin/env python3
"""
Defines the shared fixtures that are then used throughout the different test files

Let's create them only once per test module.

"""

from itertools import product

import pandas as pd
import pytest

from snicarfx.classes import ColumnProperties, ModelConfig, SolarIrradiance

TEST_INPUT_FILE = "./tests/inputs_tests.yaml"


@pytest.fixture(scope="module")
def model_config():
    """Provides a shared instance of ModelConfig."""
    return ModelConfig(TEST_INPUT_FILE)


@pytest.fixture(scope="module")
def column(model_config):
    """Provides a shared ColumnProperties instance using shared ModelConfig."""
    return ColumnProperties(model_config)


@pytest.fixture(scope="module")
def irradiance(model_config):
    """Provides a SolarIrradiance instance using shared ModelConfig."""
    return SolarIrradiance(model_config)


@pytest.fixture(scope="module")
def expected_shapes(column):
    return {
        "1d_layers": (column.nbr_lyr,),
        "1d_wavelengths": (column.modelconfig.inputs["RTM"]["NBR_WVL"],),
        "2d_layers_wavelengths": (
            column.nbr_lyr,
            column.modelconfig.inputs["RTM"]["NBR_WVL"],
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
def expected_mean_Fs():
    return 0.0010316714047111436


@pytest.fixture(scope="module")
def expected_mean_flx_slr():
    return 0.0020833333333333324


@pytest.fixture(scope="module")
def expected_Fd():
    return 0.0


@pytest.fixture(scope="module")
def relative_tolerance_column_properties():
    return 1e-9


def parameter_grid():
    """
    parameter grid to test snicar-fx against Matlab benchmark data
    """

    lyrList = [0, 1]
    densList = [400, 500, 600, 700, 800]
    reffList = [200, 400, 600, 800, 1000]
    zenList = [30, 40, 50, 60]
    bcList = [500, 1000, 2000]
    dzList = [
        [0.02, 0.04, 0.06, 0.08, 0.1],
        [0.04, 0.06, 0.08, 0.10, 0.15],
        [0.05, 0.10, 0.15, 0.2, 0.5],
        [0.15, 0.2, 0.25, 0.3, 0.5],
        [0.5, 0.5, 0.5, 1, 10],
    ]

    return list(product(lyrList, densList, reffList, zenList, bcList, dzList))


@pytest.fixture(scope="module")
def benchmark_matlab_data():
    return pd.read_csv("./tests/test_data/matlab_benchmark_data.csv", header=None)


@pytest.fixture(scope="module")
def absolute_tolerance_benchmark():
    return 1e-5
