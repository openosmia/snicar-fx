>#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the shared fixtures that are then used throughout the different test files

Let's create them only once per test module.

"""


import pytest
from snicarfx.classes import ColumnProperties, SolarIrradiance, ModelConfig

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

def expected_tau():
    return 325.

def expected_mean_Fs():
    return 0.0010316714047111436

def expected_mean_flx_slr():
    return 0.0020833333333333324

def expected_Fd():
    return 0.0

@pytest.fixture(scope="module")
def relative_tolerance_column_properties():
    return 1e-9
