"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import copy

import pytest
from pydantic import BaseModel

from snicarfx import Config


def test_config_attributes(config):
    """
    Verify that the required attributes of Config exist and have the correct
    types.

    Parameters
    ----------
    config : Config
        Instance of the Config class
    """

    assert isinstance(config, BaseModel)

    assert hasattr(config, "SOLVER")

    assert hasattr(config, "SOLAR")

    assert hasattr(config, "ATMOSPHERE")

    assert hasattr(config, "LAND")

    assert hasattr(config, "from_yaml")
    assert callable(config.from_yaml)

    assert hasattr(config, "check_lengths")
    assert callable(config.check_lengths)

    assert config.model_config["extra"] == "forbid"


# pytest will run for each dictionary
@pytest.mark.parametrize(
    "wrong_configs",
    [
        {"SOLVER": {"TYPE": "two-stream-ad", "ATMOSPHERE_COUPLING": True}},
        {"SOLVER": {"TYPE": "two-stream-ad", "OUTPUT_LEVELS": "TOA"}},
        {"SOLVER": {"TYPE": "two-stream-ad", "DELTA_SCALING": "M+"}},
        {"SOLVER": {"TYPE": "two-stream-ad", "N_LEGENDRE_MOMENTS": 15}},
        {"SOLVER": {"TYPE": "multi-stream-ada", "N_STREAMS": 15}},
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "N_STREAMS": 16,
                "N_LEGENDRE_MOMENTS": 20,
            }
        },
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "N_STREAMS": 16,
                "N_FOURIER_MODES": 20,
            }
        },
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "ATMOSPHERE_COUPLING": False,
                "OUTPUT_LEVELS": "TOA",
            }
        },
        {"SOLVER": {"TYPE": "two-stream-ad", "N_FOURIER_MODES": 2}},
        {"SOLVER": {"TYPE": "two-stream-ad", "POLAR_ANGLES": [10, 40, 10]}},
        {"SOLVER": {"TYPE": "two-stream-ad", "AZIMUTH_ANGLES": [10, 40, 10]}},
        {"SOLVER": {"TYPE": "multi-stream-ada", "POLAR_ANGLES": [40, 10, 10]}},
        {"SOLVER": {"TYPE": "multi-stream-ada", "POLAR_ANGLES": [6, 7, 50]}},
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "AZIMUTH_ANGLES": [40, 10, 10],
                "N_FOURIER_MODES": 2,
            }
        },
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "AZIMUTH_ANGLES": [6, 7, 100],
                "N_FOURIER_MODES": 2,
            }
        },
        {
            "SOLVER": {
                "TYPE": "multi-stream-ada",
                "AZIMUTH_ANGLES": [10, 100, 20],
                "N_FOURIER_MODES": 1,
            }
        },
        {"SPECTRAL": {"RESOLUTION": [400, 401, 50]}},
        {"SPECTRAL": {"RESOLUTION": [500, 400, 10]}},
        {"SPECTRAL": {"RESOLUTION": [400, 500, 10], "MODE": "band-srf-integration"}},
        {
            "SPECTRAL": {
                "RESOLUTION": [400, 500, "1cm-1"],
                "MODE": "band-snicar-default",
            }
        },
        {"ATMOSPHERE": {"SKY_CONDITIONS": "cloudy"}},
        {"ATMOSPHERE": {"INTEGRATED_AOD_550": 0.1, "AEROSOL_PROPERTIES": None}},
        {"LAND": {"LAYER_TYPE": [1, 1, 1], "GRAIN_SHAPE": [1, 1, 1]}},
        {"LAND": {"DENSITY": [924, 924, 924], "LWC": [0.0001, 0.0001, 0.0001]}},
        {"LAND": {"DENSITY": [920, 920, 920], "LWC": [0.02, 0.02, 0.02]}},
        {
            "SOLVER": {"TYPE": "multi-stream-ada", "N_FOURIER_MODES": 2},
            "SOLAR": {"SAA": None},
        },
        {
            "SOLVER": {"TYPE": "multi-stream-ada", "N_FOURIER_MODES": 1},
            "SOLAR": {"SAA": 100},
        },
        {
            "SOLVER": {"TYPE": "multi-stream-ada"},
            "LAND": {"LAYER_TYPE": [1, 1, 1], "GRAIN_SHAPE": [0, 0, 0]},
        },
        {
            "SPECTRAL": {"MODE": "band-snicar-default"},
            "SOLVER": {"TYPE": "multi-stream-ada", "ATMOSPHERE_COUPLING": True},
        },
        {
            "ATMOSPHERE": {"AEROSOL_PROPERTIES": "test.nc"},
            "SOLVER": {"TYPE": "multi-stream-ada", "ATMOSPHERE_COUPLING": False},
        },
        {
            "SPECTRAL": {"MODE": "monochromatic", "RESOLUTION": [200, 600, 10]},
            "SOLVER": {"TYPE": "multi-stream-ada"},
        },
        {
            "ATMOSPHERE": {"SKY_CONDITIONS": "clear"},
            "SPECTRAL": {"MODE": "monochromatic", "RESOLUTION": [400, 500, 10]},
            "SOLVER": {"TYPE": "multi-stream-ada", "ATMOSPHERE_COUPLING": False},
        },
    ],
)
def test_validation_errors_raise_exception(config_dict, wrong_configs):
    """
    Verify that an error is raised for invalid configurations.
    """

    temp_config = copy.deepcopy(config_dict)

    for key, value in wrong_configs.items():
        if (
            key in config_dict
            and isinstance(config_dict[key], dict)
            and isinstance(value, dict)
        ):
            temp_config[key].update(value)
        else:
            temp_config[key] = value

    # with pytest.raises(ValueError):
    #     Config.model_validate(temp_config)
    try:
        Config.model_validate(temp_config)
        pytest.fail("Expected error not raised")
    except ValueError as e:
        print(f"\n {e}")


def test_print_help(capsys):
    """
    Ensure print_help() function displays key sections.
    """
    Config.print_help()
    printer = capsys.readouterr()

    # check that it's not a single character
    assert len(printer.out) > 2

    # check that the main fields are printed
    assert "SOLVER" in printer.out
    assert "LAND" in printer.out
    assert "SPECTRAL" in printer.out
    assert "SOLAR" in printer.out
    assert "ATMOSPHERE" in printer.out
