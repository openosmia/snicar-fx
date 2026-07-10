"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from pydantic import BaseModel
import pytest
import copy
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
@pytest.mark.parametrize("wrong_configs", [
    {"SOLVER": {"TYPE": "two-stream-ad", "ATMOSPHERE_COUPLING": True}},
    {"SOLVER": {"TYPE": "two-stream-ad", "OUTPUT_LEVELS": "TOA"}},
    {"SOLVER": {"TYPE": "two-stream-ad", "DELTA_SCALING": "M+"}},
    {"SOLVER": {"TYPE": "two-stream-ad", "N_LEGENDRE_MOMENTS": "15"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "N_STREAMS": "15"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "N_STREAMS": "15", "N_LEGENDRE_MOMENTS": "20"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "N_STREAMS": "15", "N_FOURIER_MODES": "20"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "ATMOSPHERE_COUPLING": False, "OUTPUT_LEVELS": "TOA"}},
    {"SOLVER": {"TYPE": "two-stream-ad", "N_FOURIER_MODES": "2"}},
    {"SOLVER": {"TYPE": "two-stream-ad", "POLAR_ANGLES": "[10, 40, 10]"}},
    {"SOLVER": {"TYPE": "two-stream-ad", "AZIMUTH_ANGLES": "[10, 40, 10]"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "POLAR_ANGLES": "[40, 10, 10]"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "POLAR_ANGLES": "[6, 7, 100]"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "AZIMUTH_ANGLES": "[40, 10, 10]"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "AZIMUTH_ANGLES": "[6, 7, 100]"}},
    {"SOLVER": {"TYPE": "multi-stream-ada", "AZIMUTH_ANGLES": "[10, 100, 20]",  "N_FOURIER_MODES": "1"}},
  
])

def test_validation_errors_raise_exception(config_dict, wrong_configs):
    """
    Verify that an error is raised for invalid configurations.
    """

    temp_config = copy.deepcopy(config_dict)
    
    for key, value in wrong_configs.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
            temp_config[key].update(value)
        else:
            temp_config[key] = value

    with pytest.raises(ValueError):
        Config.model_validate(temp_config)
