#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the ModelConfig class.

No need to test further for now since pretty much all of the attributes
of ModelConfig come from the YAML parser which is itself tested.

"""

from snicarfx.classes import ColumnProperties, SolarIrradiance, ModelConfig


def test_modelconfig_loading(model_config):
    """
    Check general ModelConfig initialization
    """

    # Check the instance type is correct
    assert isinstance(model_config, ModelConfig)

    # Check expected attributes exist and have valid values
    assert hasattr(model_config, "dir_base")
    assert hasattr(model_config, "lap_path")
    assert hasattr(model_config, "solar_fluxes_path")
    assert hasattr(model_config, "inputs")


def test_modelconfig_top_level_keys(model_config):
    """
    Check top level dictionnary keys
    """
    assert "RTM" in model_config.inputs
    assert "ICE" in model_config.inputs
    assert "LIGHT_ABSORBING_PARTICLES" in model_config.inputs
    assert len(model_config.inputs) == 3
