#!/usr/bin/env python3
"""

Test the ModelInputs class.

No need to test further for now since pretty much all of the attributes
of ModelInputs come from the YAML parser which is itself tested.

"""

from src.snicarfx.core import ModelInputs


def test_modelinputs_loading(model_inputs):
    """
    Check general ModelConfig initialization
    """

    # Check the instance type is correct
    assert isinstance(model_inputs, ModelInputs)

    # Check expected attributes exist and have valid values
    assert hasattr(model_inputs, "dir_base")
    assert hasattr(model_inputs, "lap_path")
    assert hasattr(model_inputs, "solar_fluxes_path")
    assert hasattr(model_inputs, "inputs")


def test_model_inputs_top_level_keys(model_inputs):
    """
    Check top level dictionnary keys
    """
    assert "RTM" in model_inputs.inputs
    assert "ICE" in model_inputs.inputs
    assert "LIGHT_ABSORBING_PARTICLES" in model_inputs.inputs
    assert len(model_inputs.inputs) == 3
