"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

from snicarfx.core import ModelInputs


def test_modelinputs_loading(model_inputs):
    """
    Verify that an instance of ModelInputs has the correct type and that 
    required attributes exist.
    
    Parameters
    ----------
    model_inputs : ModelInputs
        Instance of the ModelInputs class
    """

    # Check the instance type is correct
    assert isinstance(model_inputs, ModelInputs)

    # Check expected attributes exist and have valid values
    assert hasattr(model_inputs, "lap_path")
    assert hasattr(model_inputs, "solar_fluxes_path")
    assert hasattr(model_inputs, "inputs")


def test_model_inputs_top_level_keys(model_inputs):
    """
    Verify that the keys of the `inputs` attributes of an instance of ModelInputs
    are correct. 
    
    Parameters
    ----------
    model_inputs : ModelInputs
        Instance of the ModelInputs class
    """
    assert "RTM" in model_inputs.inputs
    assert "ICE" in model_inputs.inputs
    assert "LIGHT_ABSORBING_PARTICLES" in model_inputs.inputs
    assert len(model_inputs.inputs) == 3
