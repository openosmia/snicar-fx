"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

from snicarfx.core import ColumnProperties, ModelInputs, SolarIrradiance
from snicarfx.core.config_validator import Config
from snicarfx.core.two_stream_solver import solve_two_stream_rt


def run_two_stream(input_file):
    """
    Run the SNICAR-fx using inputs from the YAML configuration file using the 
    two-stream solver.
    
    This function performs the following steps:
    
    - Validates the YAML input file.
    - Initializes model inputs.
    - Computes optical properties of the snow/ice column.
    - Computes solar irradiance at the surface.
    - Solves the radiative transfer equations using the two-stream solver.
    
    Parameters
    ----------
    input_file : str 
        Path to the YAML input configuration file.
    
    Returns
    -------
    outputs : object
        outputs of the radiative transfer solver. 
    
    Raises
    ------
    ValueError
        If the input file is invalid or fails schema validation.
    """

    # make sure the input file is valid
    Config.validate_yaml_file(input_file)

    # create the different instances
    model_inputs = ModelInputs(input_file)
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs)

    # solve radiative transfer equations
    outputs = solve_two_stream_rt(column, irradiance)

    return outputs


if __name__ == "__main__":
    pass
