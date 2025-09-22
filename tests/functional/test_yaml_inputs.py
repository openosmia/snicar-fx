"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

from snicarfx.core.config_validator import Config


def test_test_yaml_input_file(test_input_file):
    """
    Test the range and type of the model parameters parsed from the yaml input 
    file used to test snicar-fx.
    
    Parameters
    ----------
    test_input_file : str
        Path to test input file
    """
    # validate configuration
    Config.validate_yaml_file(test_input_file)


def test_core_yaml_input_file(core_input_file):
    """
    Test the range and type of the model parameters parsed from the default 
    yaml input file used to run snicar-fx.
    
    Parameters
    ----------
    core_input_file : str
        Path to default input file
    """

    # validate configuration
    Config.validate_yaml_file(core_input_file)
