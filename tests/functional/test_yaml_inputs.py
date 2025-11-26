"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from snicarfx.core.session.config import Config


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
    Config.from_yaml(test_input_file)


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
    Config.from_yaml(core_input_file)
