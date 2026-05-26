"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from snicarfx.core.session.config import Config


def test_test_yaml_input_file_twostream(test_input_file_twostream):
    """
    Test the range and type of the model parameters parsed from the yaml input
    file used to test snicar-fx.

    Parameters
    ----------
    test_input_file_twostream : str
        Path to test input file for two-stream configuration
    """

    # validate configuration
    Config.from_yaml(test_input_file_twostream)


def test_test_yaml_input_file_multistream_coupled(test_input_file_multistream_coupled):
    """
    Test the range and type of the model parameters parsed from the yaml input
    file used to test snicar-fx.

    Parameters
    ----------
    test_input_file_multistream_coupled : str
        Path to test input file for multi-stream coupled configuration
    """

    # validate configuration
    Config.from_yaml(test_input_file_multistream_coupled)

def test_test_yaml_input_file_multistream_uncoupled(test_input_file_multistream_uncoupled):
    """
    Test the range and type of the model parameters parsed from the yaml input
    file used to test snicar-fx.

    Parameters
    ----------
    test_input_file_multistream_coupled : str
        Path to test input file for multi-stream uncoupled configuration
    """

    # validate configuration
    Config.from_yaml(test_input_file_multistream_uncoupled)

