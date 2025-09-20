#!/usr/bin/env python3
"""

@author: snicar-fx team

"""


from snicarfx.core.config_validator import Config


def test_test_yaml_input_file(test_input_file):
    """
    Test the fields of the input file used to test snicar-fx
    """

    # validate configuration
    Config.validate_yaml_file(test_input_file)


def test_core_yaml_input_file(core_input_file):
    """
    Test the fields of the input file used as an example to run
    snicar-fx
    """

    # validate configuration
    Config.validate_yaml_file(core_input_file)
