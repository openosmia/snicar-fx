#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: snicar-fx team

"""

import yaml
import pytest
from snicarfx.core.config_validator import Config


def test_test_yaml_input_file(test_input_file):
    """
    Test the fields of the input file used to test snicar-fx
    """

    # open file
    with open(test_input_file) as ymlfile:
        inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # validate configuration
    config = Config.model_validate(inputs)
