#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file includes two main types of tests:

1) tests on the outputs of snicarfx against a benchmark of SNICAR_ADv4
implemented in Matlab

2) regular tests of the snicarfx software

"""

from snicarfx.classes import ColumnProperties, SolarIrradiance, ModelConfig
from snicarfx.rt_solvers import solve_adding_doubling

TEST_INPUT_FILE = "./tests/inputs_tests.yaml"
