#!/usr/bin/python

from snicarfx.core import ColumnProperties, ModelInputs, SolarIrradiance
from snicarfx.core.adding_doubling_solver import solve_adding_doubling
from snicarfx.core.config_validator import Config


def run(input_file):
    """Calculate column optical properties to feed the RT solver.

    Args:
        None

    Returns:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    """

    # make sure the input file is valid
    Config.validate_yaml_file(input_file)

    # create the different instances
    model_inputs = ModelInputs(input_file)
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs)

    # solve radiative transfer equations
    outputs = solve_adding_doubling(column, irradiance)

    return outputs


if __name__ == "__main__":
    pass
