#!/usr/bin/python

from biosnicar.classes import (
    ColumnProperties,
    SolarIrradiance,
    ModelConfig
)
from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver


def run(input_file):
    """Calculate column optical properties to feed the RT solver.

    Args:
        None

    Returns:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    """

        
    model_config = ModelConfig(input_file)
    column = ColumnProperties(model_config)
    irradiance = SolarIrradiance(model_config) 
    
    outputs = adding_doubling_solver(
        column, irradiance
    )
        
    return outputs

if __name__ == "__main__":
    pass

    
