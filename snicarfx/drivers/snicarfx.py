#!/usr/bin/python

from biosnicar.classes import ColumnProperties, ModelInputs, SolarIrradiance
from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver


def run(input_file):
    """Calculate column optical properties to feed the RT solver.

    Args:
        None

    Returns:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    """

        
    model_inputs = ModelInputs(input_file)
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs) 
    
    outputs = adding_doubling_solver(
        column, irradiance
    )
        
    return outputs

if __name__ == "__main__":
    pass

    
