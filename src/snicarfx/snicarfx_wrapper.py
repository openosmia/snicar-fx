"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from snicarfx.core import LandColumn, AtmosphereColumn, SolarIrradiance
from snicarfx.core import Config
from snicarfx.core.solvers.two_stream_solver import solve_two_stream_rt


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

    input_file = "./src/snicarfx/inputs.yaml"

    # make sure the input file is valid
    config = Config.from_yaml(input_file)

    # create the different instances
    column = ColumnProperties(config)
    irradiance = SolarIrradiance(config)

    # solve radiative transfer equations
    outputs = solve_two_stream_rt(column, irradiance)

    simulation = Simulation("inputs.yaml")

    results = simulation.run()

    return outputs


if __name__ == "__main__":
    pass
