"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import yaml

import snicarfx


class ModelInputs:
    """
    Load and organize model configuration from a YAML input file.

    This class parses a YAML input file containing nested model parameters
    and sets up paths for required data directories.

    Attributes
    ----------
    inputs : dict
        Parsed configuration dictionary loaded from the YAML input file.
    dir_base : str
        Base directory of the `snicarfx` package, used to resolve data paths.
    data_path : str
        Path to the data directory.
    lap_path : str
        Path to the light-absorbing particles data directory.
    solar_fluxes_path : str
        Path to the solar fluxes data directory.
    """

    def __init__(self, input_file):
        """
        Initialize a ModelInputs instance from a YAML configuration file.

        This method reads the YAML input file, parses its contents into a
        nested dictionary, and sets paths to required data directories
        based on the location of the `snicarfx` package.

        Parameters
        ----------
        input_file : str
            Path to the YAML file containing model input parameters.
        """
        with open(input_file) as ymlfile:
            self.inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.dir_base = snicarfx.__file__.rsplit("/", 3)[0]
        self.data_path = self.dir_base + "/data/"
        self.lap_path = self.data_path + "light_absorbing_particles/"
        self.solar_fluxes_path = self.data_path + "solar_fluxes/"
