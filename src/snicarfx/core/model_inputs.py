import yaml

import snicarfx


class ModelInputs:
    """Model configuration.

    Attributes:
        inputs: string with data from input file
        dir_base: base directory
        op_path: directory to op data
        lap_path: directory to lap
        solar_fluxes_path: directory to solar_fluxes

    """

    def __init__(self, input_file):
        with open(input_file) as ymlfile:
            self.inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        self.dir_base = snicarfx.__file__.rsplit('/', 3)[0]
        self.op_path = self.dir_base + '/data/optical_properties/'
        self.lap_path = self.op_path + 'light_absorbing_particles/'
        self.solar_fluxes_path = self.op_path + 'solar_fluxes/'



        
        