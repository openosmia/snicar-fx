import os

import yaml

import snicarfx


class ModelConfig:
    """Model configuration.

    Attributes:
        dir_base: base directory
        dir_wvl: path to wavelengths in csv file
        sphere_ice_path: directory containing OPs for spherical ice grains
        hex_ice_path: directory containing OPs for hexagonal ice grains
        bubbly_ice_path: directory containing OPs for bubbly ice
        ri_ice_path: path to file containing pure ice refractive index
        op_dir_stubs: sstring stubs for ice optical property files
        wavelengths: array of wavelengths in nm (default 0.205 - 4.995 um)
        nbr_wvl: number of wavelengths (default 480)
        vis_max_idx: index for upper visible wavelength (default 0.75 um)
        nir_max_idx: index for upper NIR wavelength (default 4.995 um)

    """

    def __init__(self, input_file):
        with open(input_file) as ymlfile:
            self.inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.dir_base = (
            str(os.path.dirname(os.path.dirname(snicarfx.__file__)))
            + "/")
        self.op_path = self.dir_base + '/data/optical_properties/'
        self.lap_path = self.op_path + 'light_absorbing_particles/'
        self.solar_fluxes_path = self.op_path + 'solar_fluxes/'



        
        