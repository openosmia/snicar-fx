import os
import numpy as np
import yaml
import biosnicar

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
        with open(input_file, "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        wvl1 = inputs["RTM"]["WVL_START"]
        wvl2 = inputs["RTM"]["WVL_END"]
        resolution = inputs["RTM"]["RESOLUTION"]
        self.wavelengths = np.arange(
            wvl1*1e-3, 
            wvl2*1e-3,
            resolution*1e-3)
        self.nbr_wvl = inputs["RTM"]["NBR_WVL"] 
        self.vis_max_idx = inputs["RTM"]["VIS_MAX_IDX"]
        self.nir_max_idx = inputs["RTM"]["NIR_MAX_IDX"] 
        self.dir_base = str(os.path.dirname(os.path.dirname(biosnicar.__file__)))+ "/"


        
        