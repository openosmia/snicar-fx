import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import biosnicar

class Ice:
    """Snow or ice column physical & optical properties.

    Instances contain all the physical properties of each vertical layer of the
    snow or ice column and the underlying surface.

    Attributes:
        dz: array containing thickness of each layer in m
        layer_type: array containing type (0 = grains, 1 = solid ice) in each layer
        rho: array containing density of each layer in kg/m3
        sfc: array with reflectance of underlying surface per wavelength
        rf: refractive index to use, 0, 1, 2 or 3 (see docs for definition)
        shp: grain shape per layer where layer_type==0
        rds: grain radius (layer_type==0) or bubble radius (layer_type==1) in each layer
        hex_side: length of each side of hexagonal face for grain_shp==4
        hex_length: column length for hexagonal face for grain_shp==4
        nbr_lyr: number of vertical layers
    """

    def __init__(self, input_file):
        with open(input_file, "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.dz = inputs["ICE"]["DZ"]
        self.layer_type = inputs["ICE"]["LAYER_TYPE"]
        self.rho = inputs["ICE"]["DENSITY"]
        self.sfc = np.ones(inputs["RTM"]["NBR_WVL"]) * inputs["ICE"]["SFC"]
        self.rf_type = inputs["ICE"]["RF_TYPE"]
        self.grain_shape = inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = inputs["ICE"]["LWC"]
        self.ssa = inputs["ICE"]["SPECIFIC_SURFACE_AREA"]
        self.hex_side = inputs["ICE"]["HEX_SIDE"]
        self.hex_length = inputs["ICE"]["HEX_LENGTH"]
        self.nbr_lyr = len(self.dz)
        
        self.nbr_wvl = inputs["RTM"]["NBR_WVL"]
        
        wvl1 = inputs["RTM"]["WVL_START"]
        wvl2 = inputs["RTM"]["WVL_END"]
        resolution = inputs["RTM"]["RESOLUTION"]
        self.path_op = (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
                    + "/" 
                    + f'data/OP_data/{wvl1}_{wvl2}_{resolution}/'
            )
        # self.path_op = (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
        #             + "/" 
        #             + 'data/OP_data/480band/'
        #     )
        
        # init the ssps
        self.ext = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.g = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.L_snw = np.ones(self.nbr_lyr)
        
        
    def set_refractive_index(self):
        """Calculates ice refractive index from class attributes.

        Args:
            self

        Returns:
            ref_idx_im: imaginary part of refractive index
            ref_idx_re: real part of refractive index
            fl_r_dif_a: precomputed diffuse reflectance for above)
            fl_r_dif_b: precomputed diffuse reflectance for below)

        Raises:
            ValueError if rf out of range
        """
        if self.rf_type not in ["Wrn84", "Wrn08", "Pic16", "Coop21"]:
            raise ValueError("Ice refractive index not found")

        refidx_file = xr.open_dataset(self.path_op + "refractive_indices.nc")
        
        self.ref_idx_re = refidx_file[str("re_" + self.rf_type)].values
        self.ref_idx_im = refidx_file[str("im_" + self.rf_type)].values
        
        self.ref_idx_im_water = pd.read_csv(self.path_op 
            + 'refractive_index_water_273K_Rowe2020.csv'
        ).k.values
        
    def set_diffuse_fresnel_coefficients(self):
        """Calculates fresnel diffuse reflectivity coefficients from class 
        attributes.

        Args:
            self

        Returns:
            fl_r_dif_a: precomputed diffuse reflectance for above)
            fl_r_dif_b: precomputed diffuse reflectance for below)

        Raises:
            ValueError if rf out of range
        """
        
        fresnel_diffuse_file = xr.open_dataset(self.path_op 
                                               + "fresnel_diffuse_coefficients.nc")
        
        self.fl_r_dif_a = fresnel_diffuse_file[
            str("R_dif_fa_ice_" + self.rf_type)
        ].values
        self.fl_r_dif_b = fresnel_diffuse_file[
            str("R_dif_fb_ice_" + self.rf_type)
        ].values
    
 