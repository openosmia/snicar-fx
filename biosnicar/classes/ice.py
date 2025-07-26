import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import biosnicar

class Ice:
    """Snow or ice column physical properties.

    Instances of Ice contain all the physical properties of each vertical layer of the
    snow or ice column and the underlying surface.

    Attributes:
        dz: array containing thickness of each layer in m
        layer_type: array containing type (0 = grains, 1 = solid ice) in each layer
        rho: array containing density of each layer in kg/m3
        sfc: array with reflectance of underlying surface per wavelength
        rf: refractive index to use, 0,1,2 or 3 (see docs for definition)
        shp: grain shape per layer where layer_type==1
        rds: grain radius (layer_type==0) or bubble radius (layer_type==0) in each layer
        water: radius of grain+water coating in each layer where layer_type==0
        hex_side: length of each side of hexagonal face for grain_shp==4
        hex_length: column length for hexagonal face for grain_shp==4
        shp_fctr: ratio of nonspherical eff radii to equal vol sphere, in each layer
        ar: aspect ratio of grains in each layer where layer_type==0
        nbr_lyr: number of vertical layers
    """

    def __init__(self, input_file):
        # use config to calculate refractive indices
        with open(input_file, "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.dz = inputs["ICE"]["DZ"]
        self.layer_type = inputs["ICE"]["LAYER_TYPE"]
        self.rho = inputs["ICE"]["RHO"]
        self.sfc = np.ones(inputs["RTM"]["NBR_WVL"]) * inputs["ICE"]["SFC"]
        self.rf = inputs["ICE"]["RF"]
        self.grain_shape = inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = inputs["ICE"]["LWC"]
        self.rds = inputs["ICE"]["RDS"]
        self.hex_side = inputs["ICE"]["HEX_SIDE"]
        self.hex_length = inputs["ICE"]["HEX_LENGTH"]
        self.nbr_lyr = len(self.dz)
        
        self.nbr_wvl = inputs["RTM"]["NBR_WVL"]
        
        wvl1 = inputs["RTM"]["WVL_START"]
        wvl2 = inputs["RTM"]["WVL_END"]
        resolution = inputs["RTM"]["RESOLUTION"]
        self.wvl = np.arange(
            wvl1*1e-3, 
            wvl2*1e-3,
            resolution*1e-3)
        # self.path_op = (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
        #             + "/" 
        #             + f'data/OP_data/{wvl1}_{wvl2}_{resolution}/'
        #     )
        self.path_op = (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
                    + "/" 
                    + f'data/OP_data/480band/'
            )
        
        
    def calculate_refractive_index(self):
        """Calculates ice refractive index from initialized class attributes.

        Takes self.rf and config from inpouts.yaml and uses them to calculate
        new attributes related to the ice refractive index.

        Args:
            self

        Returns:
            ref_idx_im: imaginary part of refractive index
            ref_idx_re: real part of refractive index
            fl_r_dif_a: precomputed diffuse reflectance "perpendicular polarized)
            fl_r_dif_b: precomputed diffuse reflectance "parallel polarized)

        Raises:
            ValueError if rf out of range
        """
        if self.rf not in ["Wrn84", "Wrn08", "Pic16", "Coop21"]:
            raise ValueError("Ice refractive index not found")

        refidx_file = xr.open_dataset(self.path_op + "rfidx_ice.nc")
        
        fresnel_diffuse_file = xr.open_dataset(self.path_op + "fl_reflection_diffuse.nc")

        self.ref_idx_re = refidx_file[str("re_" + self.rf)].values
        self.ref_idx_im = refidx_file[str("im_" + self.rf)].values

        self.fl_r_dif_a = fresnel_diffuse_file[
            str("R_dif_fa_ice_" + self.rf)
        ].values
        self.fl_r_dif_b = fresnel_diffuse_file[
            str("R_dif_fb_ice_" + self.rf)
        ].values
        
        self.ref_idx_im_water = pd.read_csv(self.path_op 
            + 'refractive_index_water_273K_Rowe2020.csv'
        ).k.values
    
    def calculate_column_ops(self):
        
        # init the ssps
        self.ext = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        self.g = np.ones((self.nbr_lyr,
                            self.nbr_wvl))
        
        for lyr in range(self.nbr_lyr):
            if self.layer_type[lyr] > 0: # ice - only air inclusions for now
                # eq_rds = 3 * vlm_frac_air / (self.ssa * self.rho[lyr]) # Eq from Whicker
                vlm_frac_ice = (self.rho[lyr] - self.lwc[lyr] * 1000) / 917
                vlm_frac_air = 1 - self.lwc[lyr] - vlm_frac_ice
                sca_cff_vlm_air_bbl = np.ones(self.nbr_wvl) * 2 * 0.75 / (self.rds[lyr] * 1e-6)
                scattering_cff = (
                    sca_cff_vlm_air_bbl 
                    * vlm_frac_air 
                    / self.rho[lyr]
                    )

                abs_cff = (4 
                           * np.pi 
                           / (self.wvl * 1e-6) 
                           / self.rho[lyr]
                           * (
                               vlm_frac_ice * self.ref_idx_im
                               + self.lwc[lyr] * self.ref_idx_im_water
                               )
                           )
                    
                self.ext[lyr, :] = (
                    scattering_cff
                    + abs_cff
                    )
                self.ss_alb[lyr, :] = (
                    scattering_cff 
                    / self.ext[lyr, :]
                    )
                self.g[lyr, :] = np.ones(self.nbr_wvl) * 0.86
                
            else: #snow
                ssa = 3 / (917 * self.rds[lyr] * 1e-6)
                self.ext[lyr, :] = (self.rho[lyr] * ssa / 2) / self.rho[lyr]
                W = 0.0611 + 0.17 * (self.ref_idx_re - 1.3)
                k_eq = (self.lwc[lyr] * self.ref_idx_im_water 
                          + (1-self.lwc[lyr]) * self.ref_idx_im
                          )
                c = 24.0 * np.pi * k_eq / (917.0 * self.wvl * 1e-6) / ssa
                
                if self.grain_shape[lyr] == 0: 
                    B0 = 1.25
                    g0=0.895
                    B = B0 + 0.4 * (self.ref_idx_re - 1.3)
                    phi = 2.0 / 3 * B / (1 - W)
                    self.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))
                    y = 0.728 + 0.752 * (self.ref_idx_re - 1.3)
                    ginf = 0.9751 - 0.105 * (self.ref_idx_re - 1.3)
                    g00 = g0 - 0.38 * (self.ref_idx_re - 1.3)
                    self.g[lyr,:] = ginf - (ginf - g00) * np.exp(-y * c) 
                    
                elif self.grain_shape[lyr] == 1:
                    self.g[lyr,:] = np.ones(self.nbr_wvl) * 0.815
                    B = self.ref_idx_re**2
                    phi = 2.0 / 3 * B / (1 - W)
                    self.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))
            

        
        


        