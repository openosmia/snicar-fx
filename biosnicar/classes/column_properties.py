import numpy as np
import xarray as xr

class ColumnProperties:
    """Snow or ice column physical & optical properties, including light 
    absorbing particles.

    Attributes:
        TO DO
    """

    def __init__(self, modelconfig):
        self.modelconfig = modelconfig
        self.thickness = modelconfig.inputs["ICE"]["THICKNESS"]
        self.layer_type = modelconfig.inputs["ICE"]["LAYER_TYPE"]
        self.density = modelconfig.inputs["ICE"]["DENSITY"]
        self.nbr_wvl = modelconfig.inputs["RTM"]["NBR_WVL"]
        self.sfc = np.ones(
            self.nbr_wvl
            ) * modelconfig.inputs["ICE"]["SFC"]
        self.rf_type = modelconfig.inputs["ICE"]["RF_TYPE"]
        self.grain_shape = modelconfig.inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = modelconfig.inputs["ICE"]["LWC"]
        self.ssa = modelconfig.inputs["ICE"]["SPECIFIC_SURFACE_AREA"]
        self.nbr_lyr = len(self.density) 
        
        self.wavelengths = np.arange(
                            self.modelconfig.inputs["RTM"]["WVL_START"]*1e-9, 
                            self.modelconfig.inputs["RTM"]["WVL_END"]*1e-9,
                            self.modelconfig.inputs["RTM"]["RESOLUTION"]*1e-9)

        # init the ssps
        self.ext_cff = np.ones((self.nbr_lyr,
                            self.modelconfig.inputs["RTM"]["NBR_WVL"]))
        self.ss_alb = np.ones((self.nbr_lyr,
                            self.modelconfig.inputs["RTM"]["NBR_WVL"]))
        self.asm_prm = np.ones((self.nbr_lyr,
                            self.modelconfig.inputs["RTM"]["NBR_WVL"]))
        self.tau = np.ones((self.nbr_lyr,
                            self.modelconfig.inputs["RTM"]["NBR_WVL"]))
        self.layer_mass = np.zeros(self.nbr_lyr)
    
        # calculate ref idx and fresnel coefficients at input resolution
        self.set_refractive_index_and_diffuse_fresnel_coeffs()
        self.get_lap_properties()
        
    def set_refractive_index_and_diffuse_fresnel_coeffs(self):
        """Calculates ice refractive index and pre-calculated diffuse
        fresnel coefficients at user-defined resolution.

        Args:
            self
            
        """

        # set spectral resolution
        resolution = self.modelconfig.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.modelconfig.inputs["RTM"]["WVL_START"]
        wvl_end = self.modelconfig.inputs["RTM"]["WVL_END"]
        wvl_high_res = np.arange(200, 5001, 1)
        
        idx1 = np.where(wvl_high_res == wvl_start)[0][0]
        idx2 = np.where(wvl_high_res == wvl_end)[0][0]
        
        refidx_file = xr.open_dataset(self.modelconfig.op_path
                                      + "refractive_indices.nc")
        fresnel_diffuse_file = xr.open_dataset(
            self.modelconfig.op_path 
            + "fresnel_diffuse_coefficients.nc")
        
        self.ref_idx_re = refidx_file[
            str("re_" + self.rf_type)
            ].values[idx1: idx2: resolution]
        self.ref_idx_im = refidx_file[
            str("im_" + self.rf_type)
            ].values[idx1: idx2: resolution]
        self.ref_idx_im_water = refidx_file[
            "im_Row20"
            ].values[idx1: idx2: resolution]
        self.fl_r_dif_a = fresnel_diffuse_file[
            str("R_dif_fa_ice_" + self.rf_type)
        ].values[idx1: idx2: resolution]
        self.fl_r_dif_b = fresnel_diffuse_file[
            str("R_dif_fb_ice_" + self.rf_type)
        ].values[idx1: idx2: resolution]
        
    def get_lap_properties(self):
        
        """Calculates light absorbing particle properties at user-defined 
        resolution.

        Args:
            self
            
        """

        # set spectral resolution
        resolution = self.modelconfig.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.modelconfig.inputs["RTM"]["WVL_START"]
        wvl_end = self.modelconfig.inputs["RTM"]["WVL_END"]
        
        # initialize properties
        nb_laps = len(self.modelconfig.inputs["LIGHT_ABSORBING_PARTICLES"])
        
        self.lap_ss_alb = np.zeros((nb_laps,  
                                    self.modelconfig.inputs["RTM"]["NBR_WVL"])
                                   )
        self.lap_asm_prm = np.zeros((nb_laps, 
                                    self.modelconfig.inputs["RTM"]["NBR_WVL"])
                                   )
        self.lap_ext_cff = np.zeros((nb_laps,  
                                    self.modelconfig.inputs["RTM"]["NBR_WVL"])
                                   )
        
        
        # get concentrations
        self.lap_concentrations = np.vstack(
                [np.array(
                    # convert from ppb to kg kg-1
                    self.modelconfig.inputs[
                        "LIGHT_ABSORBING_PARTICLES"][name]["CONC"]) * 1e-9
                 if self.modelconfig.inputs[
                         "LIGHT_ABSORBING_PARTICLES"][name]["UNIT"] == 0
                 else np.array(
                     # convert from cells mL-1 to kg kg-1 (1cell=1ng)
                     self.modelconfig.inputs[
                         "LIGHT_ABSORBING_PARTICLES"][name]["CONC"]) * 0.917 * 1e-9
                 for name in self.modelconfig.inputs[
                         "LIGHT_ABSORBING_PARTICLES"]
                     ]
                ).T
    
        # get properties in a large array        
        for i, lap in enumerate(
                self.modelconfig.inputs["LIGHT_ABSORBING_PARTICLES"]):
            
            # first get the ext coeff tag 
            if self.modelconfig.inputs[
                    "LIGHT_ABSORBING_PARTICLES"][lap]["COATED"]:
                ext_cff_tag = "ext_cff_mss_ncl"
            else:
                ext_cff_tag = "ext_cff_mss"
            
            
            # then interpolate the properties to the right resolution
            properties = xr.open_dataset(
                self.modelconfig.lap_path +
                self.modelconfig.inputs[
                    "LIGHT_ABSORBING_PARTICLES"][lap]["FILE"]
                )
            
            ss_alb = np.interp(np.arange(wvl_start, 
                                         wvl_end, 
                                         resolution),
                                properties.wvl.values*1e9, # from m to nm
                                properties["ss_alb"].values)
            self.lap_ss_alb[i, :] = ss_alb
            asm_prm = np.interp(np.arange(wvl_start, 
                                         wvl_end, 
                                         resolution),
                                properties.wvl.values*1e9, # from m to nm
                                properties["asm_prm"].values)
            self.lap_asm_prm[i, :] = asm_prm
            ext_cff = np.interp(np.arange(wvl_start, 
                                         wvl_end, 
                                         resolution),
                                properties.wvl.values*1e9, # from m to nm
                                properties[ext_cff_tag].values) 
            self.lap_ext_cff[i, :] = ext_cff
            
            
        
        
        
        
        
        
        
        
    
 