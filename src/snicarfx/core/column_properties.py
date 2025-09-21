import numpy as np
import xarray as xr


class ColumnProperties:
    """
    Physical and optical properties of a snow or ice column.

    This class computes and stores the properties of a snow/ice column for each
    layer based on the YAML input file. 

    Attributes
    ----------
    model_inputs : ModelInputs
        An instance of the ModelInputs class containing model input data parsed
        from the YAML configuration file.
    layer_type : list 
        Type of layer (0 for snow grains in air, 1 for air bubbles in ice).
    nbr_lyr : int
        Number of layers in the column.
    thickness_profile : list 
        Thicknesses [m] of each layer of the snow or ice column.
    density : list 
        Density of snow/ice for each layer [kg/m3].
    rf_type : str
        Source of refractive index data.
    grain_shape : list
        Identifier for grain shape model for each layer.
    lwc : list
        Liquid water content fraction for each layer [0-1].
    ssa : list
        Specific surface area of snow or ice in each layer [m2/kg].
    sfc : float
        Reflectance of underlying surface (wavelength independent).
    wavelengths : list
        Spectral grid used for all optical property calculations [m].
    nbr_wvl : int
        Number of wavelengths in the spectral grid.
    lap_ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each LAP [unitless].
    lap_asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each LAP [unitless].
    lap_ext_cff : ndarray
        Wavelength-dependent mass extinction coefficient of each LAP [m2/kg].
    lap_concentrations : ndarray
        Mass concentrations of LAPs per layer [kg/kg].
    ext_cff : ndarray
        Wavelength-dependent mass extinction coefficient of each layer [m2/kg].
    ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each layer [unitless].
    asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each layer [unitless].
    tau : ndarray
        Wavelength-dependent optical thickness of each layer [unitless].
    layer_mass : ndarray
        Mass per unit area of each layer [kg/m2].

    """

    def __init__(self, model_inputs):
        self.model_inputs = model_inputs
        self.layer_type = model_inputs.inputs["ICE"]["LAYER_TYPE"]
        self.nbr_lyr = len(self.layer_type)
        self.thickness_profile = model_inputs.inputs["ICE"]["THICKNESS"]
        self.density = model_inputs.inputs["ICE"]["DENSITY"]
        self.rf_type = model_inputs.inputs["ICE"]["RF_TYPE"]
        self.grain_shape = model_inputs.inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = model_inputs.inputs["ICE"]["LWC"]
        self.ssa = model_inputs.inputs["ICE"]["SPECIFIC_SURFACE_AREA"]

        self.wavelengths = (
            np.arange(
                self.model_inputs.inputs["RTM"]["WVL_START"],
                self.model_inputs.inputs["RTM"]["WVL_END"],
                self.model_inputs.inputs["RTM"]["RESOLUTION"],
            )
            * 1e-9
        )

        self.nbr_wvl = len(self.wavelengths)
        self.sfc = np.ones(self.nbr_wvl) * model_inputs.inputs["ICE"]["SFC"]

        # init the ssps
        self.ext_cff = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.asm_prm = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.layer_mass = np.zeros(self.nbr_lyr)

        self.set_refractive_index_and_diffuse_fresnel_coeffs()
        self.set_column_ops_without_laps()

        if self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"] is not None:
            self.set_lap_properties()
            self.update_column_ops_with_laps()

    def set_refractive_index_and_diffuse_fresnel_coeffs(self):
        """
        Load and set refractive indices and diffuse Fresnel coefficients.
    
        This method loads high-resolution ice/water refractive index data as 
        well as diffuse Fresnel reflection coefficients, and interpolates them
        to the model's spectral resolution.
        """

        # set spectral resolution
        resolution = self.model_inputs.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.model_inputs.inputs["RTM"]["WVL_START"]
        wvl_end = self.model_inputs.inputs["RTM"]["WVL_END"]
        wvl_high_res = np.arange(200, 5001, 1)

        idx1 = np.where(wvl_high_res == wvl_start)[0][0]
        idx2 = np.where(wvl_high_res == wvl_end)[0][0]

        refidx_file = xr.open_dataset(
            self.model_inputs.data_path + "refractive_indices.nc"
        )
        fresnel_diffuse_file = xr.open_dataset(
            self.model_inputs.data_path + "fresnel_diffuse_coefficients.nc"
        )

        self.ref_idx_re = refidx_file[str("re_" + self.rf_type)].values[
            idx1:idx2:resolution
        ]
        self.ref_idx_im = refidx_file[str("im_" + self.rf_type)].values[
            idx1:idx2:resolution
        ]
        self.ref_idx_im_water = refidx_file["im_Row20"].values[idx1:idx2:resolution]
        self.fl_r_dif_a = fresnel_diffuse_file[
            str("R_dif_fa_ice_" + self.rf_type)
        ].values[idx1:idx2:resolution]
        self.fl_r_dif_b = fresnel_diffuse_file[
            str("R_dif_fb_ice_" + self.rf_type)
        ].values[idx1:idx2:resolution]

    def set_lap_properties(self):
        """
        Load and set optical properties of light-absorbing particles (LAPs).
    
        This method sets the properties of each LAP defined in the input 
        configuration, converting their concentrations to consistent units, 
        and interpolating their properties to the model's spectral grid.
        """

        # set spectral resolution
        resolution = self.model_inputs.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.model_inputs.inputs["RTM"]["WVL_START"]
        wvl_end = self.model_inputs.inputs["RTM"]["WVL_END"]

        # initialize properties
        nb_laps = len(self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"])

        self.lap_ss_alb = np.zeros((nb_laps, self.nbr_wvl))
        self.lap_asm_prm = np.zeros((nb_laps, self.nbr_wvl))
        self.lap_ext_cff = np.zeros((nb_laps, self.nbr_wvl))

        # get concentrations
        self.lap_concentrations = np.vstack(
            [
                (
                    np.array(
                        # convert from ppb to kg kg-1
                        self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"][name][
                            "CONC"
                        ]
                    )
                    * 1e-9
                    if self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"][name][
                        "UNIT"
                    ]
                    == 0
                    else np.array(
                        # convert from cells mL-1 to kg kg-1 (1cell=1ng)
                        self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"][name][
                            "CONC"
                        ]
                    )
                    * 0.917
                    * 1e-9
                )
                for name in self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"]
            ]
        ).T

        # get properties in a large array
        for i, lap in enumerate(self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"]):

            # first get the ext coeff tag
            if self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"][lap]["COATED"]:
                ext_cff_tag = "ext_cff_mss_ncl"
            else:
                ext_cff_tag = "ext_cff_mss"

            # then interpolate the properties to the right resolution
            properties = xr.open_dataset(
                self.model_inputs.lap_path
                + self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"][lap]["FILE"]
            )

            ss_alb = np.interp(
                np.arange(wvl_start, wvl_end, resolution),
                properties.wvl.values * 1e9,  # from m to nm
                properties["ss_alb"].values,
            )
            self.lap_ss_alb[i, :] = ss_alb
            asm_prm = np.interp(
                np.arange(wvl_start, wvl_end, resolution),
                properties.wvl.values * 1e9,  # from m to nm
                properties["asm_prm"].values,
            )
            self.lap_asm_prm[i, :] = asm_prm
            ext_cff = np.interp(
                np.arange(wvl_start, wvl_end, resolution),
                properties.wvl.values * 1e9,  # from m to nm
                properties[ext_cff_tag].values,
            )
            self.lap_ext_cff[i, :] = ext_cff
            
    def set_column_ops_without_laps(self):
        """
        Compute optical properties of a clean snow/ice column (no LAPs).
    
        This method calculates wavelength-dependent extinction coefficients, 
        single scattering albedo, asymmetry parameters, and optical thickness 
        for each layer based on the input physical parameters and refractive 
        indices. Different models are used depending on whether layers are made
        of snow grains in air or ice with air inclusions, but all use geometric
        optics approximation (grain/bubble larger than the wavelength).
        """
        self.layer_mass = (np.array(self.density) 
                           * np.array(self.thickness_profile)
                           )

        for lyr in range(self.nbr_lyr):

            if self.layer_type[lyr] > 0:  # air inclusions in ice
                vlm_frac_ice = (self.density[lyr] - self.lwc[lyr] * 1000) / 917
                vlm_frac_air = 1 - self.lwc[lyr] - vlm_frac_ice
                eq_rds = (
                    3 * vlm_frac_air / (self.ssa[lyr] * self.density[lyr])
                )  # Eq from whicker
                sca_cff_vlm_air_bbl = np.ones(self.nbr_wvl) * 2 * 0.75 / (eq_rds)
                scattering_cff = sca_cff_vlm_air_bbl * vlm_frac_air / self.density[lyr]

                abs_cff = (
                    4
                    * np.pi
                    / (self.wavelengths)
                    / self.density[lyr]
                    * (
                        vlm_frac_ice * self.ref_idx_im
                        + self.lwc[lyr] * self.ref_idx_im_water
                    )
                )

                self.ext_cff[lyr, :] = scattering_cff + abs_cff

                self.ss_alb[lyr, :] = scattering_cff / self.ext_cff[lyr, :]

                # Kokhanovsky 2002
                self.asm_prm[lyr, :] = 0.49274 + 0.44466 / (
                    0.69233 * np.sqrt(np.pi / 2)
                ) * np.exp(-2 * ((1 / self.ref_idx_re - 1.04882) / 0.69233) ** 2)
                
                self.asm_prm = np.clip(self.asm_prm, 0, 1)

                self.tau[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]

            else:  # ice grains in air
                # under geometric optics assumptions, the extinction 
                # cross section is the extinction efficiency (=2) multiplied
                # by the cross section K. To get the mass extinction coeff 
                # in m2 kg-1, we then divide by the particle volume V and the 
                # ice density D, i.e. ext = 2 * (K / V) / D.
                # For convex grains, K = S / 4 with S the surface area of
                # the ice grain (Eq. 2.47 in Kokhanovsky 2001).
                # Since the is SSA = S / (V * D), then ext = 2 * SSA.
                
                self.ext_cff[lyr, :] = (
                    self.ssa[lyr] / 2
                ) 
                self.tau[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]
                
                k_eq = (
                    self.lwc[lyr] * self.ref_idx_im_water
                    + (1 - self.lwc[lyr]) * self.ref_idx_im
                )
                
                # cf Eq. 7, 8 in Kokhanovsky 2024
                # z = 4 * pi * k / wl * deff = 4 * pi * k / wl * 3 / 2 * V / K 
                # with V / K = 4 / (SSA * D)
                z = (4 * np.pi * k_eq / (self.wavelengths)
                     * 3 / 2 
                     * 4
                     / (self.ssa[lyr] * 917)
                     )

                if self.grain_shape[lyr] == 0:
                    
                    # Eq. 2.45 in Kokhanovsky 2001, Eq. 10 in Kokhanovsky 2024
                    eta = (0.3639 
                           + 1.676 * (self.ref_idx_re - 1) 
                           - 1.6284 * (self.ref_idx_re - 1)**2
                           )
                    ginf = 1.008 - 0.11 * (self.ref_idx_re - 1)
                    g0 = 1.006 - 0.3641 * (self.ref_idx_re - 1)
                    self.asm_prm[lyr, :] = ginf - (ginf - g0) * np.exp(-z * eta)
                    
                    # Table 5 from Kokhanovsky and Macke 1997
                    n_tab = [1.1, 1.2, 1.333, 1.4, 1.5, 1.6, 1.7]
                    b_tab = [1.11, 1.18, 1.24, 1.26, 1.29, 1.31, 1.33]
                    b = np.interp(self.ref_idx_re, n_tab, b_tab)

                elif self.grain_shape[lyr] == 1:
                    # Robledano 2023 measurements
                    self.asm_prm[lyr, :] = np.ones(self.nbr_wvl) * 0.815
                    b = self.ref_idx_re**2
                

                # Eq. 2.45 in Kokhanovsky 2001
                rho = 0.0123 + 0.1622 * (self.ref_idx_re - 1)
                # Eq. 6 in Kokhanovsky and Macke 1997
                phi = 2.0 / 3 * b / (1 - rho)
                # Eq. 7 in Kokhanovsky and Macke 1997 (ss_alb = (1-Cabs)/Cext)
                self.ss_alb[lyr, :] = 1 - 0.5 * (1 - rho) * (1 - np.exp(-z * phi))

    def update_column_ops_with_laps(self):
        """
        Update the optical properties of the snow/ice column to account for 
        the effct of light-absorbing particles.
        
        This method computes the combined optical properties of the snow/ice
        matrix and the embedded LAPs, following two-stream approximation mixing
        formulas. It adjusts optical thickness, single scattering albedo, and
        asymmetry parameters.
        """

        asm_prm_lap = np.zeros([self.nbr_lyr, self.nbr_wvl])
        ss_alb_lap = np.zeros_like(asm_prm_lap)
        tau_lap = np.zeros_like(asm_prm_lap)
        lap_mass = np.zeros_like(self.lap_concentrations)

        lap_mass = np.array(self.layer_mass)[:, np.newaxis] * self.lap_concentrations

        tau_lap = lap_mass @ self.lap_ext_cff

        ss_alb_lap = lap_mass @ (self.lap_ext_cff * self.lap_ss_alb)

        asm_prm_lap = lap_mass @ (self.lap_ext_cff * self.lap_ss_alb * self.lap_asm_prm)

        self.layer_mass = self.layer_mass - np.sum(lap_mass, axis=1)

        self.tau = self.layer_mass[:, np.newaxis] * self.ext_cff

        tau_clean = self.tau.copy()
        ss_alb_clean = self.ss_alb.copy()
        asm_prm_clean = self.asm_prm.copy()

        self.tau = tau_lap + tau_clean
        self.ss_alb = (1 / self.tau) * (ss_alb_lap + (ss_alb_clean * tau_clean))
        self.asm_prm = (1 / (self.tau * (self.ss_alb))) * (
            asm_prm_lap + (asm_prm_clean * ss_alb_clean * tau_clean)
        )
