import numpy as np
import xarray as xr


class ColumnProperties:
    """Snow or ice column physical & optical properties, including light
    absorbing particles.

    """

    def __init__(self, modelconfig):
        self.modelconfig = modelconfig
        self.thickness_profile = np.array(modelconfig.inputs["ICE"]["THICKNESS"])
        self.layer_type = modelconfig.inputs["ICE"]["LAYER_TYPE"]
        self.density = np.array(modelconfig.inputs["ICE"]["DENSITY"])
        self.rf_type = modelconfig.inputs["ICE"]["RF_TYPE"]
        self.grain_shape = modelconfig.inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = np.array(modelconfig.inputs["ICE"]["LwC"])
        self.ssa = np.array(modelconfig.inputs["ICE"]["SPECIFIC_SURFACE_AREA"])
        self.nbr_lyr = len(self.density)

        self.wavelengths = (
            np.arange(
                self.modelconfig.inputs["RTM"]["wVL_START"],
                self.modelconfig.inputs["RTM"]["wVL_END"],
                self.modelconfig.inputs["RTM"]["RESOLUTION"],
            )
            * 1e-9
        )

        self.nbr_wvl = len(self.wavelengths)
        self.sfc = np.ones(self.nbr_wvl) * modelconfig.inputs["ICE"]["SFC"]

        # init the ssps
        self.ext_cff = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.asm_prm = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.layer_mass = np.zeros(self.nbr_lyr)

        self.set_refractive_index_and_diffuse_fresnel_coeffs()
        self.calculate_column_ops_clean()

        if self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"] is not None:
            self.get_lap_properties()
            self.add_laps_to_column_ops()

    def set_refractive_index_and_diffuse_fresnel_coeffs(self):
        """Calculates ice refractive index and pre-calculated diffuse
        fresnel coefficients at user-defined resolution.

        Args:
            self

        """

        # set spectral resolution
        resolution = self.modelconfig.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.modelconfig.inputs["RTM"]["wVL_START"]
        wvl_end = self.modelconfig.inputs["RTM"]["wVL_END"]
        wvl_high_res = np.arange(200, 5001, 1)

        idx1 = np.where(wvl_high_res == wvl_start)[0][0]
        idx2 = np.where(wvl_high_res == wvl_end)[0][0]

        refidx_file = xr.open_dataset(
            self.modelconfig.op_path + "refractive_indices.nc"
        )
        fresnel_diffuse_file = xr.open_dataset(
            self.modelconfig.op_path + "fresnel_diffuse_coefficients.nc"
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

    def get_lap_properties(self):
        """Calculates light absorbing particle properties at user-defined
        resolution.


        """

        # set spectral resolution
        resolution = self.modelconfig.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.modelconfig.inputs["RTM"]["wVL_START"]
        wvl_end = self.modelconfig.inputs["RTM"]["wVL_END"]

        # initialize properties
        nb_laps = len(self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"])

        self.lap_ss_alb = np.zeros((nb_laps, self.nbr_wvl))
        self.lap_asm_prm = np.zeros((nb_laps, self.nbr_wvl))
        self.lap_ext_cff = np.zeros((nb_laps, self.nbr_wvl))

        # get concentrations
        self.lap_concentrations = np.vstack(
            [
                (
                    np.array(
                        # convert from ppb to kg kg-1
                        self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"][name][
                            "CONC"
                        ]
                    )
                    * 1e-9
                    if self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"][name][
                        "UNIT"
                    ]
                    == 0
                    else np.array(
                        # convert from cells mL-1 to kg kg-1 (1cell=1ng)
                        self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"][name][
                            "CONC"
                        ]
                    )
                    * 0.917
                    * 1e-9
                )
                for name in self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"]
            ]
        ).T

        # get properties in a large array
        for i, lap in enumerate(self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"]):

            # first get the ext coeff tag
            if self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"][lap]["COATED"]:
                ext_cff_tag = "ext_cff_mss_ncl"
            else:
                ext_cff_tag = "ext_cff_mss"

            # then interpolate the properties to the right resolution
            properties = xr.open_dataset(
                self.modelconfig.lap_path
                + self.modelconfig.inputs["LIGHT_AbSORbING_PARTICLES"][lap]["FILE"]
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

    def calculate_column_ops_clean(self):
        """Calculate optical properties of a clean snow/ice column"""

        self.layer_mass = self.density * self.thickness_profile

        for lyr in range(self.nbr_lyr):

            if self.layer_type[lyr] > 0:  # ice - only air inclusions for now
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

                # self.asm_prm[lyr, :] = np.ones(self.nbr_wvl) * 0.86
                # Kokhanovsky 2002
                self.asm_prm[lyr, :] = 0.49274 + 0.44466 / (
                    0.69233 * np.sqrt(np.pi / 2)
                ) * np.exp(-2 * ((1 / self.ref_idx_re - 1.04882) / 0.69233) ** 2)
                self.asm_prm = np.clip(self.asm_prm, 0, 1)

                self.tau[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]

            else:  # snow
                self.ext_cff[lyr, :] = (
                    self.density[lyr] * self.ssa[lyr] / 2
                ) / self.density[lyr]
                self.tau[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]
                w = 0.0611 + 0.17 * (self.ref_idx_re - 1.3)
                k_eq = (
                    self.lwc[lyr] * self.ref_idx_im_water
                    + (1 - self.lwc[lyr]) * self.ref_idx_im
                )
                c = 24.0 * np.pi * k_eq / (917.0 * self.wavelengths) / self.ssa[lyr]

                # change specific single scat albedo and g depending on shape
                if self.grain_shape[lyr] == 0:
                    b0 = 1.25
                    g0 = 0.895
                    b = b0 + 0.4 * (self.ref_idx_re - 1.3)
                    phi = 2.0 / 3 * b / (1 - w)
                    self.ss_alb[lyr, :] = 1 - 0.5 * (1 - w) * (1 - np.exp(-c * phi))
                    y = 0.728 + 0.752 * (self.ref_idx_re - 1.3)
                    ginf = 0.9751 - 0.105 * (self.ref_idx_re - 1.3)
                    g00 = g0 - 0.38 * (self.ref_idx_re - 1.3)
                    self.asm_prm[lyr, :] = ginf - (ginf - g00) * np.exp(-y * c)

                elif self.grain_shape[lyr] == 1:
                    self.asm_prm[lyr, :] = np.ones(self.nbr_wvl) * 0.815
                    b = self.ref_idx_re**2
                    phi = 2.0 / 3 * b / (1 - w)
                    self.ss_alb[lyr, :] = 1 - 0.5 * (1 - w) * (1 - np.exp(-c * phi))

    def add_laps_to_column_ops(self):
        """Calculate optical properties of a snow/ice column mixed with light
        absorbing particles.


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
