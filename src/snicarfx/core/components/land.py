"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


class LandColumn:
    """
    Physical and optical properties of a snow or ice column.

    This class computes and stores the properties of a snow/ice column for each
    layer based on the YAML input file.

    Attributes
    ----------
    config : Config
        An instance of the Config class containing model input data parsed
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
    ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each layer [unitless].
    asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each layer [unitless].
    tau : ndarray
        Wavelength-dependent optical thickness of each layer [unitless].
    layer_mass : ndarray
        Mass per unit area of each layer [kg/m2].
    laps: dict
        Light absorbing particles included in the model configuration.
    lap_concentrations : ndarray
        Mass concentrations of LAPs per layer [kg/kg].
    lap_ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each LAP [unitless].
    lap_asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each LAP [unitless].
    lap_ext_cff : ndarray
        Wavelength-dependent mass extinction coefficient of each LAP [m2/kg].
    n_expansion : int
        Order of the Legendre expansion of the phase function.
    legendre_moments: ndarray
        Moments of the Legendre expansion of the Henyey-Greenstein phase function

    """

    def __init__(self, config, PACKAGE_ROOT):

        # set module root path for data loading
        self.PACKAGE_ROOT = PACKAGE_ROOT

        self.layer_type = config.LAND.LAYER_TYPE
        self.nbr_lyr = len(self.layer_type)
        self.thickness_profile = config.LAND.THICKNESS
        self.density = config.LAND.DENSITY
        self.rf_type = config.LAND.RF_TYPE
        self.grain_shape = config.LAND.GRAIN_SHAPE
        self.lwc = config.LAND.LWC
        self.ssa = config.LAND.SPECIFIC_SURFACE_AREA

        self.wavelengths = (
            np.arange(
                config.SOLVER.WVL_START,
                config.SOLVER.WVL_END,
                config.SOLVER.RESOLUTION,
            )
            * 1e-9
        )

        self.nbr_wvl = len(self.wavelengths)
        self.sfc = np.ones(self.nbr_wvl) * config.LAND.SFC

        # ssps
        self.ss_alb = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.ext_cff = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.asm_prm = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.legendre_moments = np.zeros((self.n_expansion, self.nbr_lyr, self.nbr_wvl))

        self.set_refractive_index_and_diffuse_fresnel_coeffs()
        self.set_column_ops_without_laps()

        if config.LAND.LIGHT_ABSORBING_PARTICLES is not None:
            self.laps = config.LAND.LIGHT_ABSORBING_PARTICLES.root
            self.set_lap_properties()
            self.update_column_ops_with_laps()

    def set_refractive_index_and_diffuse_fresnel_coeffs(self):
        """
        Load and set refractive indices and diffuse Fresnel coefficients.

        This method loads high-resolution ice/water refractive index data as
        well as diffuse Fresnel reflection coefficients, and interpolates them
        to the model's spectral resolution.
        """

        refidx_file = xr.open_dataset(
            f"{self.PACKAGE_ROOT}/data/refractive_indices.nc"
        ).sel(wvl=self.wavelengths)
        fresnel_diffuse_file = xr.open_dataset(
            f"{self.PACKAGE_ROOT}/data/fresnel_diffuse_coefficients.nc"
        ).sel(wvl=self.wavelengths)

        self.ref_idx_re = refidx_file[str("re_" + self.rf_type)].values
        self.ref_idx_im = refidx_file[str("im_" + self.rf_type)].values
        self.ref_idx_im_water = refidx_file["im_Row20"].values

        self.fl_r_dif_a = fresnel_diffuse_file[
            str("R_dif_fa_ice_" + self.rf_type)
        ].values
        self.fl_r_dif_b = fresnel_diffuse_file[
            str("R_dif_fb_ice_" + self.rf_type)
        ].values

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

        self.layer_mass = np.array(self.density) * np.array(self.thickness_profile)

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

                self.ss_alb[lyr, :] = scattering_cff / (scattering_cff + abs_cff)

                # Kokhanovsky 2002
                self.asm_prm[lyr, :] = 0.49274 + 0.44466 / (
                    0.69233 * np.sqrt(np.pi / 2)
                ) * np.exp(-2 * ((1 / self.ref_idx_re - 1.04882) / 0.69233) ** 2)

                self.asm_prm = np.clip(self.asm_prm, 0, 1)

                self.ext_cff[lyr, :] = scattering_cff + abs_cff

                self.tau[lyr, :] = self.ext_cff[lyr, :] * self.layer_mass[lyr]

            else:  # ice grains in air
                # under geometric optics assumptions, the extinction
                # cross section is the extinction efficiency (=2) multiplied
                # by the cross section K. To get the mass extinction coeff
                # in m2 kg-1, we then divide by the particle volume V and the
                # ice density D, i.e. ext = 2 * (K / V) / D.
                # For convex grains, K = S / 4 with S the surface area of
                # the ice grain (Eq. 2.47 in Kokhanovsky 2001).
                # Since the is SSA = S / (V * D), then ext = 2 * SSA.

                self.ext_cff[lyr, :] = self.ssa[lyr] / 2
                self.tau[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]

                k_eq = (
                    self.lwc[lyr] * self.ref_idx_im_water
                    + (1 - self.lwc[lyr]) * self.ref_idx_im
                )

                # cf Eq. 7, 8 in Kokhanovsky 2024
                # z = 4 * pi * k / wl * deff = 4 * pi * k / wl * 3 / 2 * V / K
                # with V / K = 4 / (SSA * D)
                z = (
                    4
                    * np.pi
                    * k_eq
                    / (self.wavelengths)
                    * 3
                    / 2
                    * 4
                    / (self.ssa[lyr] * 917)
                )

                if self.grain_shape[lyr] == 0:

                    # Eq. 2.45 in Kokhanovsky 2001, Eq. 10 in Kokhanovsky 2024
                    eta = (
                        0.3639
                        + 1.676 * (self.ref_idx_re - 1)
                        - 1.6284 * (self.ref_idx_re - 1) ** 2
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

            # Delta truncation: Legendre term of HG function at 2M = 16
            # Wicombe 1977 Eq. 15
            f = self.asm_prm ** (self.n_expansion + 1)

            # Calculate legendre moments
            # Wiscombe 1977 Eq. 14
            self.legendre_moments[:, lyr, :] = (
                self.asm_prm[None, lyr, :] ** np.arange(self.n_expansion)[:, None]
                - f[None, lyr, :]
            ) / (1 - f[None, lyr, :])

    def set_lap_properties(self):
        """
        Load and set optical properties of light-absorbing particles (LAPs).

        This method sets the properties of each LAP defined in the input
        configuration, converting their concentrations to consistent units,
        and interpolating their properties to the model's spectral grid.
        """

        self.lap_concentrations = (
            np.array([obj.CONC for obj in self.laps.values()]) * 1e-9
        ).T

        self.lap_ss_alb = np.stack(
            [
                xr.open_dataset(
                    f"{self.PACKAGE_ROOT}/data/light_absorbing_particles/" + cfg.FILE
                )
                .interp(wvl=self.wavelengths)["ss_alb"]
                .values
                for lap, cfg in self.laps.items()
            ],
            axis=0,
        )

        self.lap_asm_prm = np.stack(
            [
                xr.open_dataset(
                    f"{self.PACKAGE_ROOT}/data/light_absorbing_particles/" + cfg.FILE
                )
                .interp(wvl=self.wavelengths)["asm_prm"]
                .values
                for lap, cfg in self.laps.items()
            ],
            axis=0,
        )

        self.lap_ext_cff = np.stack(
            [
                xr.open_dataset(
                    f"{self.PACKAGE_ROOT}/data/light_absorbing_particles/" + cfg.FILE
                )
                .interp(wvl=self.wavelengths)["ext_cff_mss"]
                .values
                for lap, cfg in self.laps.items()
            ],
            axis=0,
        )

    def update_column_ops_with_laps(self):
        """
        Update the optical properties of the snow/ice column to account for
        the effct of light-absorbing particles.

        This method computes the combined optical properties of the snow/ice
        matrix and the embedded LAPs, following two-stream approximation mixing
        formulas. It adjusts optical thickness, single scattering albedo, and
        asymmetry parameters.
        """

        # combine properties of all LAPs
        lap_mass = np.array(self.layer_mass)[:, np.newaxis] * self.lap_concentrations

        tau_all_laps = lap_mass @ self.lap_ext_cff

        ss_alb_all_laps = lap_mass @ (self.lap_ext_cff * self.lap_ss_alb)

        asm_prm_all_laps = lap_mass @ (
            self.lap_ext_cff * self.lap_ss_alb * self.lap_asm_prm
        )

        # update layer mass in tau by removing lap mass
        # ext_cff_before_lap_correction = self.tau.copy() / self.layer_mass.copy()[:, None]

        self.layer_mass = self.layer_mass - np.sum(lap_mass, axis=1)
        self.tau = self.layer_mass[:, np.newaxis] * self.ext_cff

        # combine LAPs + snow/ice
        tau_clean = self.tau.copy()
        ss_alb_clean = self.ss_alb.copy()
        asm_prm_clean = self.asm_prm.copy()

        self.tau = tau_all_laps + tau_clean
        self.ss_alb = (1 / self.tau) * (ss_alb_all_laps + (ss_alb_clean * tau_clean))
        self.asm_prm = (1 / (self.tau * (self.ss_alb))) * (
            asm_prm_all_laps + (asm_prm_clean * ss_alb_clean * tau_clean)
        )
