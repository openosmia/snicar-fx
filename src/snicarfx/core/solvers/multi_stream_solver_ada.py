"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
from scipy.special import factorial, lpmv, eval_legendre
from numpy.linalg import solve


class _MultiStreamSolverADA:
    """
    Compute and store the variables necessary to solve the
    unpolarized radiative transfer equation using the ADA solver, which employs
    the Advanced Matrix Operator Method (AMOM) and the adding method.

    The solver in this file is a translation of the Fortran-based solver from CRTM,
    originally written by Quanhua Liu (QSS at JCSDA;
    quanhua.liu@noaa.gov), Yong Han (NOAA/NESDIS, yong.han@noaa.gov) and
    Paul van Delst (CIMMS/SSEC, paul.vandelst@noaa.gov).

    References:
    Liu and Weng, 2013: 10.1109/JSTARS.2013.2247026
    Liu and Weng, 2006: https://doi.org/10.1175/JAS3808.1

    Attributes
    ----------
    solar_irradiance : ndarray
        Direct spectral solar irradiance.
    solar_flag : bool
        Controls whether to add a solar source.
    cos_sun : float
        Cosine of the solar zenith angle.
    cosmic_background : float
        Cosmic radiation background.
    n_fourier : int
        Number of Fourier modes used in the expansion of the intensity in azimuth.
    mth_azi : int
        Current Fourier mode.
    nbr_wvl : ndarray
        Number of wavelengths.
    nbr_lyr : int
        Number of layers for the total column.
    output_levels : str
        Levels in the column at which to return the results.
    relative_azimuths : ndarray
        Relative azimuth angle (viewing - solar).
    relative_azimuths_rad : ndarray
        Relative azimuth angle in radians (viewing - solar).
    cos_angle : ndarray
        Nodes of the gaussian quadrature.
    cos_weight : ndarray
        Weights of the gaussian quadrature.
    n_angles : int
        Number of angles used in the gaussian quadrature.
    _angle_indices : ndarray
        Indices of streams.
    run_downward_loop : bool
        Controls whether to compute radiative quantities at all levels or not.
    t_od : ndarray
        Spectral optical depth for all layers.
    w : ndarray
        Spectral single scattering albedo for all layers.
    legendre_moments : ndarray
        Legendre moments for all layers.
    surface_idx : int
        Index in the column at the interface atmosphere-land.
    total_opt : ndarray
        Cumulative spectral optical thickness.
    ff : ndarray
        Forward phase matrix.
    bb : ndarray
        Backward phase matrix.
    direct_reflectivity : ndarray
        Reflectivity to direct radiation of the downmost level (boundary).
    reflectivity : ndarray
        Reflectivity of the downmost level (boundary).
    s_layer_refl : ndarray
        Reflectivity of each layer.
    s_layer_trans : ndarray
        Transmittivity of each layer.
    s_layer_source_up : ndarray
        Upward source term for each layer.
    s_layer_source_down : ndarray
        Downward source term for each layer.
    s_level_rad_up : ndarray
        Upward diffuse radiance at each level.
    s_level_rad_upt : ndarray
        Temporary variable for upward diffuse radiance at each level.
    s_level_rad_down : ndarray
        Downward diffuse radiance at each level.
    s_level_rad_downt : ndarray
        Temporary variable for downward diffuse radiance at each level.
    s_level_rad_up_moments : ndarray
        Upward diffuse radiance at each level for each Fourier mode.
    s_level_rad_down_moments : ndarray
        Downward diffuse radiance at each level for each Fourier mode.
    s_level_refl_up : ndarray
        Upward reflectance at each level.
    s_level_refl_down : ndarray
        Downward reflectance at each level.
    """

    def __init__(
        self,
        land,
        atmosphere,
        irradiance,
        config,
    ):
        """
        Initialize all variables required for the solver and applies delta
        scaling to the single scattering properties of the ice/snow column.

        Parameters
        ----------
        land : LandColumn
            Instance of the LandColumn class, storing the physical
            and optical properties of the ice/snow column.
        atmosphere : AtmosphereColumn
            Instance of the AtmosphereColumn class, storing the physical
            and optical properties of the atmosphere column.
        irradiance : SolarIrradiance
            Instance of the SolarIrradiance class, storing the properties of the
            incoming solar irradiance.
        config : dictionary
            Solver parameters set in the input Yaml file.
        """

        self.solar_irradiance = np.array(irradiance.total_irradiance)
        self.solar_flag = True
        self.cos_sun = np.cos(np.deg2rad(np.rint(irradiance.sza)))
        self.cosmic_background = 0
        self.n_angles = config.SOLVER.N_STREAMS
        self.n_fourier = config.SOLVER.N_FOURIER_MODES

        self.wavelengths = config._wavelengths
        self.nbr_wvl = len(self.wavelengths)
        self.output_levels = config.SOLVER.OUTPUT_LEVELS
        self._angle_indices = np.arange(self.n_angles)

        if self.n_fourier > 1:
            self.azimuth_angles = np.arange(*config.SOLVER.AZIMUTH_ANGLES)
            self.relative_azimuths = np.abs(self.azimuth_angles - irradiance.saa)
            self.relative_azimuths_rad = np.deg2rad(self.relative_azimuths)

        if "BOA" in config.SOLVER.OUTPUT_LEVELS and atmosphere.use_atmosphere:
            self.run_downward_loop = True
        else:
            self.run_downward_loop = False

        # set quadrature

        self.set_gaussian_quadrature()

        # apply delta scaling

        if config.SOLVER.DELTA_SCALING == "M" or config.SOLVER.DELTA_SCALING == "M+":
            
            # Check from DISORT v.4.0.98
            if ((
                    (atmosphere.legendre_moments[atmosphere.n_expansion, :, :] < 1e-4).any()
                    or (land.legendre_moments[land.n_expansion, :, :] < 1e-4).any()
                    or (
                        atmosphere.legendre_moments[atmosphere.n_expansion + 1, :, :]
                        < 0.7 * atmosphere.legendre_moments[atmosphere.n_expansion, :, :]
                        ).any()
                    or (
                        land.legendre_moments[land.n_expansion + 1, :, :]
                        < 0.7 * land.legendre_moments[land.n_expansion, :, :]
                        ).any()
                    )
            and (config.SOLVER.DELTA_SCALING == "M+")):
                
                raise ValueError("Delta-M+ scaling cannot be applied to Legendre moments - select Delta-M instead.")
                
            (
                tau_land,
                ss_alb_land,
                legendre_moments_land,
                tau_atm,
                ss_alb_atm,
                legendre_moments_atm,
            ) = self.apply_delta_scaling(atmosphere, land, config.SOLVER)

        else:
            tau_land = np.array(land.tau)
            ss_alb_land = np.array(land.ss_alb)
            legendre_moments_land = np.array(
                land.legendre_moments[: land.n_expansion, :, :]
            )
            tau_atm = np.array(atmosphere.tau)
            ss_alb_atm = np.array(atmosphere.ss_alb)
            legendre_moments_atm = np.array(
                atmosphere.legendre_moments[: atmosphere.n_expansion, :, :]
            )

        # assemble full column

        if not atmosphere.use_atmosphere:
            self.nbr_lyr = land.nbr_lyr
            self.t_od = tau_land
            self.w = ss_alb_land
            self.legendre_moments = legendre_moments_land
            self.surface_idx = 0

        else:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.vstack([tau_atm, tau_land])
            self.w = np.vstack([ss_alb_atm, ss_alb_land])
            self.legendre_moments = np.hstack(
                [legendre_moments_atm, legendre_moments_land]
            )
            self.surface_idx = -land.nbr_lyr - 1

        # initialize arrays

        self.total_opt = np.zeros((self.nbr_lyr + 1, self.nbr_wvl))

        self.ff = np.zeros(
            (self.n_angles, self.n_angles + 1, self.nbr_lyr, self.nbr_wvl)
        )
        self.bb = np.zeros(
            (self.n_angles, self.n_angles + 1, self.nbr_lyr, self.nbr_wvl)
        )
        self.direct_reflectivity = np.zeros((self.n_angles, self.nbr_wvl))
        self.reflectivity = np.zeros((self.nbr_wvl, self.n_angles, self.n_angles))

        ## attributes for adding method
        self.s_level_refl_up = np.zeros(
            (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr + 1)
        )

        self.s_level_rad_up = np.zeros((self.n_angles, self.nbr_lyr + 1, self.nbr_wvl))

        self.s_level_rad_down = np.zeros(
            (self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
        )

        self.s_layer_source_up = np.zeros((self.n_angles, self.nbr_lyr, self.nbr_wvl))

        self.s_layer_source_down = np.zeros((self.n_angles, self.nbr_lyr, self.nbr_wvl))

        self.s_layer_refl = np.zeros(
            (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr)
        )

        self.s_layer_trans = np.zeros(
            (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr)
        )

        self.s_level_rad_up_moments = np.zeros(
            self.s_level_rad_up.shape + (self.n_fourier,)
        )
        self.s_level_rad_down_moments = np.zeros(
            self.s_level_rad_down.shape + (self.n_fourier,)
        )

        if self.run_downward_loop:

            self.s_level_refl_down = np.zeros(
                (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr + 1)
            )
            self.s_level_rad_upt = np.zeros(
                (self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )
            self.s_level_rad_downt = np.zeros(
                (self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )

    def set_gaussian_quadrature(self):
        """
        Set nodes and weights of gaussian quadrature in [0-1] (cos polar angle).

        """

        # generate nodes / weights in [-1:1] and then remap to [0-1]
        nodes, weights = np.polynomial.legendre.leggauss(self.n_angles)
        self.cos_angle = 0.5 * (nodes + 1.0)
        self.cos_weight = 0.5 * weights

    def apply_delta_scaling(self, atmosphere, land, SOLVER):
        """
        Apply optional Delta-M or Delta-M+ scaling to expansion coefficients
        to truncate strongly forward-scattering phase functions.

        Parameters
        ----------
        land : LandColumn
            Instance of the LandColumn class, storing the physical
            and optical properties of the ice/snow column.
        atmosphere : AtmosphereColumn
            Instance of the AtmosphereColumn class, storing the physical
            and optical properties of the atmosphere column.
        irradiance : SolarIrradiance
            Instance of the SolarIrradiance class, storing the properties of the
            incoming solar irradiance.
        SOLVER : dictionary
            Solver parameters set in the input Yaml file.

        """

        # initialize and set in case atmosphere is not used
        tau_atm = None
        ss_alb_atm = None
        legendre_moments_atm = None
        
        if SOLVER.DELTA_SCALING == "M":
            # Delta truncation: get highest Legendre term following
            # Wicombe 1977 Eq. (15) - 2M = N_MOMENTS
            f = np.array(land.legendre_moments[land.n_expansion])
            scale_factor = 1.0 - land.ss_alb * f
            legendre_moments_land = np.array(
                (land.legendre_moments[: land.n_expansion, :, :] - f[None, :, :])
                / (1 - f[None, :, :])
            )
            tau_land = np.array((1.0 - land.ss_alb * f) * land.tau)
            ss_alb_land = np.array((1.0 - f) * land.ss_alb / (1 - land.ss_alb * f))

            if atmosphere.use_atmosphere:
                # could be applied only from aerosol boundary down as no effect in rayleigh layers
                f = np.array(atmosphere.legendre_moments[atmosphere.n_expansion])
                legendre_moments_atm = np.array(
                    (
                        atmosphere.legendre_moments[: atmosphere.n_expansion, :, :]
                        - f[None, :, :]
                    )
                    / (1 - f[None, :, :])
                )
                tau_atm = np.array((1.0 - atmosphere.ss_alb * f) * atmosphere.tau)
                ss_alb_atm = np.array(
                    (1.0 - f) * atmosphere.ss_alb / (1 - atmosphere.ss_alb * f)
                )

                # update scale factor
                scale_factor = np.vstack([(1.0 - atmosphere.ss_alb * f), scale_factor])


        elif SOLVER.DELTA_SCALING == "M+":

            sigma_sq = ((land.n_expansion + 1) ** 2 - land.n_expansion**2) / (
                np.log((land.legendre_moments[land.n_expansion]) ** 2)
                - np.log((land.legendre_moments[land.n_expansion + 1]) ** 2)
            )
            f = np.array(
                land.legendre_moments[land.n_expansion]
                * np.exp(land.n_expansion**2 / (2 * sigma_sq))
            )
            scale_factor = 1.0 - land.ss_alb * f
            legendre_moments_land = np.array(
                (
                    land.legendre_moments[: land.n_expansion, :, :]
                    - f[None, :, :]
                    * np.exp(
                        -(np.arange(land.n_expansion) ** 2)[:, None, None]
                        / (2 * sigma_sq)
                    )
                )
                / (1 - f[None, :, :])
            )
            tau_land = np.array((1.0 - land.ss_alb * f) * land.tau)
            ss_alb_land = np.array((1.0 - f) * land.ss_alb / (1 - land.ss_alb * f))

            if atmosphere.use_atmosphere:

                sigma_sq = (
                    (atmosphere.n_expansion + 1) ** 2 - atmosphere.n_expansion**2
                ) / (
                    np.log(
                        (
                            atmosphere.legendre_moments[
                                atmosphere.n_expansion, :, :
                            ]
                        )
                        ** 2
                    )
                    - np.log(
                        (
                            atmosphere.legendre_moments[
                                atmosphere.n_expansion + 1,
                                :,
                                :,
                            ]
                        )
                        ** 2
                    )
                )
                f = atmosphere.legendre_moments[
                    atmosphere.n_expansion,
                    :,
                ] * np.exp(atmosphere.n_expansion**2 / (2 * sigma_sq))

                legendre_moments_scaled = np.array(
                    (
                        atmosphere.legendre_moments[
                            : atmosphere.n_expansion, :, :
                        ]
                        - f[None, :, :]
                        * np.exp(
                            -(np.arange(atmosphere.n_expansion) ** 2)[:, None, None]
                            / (2 * sigma_sq)
                        )
                    )
                    / (1 - f[None, :, :])
                )
                tau_scaled = np.array(
                    (1.0 - atmosphere.ss_alb[:, :] * f)
                    * atmosphere.tau[:, :]
                )
                ss_alb_scaled = np.array(
                    (1.0 - f)
                    * atmosphere.ss_alb[:, :]
                    / (1 - atmosphere.ss_alb[:, :] * f)
                )

                legendre_moments_atm = np.hstack(
                    [
                        atmosphere.legendre_moments[
                            : atmosphere.n_expansion, :, :
                        ],
                        legendre_moments_scaled,
                    ]
                )

                tau_atm = np.vstack(
                    [atmosphere.tau[:, :], tau_scaled]
                )
                ss_alb_atm = np.vstack(
                    [atmosphere.ss_alb[:, :], ss_alb_scaled]
                )

                scale_factor = np.vstack(
                    [
                        np.ones((atmosphere.nbr_lyr, self.nbr_wvl)),
                        (1.0 - atmosphere.ss_alb[:, :] * f),
                        scale_factor,
                    ]
                )
                
        return (
            tau_land,
            ss_alb_land,
            legendre_moments_land,
            tau_atm,
            ss_alb_atm,
            legendre_moments_atm,
            scale_factor,
        )
    

    def set_phase_matrices(self):
        """
        Calculate phase function expansion from legendre moments and legendre
        polynomials, then compute forward + backward phase matrices and finally
        check conservation of energy distributed among the streams.
        """

        orders = np.arange(self.mth_azi, self.legendre_moments.shape[0])

        if self.mth_azi == 0:
            phase_coeffs = (
                (2 * orders[:, None, None] + 1) * 0.5 * (self.legendre_moments)
            )

        elif self.mth_azi > 0:
            # add normalization factor
            norm = factorial(orders - self.mth_azi) / factorial(orders + self.mth_azi)
            phase_coeffs = (
                (2 * orders[:, None, None] + 1)
                * 0.5
                * (self.legendre_moments)[self.mth_azi :]
                * norm[:, None, None]
            )

        ####### Calculate Legendre polynomials/associated functions

        if self.mth_azi == 0:
            leg_poly = np.zeros((self.legendre_moments.shape[0], self.n_angles + 1))
            leg_poly[orders, :-1] = eval_legendre(
                orders[:, None], self.cos_angle[None, :]
            )
            leg_poly[orders, self.n_angles] = eval_legendre(orders, self.cos_sun)

        elif self.mth_azi > 0:
            leg_poly = np.zeros(
                (self.legendre_moments.shape[0] - self.mth_azi, self.n_angles + 1)
            )
            for i, order in enumerate(orders):
                leg_poly[i, :-1] = lpmv(self.mth_azi, order, self.cos_angle)
                leg_poly[i, self.n_angles] = lpmv(self.mth_azi, order, self.cos_sun)

        ####### Calculate phase matrices
        legs = np.arange(self.mth_azi, self.legendre_moments.shape[0])
        ifac = (-1) ** (legs - self.mth_azi)

        self.ff = np.sum(
            phase_coeffs[:, None, None, :, :]
            * leg_poly[:, :-1, None, None, None]  # -1 to exclude SZA
            * leg_poly[:, None, :, None, None],
            axis=0,
        )

        self.bb = np.sum(
            phase_coeffs[:, None, None, :, :]
            * leg_poly[:, :-1, None, None, None]  # -1 to exclude SZA
            * leg_poly[:, None, :, None, None]
            * ifac[:, None, None, None, None],
            axis=0,
        )

        energy_error = (
            np.einsum(
                "ijlk,j->ilk",
                self.ff[:, :-1, :, :] + self.bb[:, :-1, :, :],
                self.cos_weight,
            )
            - 1
        )

        if self.mth_azi == 0 and np.max(np.abs(energy_error)) > 1e-8:
            raise ValueError(
                "Error in stream energy conservation. Try increasing stream number or use aspherical shapes."
            )

        if np.any(self.ff < -0.1) or np.any(self.bb < -0.1):
            raise ValueError(
                "Invalid phase matrix elements. Try increasing stream numbers or use aspherical shapes."
            )

    def reset_state(self, m):
        """
        Change Fourier moment order and reset required variables.

        Parameters
        ----------
        m : int
            Fourier mode number.
        """

        # Fourier moment
        self.mth_azi = m

        # scalar state
        self.total_opt.fill(0.0)

        # phase matrices
        self.ff.fill(0.0)
        self.bb.fill(0.0)

        # radiances and reflectances
        self.s_level_rad_up.fill(0.0)
        self.s_level_rad_down.fill(0.0)
        self.s_level_refl_up.fill(0.0)
        self.s_layer_source_up.fill(0.0)
        self.s_layer_source_down.fill(0.0)
        self.s_layer_refl.fill(0.0)
        self.s_layer_trans.fill(0.0)

        self.direct_reflectivity.fill(0.0)
        self.reflectivity.fill(0.0)

        if self.run_downward_loop:
            self.s_level_refl_down.fill(0.0)
            self.s_level_rad_upt.fill(0.0)
            self.s_level_rad_downt.fill(0.0)

    def amom(self, k):
        """
        Compute layer transmission, reflection matrices and source
        function at the top and bottom of the layer using the advanced
        matrix operator method (AMOM; Liu and Weng 2013).

        Parameters
        ----------
        k : int
            Index of the layer for which the optical properties are calculated.

        """

        # equation 6A L&W2013 (without the Kronecker delta)
        pp = (
            self.w[k, None, None, :]
            * self.ff[:, : self.n_angles, k, :]
            * self.cos_weight[None, :, None]
            / self.cos_angle[:, None, None]
        )
        # equation 6A + 7 L&W2013 (apply Kronecker delta to get alpha)
        n = self._angle_indices
        pp[n, n, :] -= 1.0 / self.cos_angle[:, None]

        # equation 6B L&W2013
        pm = (
            self.w[k, None, None, :]
            * self.bb[:, : self.n_angles, k, :]
            * self.cos_weight[None, :, None]
            / self.cos_angle[:, None, None]
        )

        # equation 10 L&W2013 [matrix H = (alpha - beta) * (alpha + beta)]
        # moveaxis required as matmul uses the last two axes
        hh = np.matmul(np.moveaxis(pp - pm, -1, 0), np.moveaxis(pp + pm, -1, 0))

        # get eigen values & vectors
        # wavelength dimension at the front
        eig_vals, eig_vecs = np.linalg.eig(hh)

        # take the square roots
        eig_value = np.where(eig_vals > 0, np.sqrt(eig_vals), 0)

        # scale eigenvectors by square roots of eigen values
        eig_value_diag = np.eye(self.n_angles)[None, :, :] * eig_value[:, None, :]
        eig_veva = np.matmul(eig_vecs, eig_value_diag)

        eig_vef = solve(np.moveaxis(pp - pm, -1, 0), eig_veva)

        # Compute layer reflection (Gp) and transmission (Gm) matrices
        gp = (eig_vecs + eig_vef) / 2.0
        gm = (eig_vecs - eig_vef) / 2.0

        exp_x = np.exp(-eig_value * self.t_od[k, :, None])

        a1 = gp * exp_x[:, None, :]
        a4 = gm * exp_x[:, None, :]

        a2 = solve(gm, a1)
        a3 = np.matmul(gp, a2)
        a5 = np.matmul(a1, a2)
        a6 = np.matmul(a4, a2)

        gm_a5 = gm - a5

        gm_a5_t = np.moveaxis(gm_a5, -1, 1)
        a4_m_a3_t = np.moveaxis(a4 - a3, -1, 1)
        gp_m_a6_t = np.moveaxis(gp - a6, -1, 1)

        trans = solve(gm_a5_t, a4_m_a3_t)

        refl = solve(gm_a5_t, gp_m_a6_t)

        trans_t = np.moveaxis(trans, -1, 1)
        refl_t = np.moveaxis(refl, -1, 1)

        # post processing
        self.s_layer_trans[:, :, :, k] = trans_t
        self.s_layer_refl[:, :, :, k] = refl_t

        # treatment of solar radiation
        if self.solar_flag:
            n2 = 2 * self.n_angles
            n2_1 = -1
            source_up = np.zeros((self.n_angles, self.nbr_wvl))
            source_down = np.zeros((self.n_angles, self.nbr_wvl))

            # solar source
            sfactor = self.w[k, :] * self.solar_irradiance / np.pi

            if self.mth_azi == 0:
                sfactor /= 2.0

            expfactor = np.exp(-self.t_od[k, :] / self.cos_sun)
            s_transmittance = np.exp(-self.total_opt[k, :] / self.cos_sun)

            solar = np.zeros((n2, self.nbr_wvl))
            v0 = np.zeros((n2, n2, self.nbr_wvl))

            solar[: self.n_angles, :] = (
                -self.bb[:, self.n_angles, k, :] * sfactor[None, :]
            )  # bb(i, nZ+1)

            solar[self.n_angles :, :] = (
                -self.ff[:, self.n_angles, k, :] * sfactor[None, :]
            )  # ff(i, nZ+1)

            v0[: self.n_angles, : self.n_angles, :] = (
                self.w[None, None, k, :]
                * self.ff[: self.n_angles, : self.n_angles, k, :]
                * self.cos_weight[None, :, None]
            )
            v0[self.n_angles :, : self.n_angles, :] = (
                self.w[None, None, k, :]
                * self.bb[: self.n_angles, : self.n_angles, k, :]
                * self.cos_weight[None, :, None]
            )
            v0[: self.n_angles, self.n_angles :, :] = v0[
                self.n_angles :, : self.n_angles, :
            ]
            v0[self.n_angles :, self.n_angles :, :] = v0[
                : self.n_angles, : self.n_angles, :
            ]

            n = self._angle_indices
            v0[n, n, :] -= 1.0 + self.cos_angle[:, None] / self.cos_sun

            n = np.arange(self.n_angles, n2)
            v0[n, n, :] -= 1.0 - self.cos_angle[:, None] / self.cos_sun

            solar1 = solve(
                np.moveaxis(v0[:n2_1, :n2_1, :], -1, 0),
                np.moveaxis(solar[:n2_1, None, :], -1, 0),
            )

            solar1 = np.moveaxis(
                np.concatenate([solar1, np.zeros((self.nbr_wvl, 1, 1))], axis=1),
                0,
                -1,
            )

            sfac2 = solar[n2 - 1, :] - np.sum(
                v0[n2 - 1, :n2_1, :] * solar1[:n2_1, 0, :], axis=0
            )

            source_up = solar1[: self.n_angles, :].copy()
            source_down = expfactor[None, :] * solar1[self.n_angles :, :].copy()

            source_up -= np.moveaxis(
                refl_t @ np.moveaxis(solar1[self.n_angles :, :], -1, 0)
                + trans_t
                @ (
                    expfactor[:, None, None]
                    * np.moveaxis(solar1[: self.n_angles, :], -1, 0)
                ),
                0,
                -1,
            )

            source_down -= np.moveaxis(
                trans_t @ np.moveaxis(solar1[self.n_angles :, :], -1, 0)
                + refl_t
                @ (
                    expfactor[
                        :,
                        None,
                        None,
                    ]
                    * np.moveaxis(solar1[: self.n_angles, :], -1, 0)
                ),
                0,
                -1,
            )

            # Specific treatment for downward source function
            mask = (abs(v0[n2 - 1, n2 - 1, :]) > 1e-4).reshape((1, v0.shape[-1]))

            if np.sum(mask) == self.nbr_wvl:
                source_down[self.n_angles - 1, :] += (
                    (
                        expfactor
                        - np.moveaxis(
                            trans_t[:, self.n_angles - 1, self.n_angles - 1], 0, -1
                        )
                    )
                    * sfac2
                    / v0[n2 - 1, n2 - 1, :]
                )
            elif np.sum(~mask) == self.nbr_wvl:
                source_down[self.n_angles - 1, :] += (
                    expfactor * sfac2 * self.t_od[k] / self.cos_angle[self.n_angles - 1]
                )
            else:
                source_down[self.n_angles - 1, mask] += (
                    (
                        expfactor[mask]
                        - np.moveaxis(
                            trans_t[mask, self.n_angles - 1, self.n_angles - 1], 0, -1
                        )
                    )
                    * sfac2[mask]
                    / v0[n2 - 1, n2 - 1, mask]
                )
                source_down[self.n_angles - 1, ~mask] += (
                    expfactor[~mask]
                    * sfac2[~mask]
                    * self.t_od[k, ~mask]
                    / self.cos_angle[self.n_angles - 1]
                )

            source_up *= s_transmittance
            source_down *= s_transmittance

            self.s_layer_source_up[:, k, :] += source_up[:, 0, :]
            self.s_layer_source_down[:, k, :] += source_down[:, 0, :]

    def verify_balance_of_fluxes(self):
        """
        Verify that the energy coming in is either reflected back, absorbed by
        the layers or 'lost' at the model boundary.
        """

        # flux existing at top
        flx_back_top = (
            2
            * np.pi
            * np.sum(
                self.s_level_rad_up[:, 0, :]
                * np.array(self.cos_angle)[:, None]
                * np.array(self.cos_weight)[:, None],
                axis=0,
            )
        )

        # flux absorbed at the bottom model boundary: down-up
        flx_abs_bottom = (
            # diffuse down
            2
            * np.pi
            * np.sum(
                self.s_level_rad_down[:, -1, :]
                * np.array(self.cos_angle)[:, None]
                * np.array(self.cos_weight)[:, None],
                axis=0,
            )
            # direct down
            + self.solar_irradiance
            * self.cos_sun
            * np.exp(-self.total_opt[-1, :] / self.cos_sun)
            -
            # diffuse up
            2
            * np.pi
            * np.sum(
                self.s_level_rad_up[:, -1, :]
                * np.array(self.cos_angle)[:, None]
                * np.array(self.cos_weight)[:, None],
                axis=0,
            )
        )

        net_flux_layers = (
            # diffuse down
            2
            * np.pi
            * np.sum(
                self.s_level_rad_down
                * np.array(self.cos_angle)[:, None, None]
                * np.array(self.cos_weight)[:, None, None],
                axis=0,
            )
            # direct down
            + self.solar_irradiance
            * self.cos_sun
            * np.exp(-self.total_opt / self.cos_sun)
            -
            # diffuse up
            2
            * np.pi
            * np.sum(
                self.s_level_rad_up
                * np.array(self.cos_angle)[:, None, None]
                * np.array(self.cos_weight)[:, None, None],
                axis=0,
            )
        )

        flx_abs_layers = -(net_flux_layers[1:, :] - net_flux_layers[:-1, :])

        flux_balance = self.cos_sun * self.solar_irradiance - (
            np.sum(flx_abs_layers, axis=0)  # absorbed flux
            + flx_abs_bottom  # absorbed flux at bottom
            + flx_back_top  # flux exiting at top
        )

        if sum(flux_balance) > 1e-10:
            raise ValueError("Conservation of fluxes not verified")
        else:
            pass

        return None

    def get_outputs(self):
        """
        Compile and return radiative transfer results at required output levels.

        Returns
        -------
        results : dictionary
            Multi-stream solver results.

        """

        # dictionnary with outputs depending on user inputs
        results = {}

        results["polar_angle"] = np.rad2deg(np.arccos(self.cos_angle))

        if self.n_fourier > 1:
            results["azimuth_angle"] = self.azimuth_angles

        if "BOA" in self.output_levels:

            # azimuth-averaged first : only 0-th moment matters

            tau_k = self.total_opt[self.surface_idx, :]

            E_dir = self.solar_irradiance * self.cos_sun * np.exp(-tau_k / self.cos_sun)

            E_diff = (
                2.0
                * np.pi
                * np.sum(
                    self.s_level_rad_down_moments[:, self.surface_idx, :, 0]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
            )

            results["albedo_boa"] = (
                2
                * np.pi
                * np.sum(
                    self.s_level_rad_up_moments[:, self.surface_idx, :, 0]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
                / (E_diff + E_dir)
            ).flatten()

            results["bba_boa"] = np.trapezoid(
                2
                * np.pi
                * np.sum(
                    self.s_level_rad_up_moments[:, self.surface_idx, :, 0]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                ),
                x=self.wavelengths,
            ) / np.trapezoid(E_diff + E_dir, x=self.wavelengths)

            results["directional_reflectance_boa_m0"] = (
                self.s_level_rad_up_moments[:, self.surface_idx, :, 0] * np.pi
            ) / (E_diff + E_dir)

            results["directional_radiance_boa_m0"] = self.s_level_rad_up_moments[
                :, self.surface_idx, :, 0
            ]

            # double-directional radiance with Fourier reconstruction
            if self.n_fourier > 1:

                s_level_rad_up_boa = np.sum(
                    (
                        self.s_level_rad_up_moments[:, self.surface_idx, :, :, None]
                        * np.cos(
                            np.arange(self.n_fourier)[None, None, :, None]
                            * self.relative_azimuths_rad[None, None, None, :]
                        )
                    ),
                    axis=-2,
                )

                s_level_refl_up_boa = (
                    s_level_rad_up_boa
                    * np.pi
                    / (E_diff[None, :, None] + E_dir[None, :, None])
                )

                # radiance as a func of phi & mu at the bottom of the atmosphere (BOA)
                results["directional_radiance_boa"] = s_level_rad_up_boa
                # reflectance as a func of phi & mu at the bottom of the atmosphere (BOA)
                results["directional_reflectance_boa"] = s_level_refl_up_boa

        if "TOA" in self.output_levels:

            # azimuth-independent first : only 0-th moment matters

            tau_k = self.total_opt[0, :]

            E_dir = self.solar_irradiance * self.cos_sun * np.exp(-tau_k / self.cos_sun)

            E_diff = (
                2.0
                * np.pi
                * np.sum(
                    self.s_level_rad_down_moments[:, 0, :, 0]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
            )

            results["albedo_toa"] = (
                2
                * np.pi
                * np.sum(
                    self.s_level_rad_up_moments[:, 0, :, 0]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
                / (E_diff + E_dir)
            ).flatten()

            results["directional_radiance_toa_m0"] = self.s_level_rad_up_moments[
                :, 0, :, 0
            ]

            results["directional_reflectance_toa_m0"] = (
                self.s_level_rad_up_moments[:, 0, :, 0] * np.pi
            ) / (E_diff + E_dir)

            # double-directional radiance with Fourier reconstruction
            if self.n_fourier > 1:

                s_level_rad_up_toa = np.sum(
                    (
                        self.s_level_rad_up_moments[:, 0, :, :, None]
                        * np.cos(
                            np.arange(self.n_fourier)[None, None, :, None]
                            * self.relative_azimuths_rad[None, None, None, :]
                        )
                    ),
                    axis=-2,
                )

                s_level_refl_up_toa = (
                    s_level_rad_up_toa
                    * np.pi
                    / (E_diff[None, :, None] + E_dir[None, :, None])
                )

                # radiance as a func of phi & mu at the top of the atmosphere (TOA)
                results["directional_radiance_toa"] = s_level_rad_up_toa

                # reflectance as a func of phi & mu at the top of the atmosphere (TOA)
                results["directional_reflectance_toa"] = s_level_refl_up_toa

        return results


def solve_multi_stream_rt_ada(land, atmosphere, irradiance, config):
    """

    Compute upward and downward radiances for a column of homogeneous layers
    by calling AMOM for each layer and combining them with the adding method.

    Parameters
    ----------
    land : LandColumn
        Instance of the LandColumn class, storing the physical
        and optical properties of the ice/snow column.
    atmosphere : AtmosphereColumn
        Instance of the AtmosphereColumn class, storing the physical
        and optical properties of the atmosphere column.
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class, storing the properties of the
        incoming solar irradiance.
    config : dictionary
        Solver parameters set in the input Yaml file.

    Returns
    -------
    outputs : dictionary
        Results of the solvers (radiance/reflectance/albedo at TOA/BOA).


    """

    # initialize solver
    aads = _MultiStreamSolverADA(land, atmosphere, irradiance, config)

    for m in range(aads.n_fourier):

        # reset variables
        aads.reset_state(m=m)

        # calculate phase matrices
        aads.set_phase_matrices()

        for k in range(1, aads.nbr_lyr + 1):
            aads.total_opt[k, :] = aads.total_opt[k - 1, :] + aads.t_od[k - 1, :]

        aads.s_level_refl_up[:, :, :, -1] = aads.reflectivity

        # if aads.mth_azi == 0:
        #     aads.s_level_rad_up[:, -1, :] = aads.emissivity * aads.planck_surface

        # adds a solar reflection term to the upward radiance at the
        # last layer for all viewing angles
        if aads.solar_flag:

            aads.s_level_rad_up[:, -1, :] += (
                aads.direct_reflectivity
                * aads.cos_sun
                * aads.solar_irradiance[None, :]
                / np.pi
                * np.exp(-aads.total_opt[-1, :][None, :] / aads.cos_sun)
            )

        for k in range(aads.nbr_lyr - 1, -1, -1):

            # call AMOM algorithm to compute layer
            # transmission, reflection, and source functions.
            aads.amom(k)

            # Adding method to add the layer to the present level
            # to compute upward radiances and reflection matrix
            # at the new level.

            infinite_scattering = -np.matmul(
                aads.s_level_refl_up[:, :, :, k + 1],
                aads.s_layer_refl[:, :, :, k],
            )

            n = aads._angle_indices
            infinite_scattering[:, n, n] += 1

            inv_gamma_t = np.moveaxis(
                solve(
                    np.moveaxis(infinite_scattering, 1, 2),
                    np.moveaxis(aads.s_layer_trans[:, :, :, k], 2, 1),
                ),
                1,
                2,
            )

            refl_down = np.matmul(
                aads.s_level_refl_up[:, :, :, k + 1],
                np.moveaxis(
                    aads.s_layer_source_down[:, k, :], source=[0, 1], destination=[1, 0]
                )[:, :, None],
            ).reshape(aads.nbr_wvl, aads.n_angles)

            aads.s_level_rad_up[:, k, :] = (
                aads.s_layer_source_up[:, k, :]
                + np.moveaxis(
                    np.matmul(
                        inv_gamma_t,
                        (
                            refl_down
                            + np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0)
                        )[:, :, None],
                    ),
                    source=[0, 1, 2],
                    destination=[2, 0, 1],
                )[:, 0, :]
            )

            refl_trans = np.matmul(
                aads.s_level_refl_up[:, :, :, k + 1],
                aads.s_layer_trans[:, :, :, k],
            )

            aads.s_level_refl_up[:, :, :, k] = aads.s_layer_refl[
                :, :, :, k
            ] + np.matmul(inv_gamma_t, refl_trans)

        if aads.mth_azi == 0:
            for i in range(len(aads.cos_angle)):
                aads.s_level_rad_up[i, 0, :] += (
                    np.sum(aads.s_level_refl_up[:, i, :, 0]) * aads.cosmic_background
                )

        if aads.run_downward_loop:

            # preserve TOA upward radiance
            aads.s_level_rad_upt[:, 0, :] = aads.s_level_rad_up[:, 0, :].copy()

            if aads.mth_azi == 0:
                for i in range(len(aads.cos_angle)):
                    aads.s_level_rad_down[i, 0, :] = aads.cosmic_background
                    aads.s_level_rad_downt[i, 0, :] = aads.cosmic_background

            for k in range(aads.nbr_lyr):
                infinite_scattering = -np.matmul(
                    aads.s_level_refl_down[:, :, :, k],
                    aads.s_layer_refl[:, :, :, k],
                )

                n = aads._angle_indices
                infinite_scattering[:, n, n] += 1

                inv_gamma_t = np.moveaxis(
                    solve(
                        np.moveaxis(infinite_scattering, 1, 2),
                        np.moveaxis(aads.s_layer_trans[:, :, :, k], 2, 1),
                    ),
                    1,
                    2,
                )

                refl_down = np.matmul(
                    aads.s_level_refl_down[:, :, :, k],
                    np.moveaxis(
                        aads.s_layer_source_up[:, k, :],
                        source=[0, 1],
                        destination=[1, 0],
                    )[:, :, None],
                ).reshape(aads.nbr_wvl, aads.n_angles)

                aads.s_level_rad_down[:, k + 1, :] = (
                    aads.s_layer_source_down[:, k, :]
                    + np.moveaxis(
                        np.matmul(
                            inv_gamma_t,
                            (
                                refl_down
                                + np.moveaxis(aads.s_level_rad_down[:, k, :], -1, 0)
                            )[:, :, None],
                        ),
                        source=[0, 1, 2],
                        destination=[2, 0, 1],
                    )[:, 0, :]
                )

                refl_trans = np.matmul(
                    aads.s_level_refl_down[:, :, :, k],
                    aads.s_layer_trans[:, :, :, k],
                )

                aads.s_level_refl_down[:, :, :, k + 1] = aads.s_layer_refl[
                    :, :, :, k
                ] + np.matmul(inv_gamma_t, refl_trans)

                # finalize upward and downward radiances
                if np.max(np.abs(aads.s_level_refl_down[:, :, :, k + 1])) > 0:

                    infinite_scattering = -np.matmul(
                        aads.s_level_refl_down[:, :, :, k + 1],
                        aads.s_level_refl_up[:, :, :, k + 1],
                    )

                    n = aads._angle_indices
                    infinite_scattering[:, n, n] += 1

                    inv_gamma = np.linalg.inv(infinite_scattering)

                    # this does not appear in original code but we precompute
                    # as in previous calculations
                    refl_down = np.matmul(
                        aads.s_level_refl_down[:, :, :, k + 1],
                        np.moveaxis(
                            aads.s_level_rad_up[:, k + 1, :],
                            source=[0, 1],
                            destination=[1, 0],
                        )[:, :, None],
                    ).reshape(aads.nbr_wvl, aads.n_angles)

                    aads.s_level_rad_downt[:, k + 1, :] = np.moveaxis(
                        np.matmul(
                            inv_gamma,
                            (
                                refl_down
                                + np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0)
                            )[:, :, None],
                        ),
                        source=[0, 1, 2],
                        destination=[2, 0, 1],
                    )[:, 0, :]

                    temporal_vector = np.matmul(
                        inv_gamma,
                        (np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0))[
                            :, :, None
                        ],
                    )

                    aads.s_level_rad_upt[:, k + 1, :] = np.moveaxis(
                        np.matmul(
                            aads.s_level_refl_up[:, :, :, k + 1],
                            temporal_vector,
                        )
                        + np.matmul(
                            inv_gamma,
                            (np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0))[
                                :, :, None
                            ],
                        ),
                        source=[0, 1, 2],
                        destination=[2, 0, 1],
                    )[:, 0, :]

                else:

                    aads.s_level_rad_downt[:, k + 1, :] = aads.s_level_rad_down[
                        :, k + 1, :
                    ]

                    aads.s_level_rad_upt[:, k + 1, :] = (
                        np.matmul(
                            aads.s_level_refl_up[:, :, :, k + 1],
                            np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0)[
                                :, :, None
                            ],
                        )
                        + np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0)[
                            :, :, None
                        ]
                    )

            aads.s_level_rad_down = aads.s_level_rad_downt.copy()
            aads.s_level_rad_up = aads.s_level_rad_upt.copy()

        aads.s_level_rad_down_moments[:, :, :, aads.mth_azi] = (
            aads.s_level_rad_down.copy()
        )
        aads.s_level_rad_up_moments[:, :, :, aads.mth_azi] = aads.s_level_rad_up.copy()

        if aads.mth_azi == 0:
            aads.verify_balance_of_fluxes()

    outputs = aads.get_outputs()
    return outputs
