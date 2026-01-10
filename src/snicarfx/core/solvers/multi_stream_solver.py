"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from dataclasses import dataclass
import numpy as np
from scipy.special import legendre

downward_loop = True


@dataclass
class _MultiStreamSolverResults:
    """
    Stores output data from radiative transfer calculations.

    This class holds computed radiative properties of the snow or ice column,
    such as albedo, integration angles and spectral reflectance.

    Attributes
    ----------
    albedo : array
        Spectrally resolved surface albedo [unitless].
    cos_angle : array
        Angle of the Gaussian integration  [unitless].
    cos_weight : array
        Weights of the Gaussian integration [unitless].
    directional_reflectance_top: array
        Spectral reflectance at the top of the atmosphere [].
    directional_radiance_top: array
        Spectral reflectance at the top of the atmosphere [Wm-2sr-1].

    """

    albedo: np.ndarray
    cos_angle: np.ndarray
    cos_weight: np.ndarray
    directional_reflectance_top: np.ndarray
    directional_radiance_top: np.ndarray


class _MultiStreamSolver:
    """
    This class initializes and calculates the variables necessary to solve the
    radiative transfer equation with a multi-stream solver. The solver itself is
    a combination of the the Advanced Matrix Operator Method (AMOM) and the
    adding method. It is a translation of the Fortran-based solver from CRTM,
    originally written by Quanhua Liu (QSS at JCSDA;
    quanhua.liu@noaa.gov), Yong Han (NOAA/NESDIS, yong.han@noaa.gov) and
    Paul van Delst (CIMMS/SSEC, paul.vandelst@noaa.gov).

    References:
    Liu and Weng, 2013: 10.1109/JSTARS.2013.2247026
    Liu and Weng, 2006: https://doi.org/10.1175/JAS3808.1

    """

    def __init__(self, land, atmosphere, irradiance):
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
        """

        self.mth_azi = 0
        self.solar_irradiance = irradiance.flx_slr
        self.solar_flag = True
        self.cos_sun = np.cos(np.deg2rad(np.rint(irradiance.sza)))
        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999
        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10
        self.cosmic_background = 0
        self.n_angles = 8
        self.nbr_wvl = len(irradiance.flx_slr.flatten())

        # apply delta scaling to land column
        # Delta truncation: get highest Legendre term following
        # Wicombe 1977 Eq. (15) -  2M = n_expansion + 1
        f = np.array(land.asm_prm ** (land.n_expansion + 1))

        # Wiscombe 1977 Eq. 20(a, b) + 14
        land.tau = (1.0 - land.ss_alb * f) * land.tau
        land.ss_alb = (1.0 - f) * land.ss_alb / (1 - land.ss_alb * f)

        if atmosphere.use_atmosphere:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.vstack([atmosphere.tau, land.tau])
            self.w = np.vstack([atmosphere.ss_alb, land.ss_alb])
            self.legendre_moments = np.hstack(
                [atmosphere.legendre_moments, land.legendre_moments]
            )

        else:
            self.nbr_lyr = land.nbr_lyr
            self.t_od = np.array(land.tau)
            self.w = np.array(land.ss_alb)
            self.legendre_moments = np.array(land.legendre_moments)

        # initialize arrays
        self.total_opt = np.zeros((self.nbr_lyr + 1, self.nbr_wvl))

        self.ff = np.zeros(
            (self.n_angles, self.n_angles + 1, self.nbr_lyr, self.nbr_wvl)
        )
        self.bb = np.zeros(
            (self.n_angles, self.n_angles + 1, self.nbr_lyr, self.nbr_wvl)
        )
        self.direct_reflectivity = np.zeros((self.n_angles, self.nbr_wvl))
        self.emissivity = np.zeros_like(self.direct_reflectivity)
        self.reflectivity = np.zeros((self.n_angles, self.n_angles, self.nbr_wvl))

        ## attributes for adding method
        self.s_level_refl_up = np.zeros(
            (self.n_angles, self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
        )
        
        
        self.s_level_rad_up = np.zeros((self.n_angles, self.nbr_lyr + 1, self.nbr_wvl))

        self.s_level_rad_down = np.zeros((self.n_angles, self.nbr_lyr + 1, self.nbr_wvl))

        self.s_layer_source_up = np.zeros((self.n_angles, self.nbr_lyr, self.nbr_wvl))
        
        self.s_layer_source_down = np.zeros((self.n_angles, self.nbr_lyr, self.nbr_wvl))
        
        self.s_layer_refl = np.zeros(
            (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr)
        )
        
        self.s_layer_trans = np.zeros(
            (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr)
        )


        if downward_loop: 
            
            self.s_level_refl_down = np.zeros(
                (self.n_angles, self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )
            self.s_level_refl_downt = np.zeros(
                (self.n_angles, self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )
            self.s_level_rad_upt = np.zeros((self.n_angles, self.nbr_lyr + 1, self.nbr_wvl))
            self.s_level_rad_downt = np.zeros((self.n_angles, self.nbr_lyr + 1, self.nbr_wvl))




        ######################################################################
        # SET GAUSSIAN QUADRATURE
        ######################################################################

        nodes, weights = np.polynomial.legendre.leggauss(self.n_angles * 2)
        self.cos_angle = nodes[self.n_angles :]  # only positive
        self.cos_weight = weights[self.n_angles :]

        ######################################################################
        # CALCULATE PHASE COEFFS & PHASE MATRICES
        ######################################################################

        # Calculate scaled expansion coefficients
        # Wiscombe 1977 Eq. 14
        # Convention is 0.5 * (2l+1) * Bl for the expansion
        orders = np.arange(0, land.n_expansion)

        phase_coeffs = (2 * orders[:, None, None] + 1) * 0.5 * (self.legendre_moments)

        # Calculate Legendre polynomials
        leg_poly = np.zeros((land.n_expansion, self.n_angles + 1))

        # for all but the last column
        for order in orders:
            leg_poly[order, :-1] = legendre(order)(self.cos_angle)

        # add SZA in the last column
        for order in orders:
            leg_poly[order, self.n_angles] = legendre(order)(self.cos_sun)

        legs = np.arange(self.mth_azi, land.n_expansion)
        ifac = (-1) ** (legs - self.mth_azi)

        # Calculate phase matrices
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

        if np.any(self.ff < -0.1) or np.any(self.bb < -0.1):
            raise ValueError("Negative phase matrix elements")

        self.ff[self.ff < 0] = 0
        self.bb[self.bb < 0] = 0

        return None

    def amom(self, k):
        """
        Compute layer transmission, reflection matrices and source
        function at the top and bottom of the layer using the advanced
        matrix operator method (AMOM; Liu and Weng 2013) set the attributes of
        the class accordingly.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the optical properties are calculated.

        """

        # equatino 6A L&W2013 (without the Kronecker delta)
        pp = (
            self.w[k, None, None, :]
            * self.ff[:, : self.n_angles, k, :]
            * self.cos_weight[None, :, None]
            / self.cos_angle[:, None, None]
        )
        # equation 6A + 7 L&W2013 (apply Kronecker delta to get alpha)
        n = np.arange(self.n_angles)
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

        # take the square roots !!!!! must be fixed
        eig_value = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)
        eig_value = np.where(eig_vals > 1e-12, np.sqrt(eig_vals), 1e-12)

        # scale eigenvectors by square roots of eigen values
        eig_value_diag = np.eye(self.n_angles)[None, :, :] * eig_value[:, None, :]
        eig_veva = np.matmul(eig_vecs, eig_value_diag)

        eig_vef = np.linalg.solve(np.moveaxis(pp - pm, -1, 0), eig_veva)

        # Compute layer reflection (Gp) and transmission (Gm) matrices
        gp = (eig_vecs + eig_vef) / 2.0
        gm = (eig_vecs - eig_vef) / 2.0

        exp_x = np.exp(-eig_value * self.t_od[k, :, None])

        a1 = gp * exp_x[:, None, :]
        a4 = gm * exp_x[:, None, :]

        a2 = np.linalg.solve(gm, a1)
        a3 = np.matmul(gp, a2)
        a5 = np.matmul(a1, a2)
        a6 = np.matmul(a4, a2)

        gm_a5 = gm - a5

        gm_a5_t = np.moveaxis(gm_a5, -1, 1)
        a4_m_a3_t = np.moveaxis(a4 - a3, -1, 1)
        gp_m_a6_t = np.moveaxis(gp - a6, -1, 1)

        trans = np.linalg.solve(gm_a5_t, a4_m_a3_t)

        refl = np.linalg.solve(gm_a5_t, gp_m_a6_t)

        trans_t = np.moveaxis(trans, -1, 1)
        refl_t = np.moveaxis(refl, -1, 1)
        
        # post processing
        self.s_layer_trans[:, :, :, k] = trans_t
        self.s_layer_refl[:, :, :, k] = refl_t

        # not included for now since we don't model thermal
        # if self.mth_azi == 0:
        #     print(trans.shape)
        #     thermal_c = trans[:, : self.nb_streams].sum(
        #         axis=1
        #     ) + refl[:, : self.nb_streams].sum(axis=1)

        #     if self.n_angles == (self.nb_streams + 1):
        #         thermal_c[self.n_angles - 1] += trans[
        #             self.n_angles - 1, self.n_angles - 1
        #         ]
        #     print(thermal_c.shape)
        #     self.s_layer_source_up[:, k, :] = (
        #         1.0 - thermal_c
        #     ) * self.planck_atmosphere[k]

        #     self.s_layer_source_down[:, k] = self.s_layer_source_up[:, k]

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

            n = np.arange(self.n_angles)
            v0[n, n, :] -= 1.0 + self.cos_angle[:, None] / self.cos_sun

            n = np.arange(self.n_angles, n2)
            v0[n, n, :] -= 1.0 - self.cos_angle[:, None] / self.cos_sun

            solar1 = np.linalg.solve(
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
            
            

    def get_outputs(self):
        """
        Compile and return radiative transfer results as an
        _MultiStreamSolverResults instance.

        Returns
        -------
        xr.Dataset
            Multi-stream solver results in an xarray Dataset.

        """

        results = _MultiStreamSolverResults(
            albedo=self.albedo,
            cos_angle=self.cos_angle,
            cos_weight=self.cos_weight,
            directional_reflectance_top=self.directional_reflectance_top,
            directional_radiance_top=self.directional_radiance_top,
        )

        return results


def solve_multi_stream_rt(land, atmosphere, irradiance):
    """

    This subroutine calculates hemispherical albedo by calling AMOM for each
    layer and combining them with the adding method in an upward pass from the
    bottom layer to the top layer.

    ! the downward pass is not yet implemented, so that net fluxes at each
    interface are not available.

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

    Returns
    -------
    albedo : array
        Hemispherical albedo integrated with gaussian quadrature over 16 angles.

    """

    aads = _MultiStreamSolver(land, atmosphere, irradiance)

    for k in range(1, aads.nbr_lyr + 1):
        aads.total_opt[k, :] = aads.total_opt[k - 1, :] + aads.t_od[k - 1, :]

    aads.s_level_refl_up[:, :, -1, :] = aads.reflectivity

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
            np.moveaxis(
                aads.s_level_refl_up[:, :, k + 1, :],
                source=[0, 1, 2],
                destination=[1, 2, 0],
            ),
            aads.s_layer_refl[:, :, :,  k],
        )

        n = np.arange(aads.n_angles)
        infinite_scattering[:, n, n] += 1

        inv_gamma_t = np.moveaxis(
            np.linalg.solve(
                np.moveaxis(infinite_scattering, 1, 2),
                np.moveaxis(aads.s_layer_trans[:, :, :,  k], 2, 1),
            ),
            1,
            2,
        )

        refl_down = np.matmul(
            np.moveaxis(
                aads.s_level_refl_up[:, :, k + 1, :],
                source=[0, 1, 2],
                destination=[1, 2, 0],
            ),
            np.moveaxis(
                aads.s_layer_source_down[:, k, :], source=[0, 1], destination=[1, 0]
            )[:, :, None],
        ).reshape(aads.nbr_wvl, aads.n_angles)

        aads.s_level_rad_up[:, k, :] = (
            aads.s_layer_source_up[:, k, :]
            + np.moveaxis(
                np.matmul(
                    inv_gamma_t,
                    (refl_down + np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0))[
                        :, :, None
                    ],
                ),
                source=[0, 1, 2],
                destination=[2, 0, 1],
            )[:, 0, :]
        )

        refl_trans = np.matmul(
            np.moveaxis(
                aads.s_level_refl_up[:, :, k + 1, :],
                source=[0, 1, 2],
                destination=[1, 2, 0],
            ),
            aads.s_layer_trans[:, :, :,  k],
        )

        aads.s_level_refl_up[:, :, k, :] = np.moveaxis(
            aads.s_layer_refl[:, :, :,  k] + np.matmul(inv_gamma_t, refl_trans),
            source=[0, 1, 2],  # wl, i, j
            destination=[2, 0, 1],
        )  # becomes i, j, wl

    if aads.mth_azi == 0:
        for i in range(len(aads.cos_angle)):
            aads.s_level_rad_up[i, 0, :] += (
                np.sum(aads.s_level_refl_up[i, :, 0, :]) * aads.cosmic_background
            )

    if downward_loop: 
        
        # preserve TOA upward radiance
        aads.s_level_rad_upt[:, 0, :] = aads.s_level_rad_up[:, 0, :].copy()
        
        if aads.mth_azi == 0:
            for i in range(len(aads.cos_angle)):
                aads.s_level_rad_down[i, 0, :] = (
                    aads.cosmic_background
                )
                aads.s_level_rad_downt[i, 0, :] = (
                    aads.cosmic_background
                )
                
        
        for k in range(0, aads.nbr_lyr):
            infinite_scattering = -np.matmul(
                np.moveaxis(
                    aads.s_level_refl_down[:, :, k, :],
                    source=[0, 1, 2],
                    destination=[1, 2, 0],
                ),
                aads.s_layer_refl[:, :, :, k],
            )

            n = np.arange(aads.n_angles)
            infinite_scattering[:, n, n] += 1
            
            
            inv_gamma_t = np.moveaxis(
                np.linalg.solve(
                    np.moveaxis(infinite_scattering, 1, 2),
                    np.moveaxis(aads.s_layer_trans[:, :, :,  k], 2, 1),
                ),
                1,
                2,
            )
            
            
            refl_down = np.matmul(
                np.moveaxis(
                    aads.s_level_refl_down[:, :, k, :],
                    source=[0, 1, 2],
                    destination=[1, 2, 0],
                ),
                np.moveaxis(
                    aads.s_layer_source_up[:, k, :], source=[0, 1], destination=[1, 0]
                )[:, :, None],
            ).reshape(aads.nbr_wvl, aads.n_angles)
            
            
        
            aads.s_level_rad_down[:, k + 1, :] = (
                aads.s_layer_source_down[:, k, :]
                + np.moveaxis(
                    np.matmul(
                        inv_gamma_t,
                        (refl_down + np.moveaxis(aads.s_level_rad_down[:, k, :], -1, 0))[
                            :, :, None
                        ],
                    ),
                    source=[0, 1, 2],
                    destination=[2, 0, 1],
                )[:, 0, :]
            )

            
            
            refl_trans = np.matmul(
                np.moveaxis(
                    aads.s_level_refl_down[:, :, k, :],
                    source=[0, 1, 2],
                    destination=[1, 2, 0],
                ),
                aads.s_layer_trans[:, :, :,  k],
            )
            
            

            aads.s_level_refl_down[:, :, k + 1, :] = np.moveaxis(
                aads.s_layer_refl[:, :, :,  k] + np.matmul(inv_gamma_t, refl_trans),
                source=[0, 1, 2],  # wl, i, j
                destination=[2, 0, 1], # becomes i, j, wl
            )  
            
            
            # finalize upward and downward radiances
            if np.max(np.abs(aads.s_level_refl_down[:, :, k + 1, :])) > 0:
                
                infinite_scattering = -np.matmul(
                    np.moveaxis(
                        aads.s_level_refl_down[:, :, k + 1, :],
                        source=[0, 1, 2],
                        destination=[1, 2, 0],
                    ),
                    np.moveaxis(
                        aads.s_level_refl_up[:, :, k + 1, :],
                        source=[0, 1, 2],
                        destination=[1, 2, 0],
                    ),
                )

                n = np.arange(aads.n_angles)
                infinite_scattering[:, n, n] += 1

                inv_gamma = np.linalg.inv(
                    infinite_scattering
                    )
                
                                
                # this does not appear in original code but we precompute
                # as in previous calculations
                refl_down = np.matmul(
                    np.moveaxis(
                        aads.s_level_refl_down[:, :, k + 1, :],
                        source=[0, 1, 2],
                        destination=[1, 2, 0],
                    ),
                    np.moveaxis(
                        aads.s_level_rad_up[:, k + 1, :], source=[0, 1], destination=[1, 0]
                    )[:, :, None],
                ).reshape(aads.nbr_wvl, aads.n_angles)
                
                
                aads.s_level_rad_downt[:, k + 1, :] = (
                    np.moveaxis(
                        np.matmul(
                            inv_gamma,
                            (refl_down + np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0))[
                                :, :, None
                            ],
                        ),
                        source=[0, 1, 2],
                        destination=[2, 0, 1],
                    )[:, 0, :]
                )
                
                temporal_vector = np.matmul(
                    inv_gamma,
                    (np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0))[
                        :, :, None
                    ]
                    )
         
                
                aads.s_level_rad_upt[:, k + 1, :] = (
                    np.moveaxis(
                    np.matmul(
                        np.moveaxis(
                            aads.s_level_refl_up[:, :, k + 1, :],
                            source=[0, 1, 2],
                            destination=[1, 2, 0],
                        ),
                        temporal_vector
                        )
                    +
                    np.matmul(
                        inv_gamma,
                        (np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0))[
                            :, :, None
                        ]
                        ),
                        source=[0, 1, 2],
                        destination=[2, 0, 1],
                    )[:, 0, :]
                )                     
                
            else:

                aads.s_level_rad_downt[:, k + 1, :] = aads.s_level_rad_down[:, k + 1, :]
                
                aads.s_level_rad_upt[:, k + 1, :] = (
                    np.matmul(
                        np.moveaxis(
                            aads.s_level_refl_up[:, :, k + 1, :],
                            source=[0, 1, 2],
                            destination=[1, 2, 0]),
                        np.moveaxis(aads.s_level_rad_down[:, k + 1, :],
                                    -1, 0)[:, :, None]
                    )
                    + np.moveaxis(aads.s_level_rad_up[:, k + 1, :],
                                -1, 0)[:, :, None]
                    )
            
            # print(aads.s_level_rad_downt[0, :, 0])
        aads.s_level_rad_down = aads.s_level_rad_downt.copy()
        aads.s_level_rad_up = aads.s_level_rad_upt.copy()
        

    aads.albedo = (
        2
        * np.pi
        * np.sum(
            aads.s_level_rad_up[:, 0, :]
            * np.array(aads.cos_angle)[:, None]
            * np.array(aads.cos_weight)[:, None],
            axis=0,
        )
        / (aads.solar_irradiance[None, :] * aads.cos_sun)  # project solar beam
    ).flatten()

    # directional reflectance at the top of the atmosphere
    aads.directional_reflectance_top = (aads.s_level_rad_up[:, 0, :] * np.pi) / (
        aads.solar_irradiance * aads.cos_sun
    )
    
    # directional radiance at the top of the atmosphere
    aads.directional_radiance_top = aads.s_level_rad_up[:, 0, :]
    
    # compute surface values
    if atmosphere.use_atmosphere:
    
        interface_idx = -land.nbr_lyr 
        
        tau_k = aads.total_opt[interface_idx, :]
        
        E_dir = (
            aads.solar_irradiance
            * aads.cos_sun
            * np.exp(-tau_k / aads.cos_sun)
        )
        
        E_diff = (
            2.0 * np.pi
            * np.sum(
                aads.s_level_rad_down[:, interface_idx, :]
                * np.array(aads.cos_angle)[:, None]
                * np.array(aads.cos_weight)[:, None],
                axis=0,
            )
        )
        
        aads.directional_reflectance_surface = (
            aads.s_level_rad_up[:, interface_idx, :] * np.pi 
            ) / (E_diff + E_dir)
        
        aads.albedo_surface = (
            2
            * np.pi
            * np.sum(
                aads.s_level_rad_up[:, interface_idx, :]
                * np.array(aads.cos_angle)[:, None]
                * np.array(aads.cos_weight)[:, None],
                axis=0,
            )
            / (E_diff + E_dir)  
        ).flatten()
            
    
    outputs = aads.get_outputs()

    return outputs
