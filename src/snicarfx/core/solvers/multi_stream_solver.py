"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
from scipy.special import legendre


class _MultiStreamSolver:
    """
    This class initializes and calculates the variables necessary to solve the
    runpolarized adiative transfer equation with a multi-stream solver, 
    assuming azimuthal symmetry. The solver itself is
    a combination of the the Advanced Matrix Operator Method (AMOM) and the
    adding method. It is a translation of the Fortran-based solver from CRTM,
    originally written by Quanhua Liu (QSS at JCSDA;
    quanhua.liu@noaa.gov), Yong Han (NOAA/NESDIS, yong.han@noaa.gov) and
    Paul van Delst (CIMMS/SSEC, paul.vandelst@noaa.gov).

    References:
    Liu and Weng, 2013: 10.1109/JSTARS.2013.2247026
    Liu and Weng, 2006: https://doi.org/10.1175/JAS3808.1

    """

    def __init__(self, land, atmosphere, irradiance, output_levels, n_streams):
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
        output_levels : string
            Level at which radiance fields mut be returned (TOA/BOA).
        n_streams: int
            Number of discrete ordinates / angles in the gaussian quadrature.
        """

        self.solar_irradiance = irradiance.flx_slr
        self.solar_flag = True
        self.cos_sun = np.cos(np.deg2rad(np.rint(irradiance.sza)))
        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999
        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10
        self.cosmic_background = 0
        self.n_angles = n_streams
        self.nbr_wvl = len(irradiance.flx_slr.flatten())
        self.output_levels = output_levels
        self.mth_azi = 0 # 0th Fourier moment = azimuthal symmetry

        # apply delta scaling (!) to land column only -> HG function (!)
        # Delta truncation: get highest Legendre term following
        # Wicombe 1977 Eq. (15) - 2M = n_expansion + 1
        f = np.array(land.asm_prm ** (land.n_expansion + 1))

        # Wiscombe 1977 Eq. 20(a, b) + 14
        land.tau = (1.0 - land.ss_alb * f) * land.tau
        land.ss_alb = (1.0 - f) * land.ss_alb / (1 - land.ss_alb * f)
        land.legendre_moments = (
            land.legendre_moments
            - f[None, :, :]
        ) / (1 - f[None, :, :])
        

        if atmosphere.use_atmosphere:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.vstack([atmosphere.tau, land.tau])
            self.w = np.vstack([atmosphere.ss_alb, land.ss_alb])
            self.legendre_moments = np.hstack(
                [atmosphere.legendre_moments, land.legendre_moments]
            )
            self.surface_idx = -land.nbr_lyr - 1

        else:
            self.nbr_lyr = land.nbr_lyr
            self.t_od = np.array(land.tau)
            self.w = np.array(land.ss_alb)
            self.legendre_moments = np.array(land.legendre_moments)
            self.surface_idx = 0

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

        if "BOA" in output_levels and atmosphere.use_atmosphere:

            self.s_level_refl_down = np.zeros(
                (self.nbr_wvl, self.n_angles, self.n_angles, self.nbr_lyr + 1)
            )
            self.s_level_rad_upt = np.zeros(
                (self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )
            self.s_level_rad_downt = np.zeros(
                (self.n_angles, self.nbr_lyr + 1, self.nbr_wvl)
            )

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
        # because we use Bl values that do not integrate orthogonality 
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
        # (!) this would need to be changed for m > 0
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
        
        # removed from now, but may need to bring them back
        # if np.any(self.ff < -0.1) or np.any(self.bb < -0.1):
        #     raise ValueError("Invalid phase matrix elements")

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
        eig_value = np.where(eig_vals > 0, np.sqrt(eig_vals), 0)

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

        # dictionnary with outputs depending on user inputs
        results = {}

        results["outgoing_angle"] = np.rad2deg(np.arccos(self.cos_angle))

        if "BOA" in self.output_levels:

            tau_k = self.total_opt[self.surface_idx, :]

            E_dir = self.solar_irradiance * self.cos_sun * np.exp(-tau_k / self.cos_sun)

            E_diff = (
                2.0
                * np.pi
                * np.sum(
                    self.s_level_rad_down[:, self.surface_idx, :]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
            )
            
            results["directional_reflectance_boa"] = (
                self.s_level_rad_up[:, self.surface_idx, :] * np.pi
            ) / (E_diff + E_dir)

            results["albedo_boa"] = (
                2
                * np.pi
                * np.sum(
                    self.s_level_rad_up[:, self.surface_idx, :]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
                / (E_diff + E_dir)
            ).flatten()
            
                
            
            # rad_down_all = self.s_level_rad_down[:, self.surface_idx, :].copy()
            # closest_sun_angle = np.argmin(np.abs(np.array(self.cos_angle) - self.cos_sun))
            # rad_down_all[closest_sun_angle, :] += self.solar_irradiance * np.exp(-tau_k / self.cos_sun) * self.cos_sun / (2 * np.pi *self.cos_angle[closest_sun_angle] * self.cos_weight[closest_sun_angle])
            
            # E_diff_trial = (
            #     2.0
            #     * np.pi
            #     * np.sum(
            #         rad_down_all
            #         * np.array(self.cos_angle)[:, None]
            #         * np.array(self.cos_weight)[:, None],
            #         axis=0,
            #     )
            # )

            # results["albedo_boa"] = (
            #     2
            #     * np.pi
            #     * np.sum(
            #         self.s_level_rad_up[:, self.surface_idx, :]
            #         * np.array(self.cos_angle)[:, None]
            #         * np.array(self.cos_weight)[:, None],
            #         axis=0,
            #     )
            #     / (E_diff_trial)
            # ).flatten()

        if "TOA" in self.output_levels:

            # directional radiance at the top of the atmosphere (TOA)
            results["directional_radiance_toa"] = self.s_level_rad_up[:, 0, :]

            # directional reflectance at the top of the atmosphere
            results["directional_reflectance_toa"] = (
                self.s_level_rad_up[:, 0, :] * np.pi
            ) / (self.solar_irradiance * self.cos_sun)

            results["albedo_toa"] = (
                2
                * np.pi
                * np.sum(
                    self.s_level_rad_up[:, 0, :]
                    * np.array(self.cos_angle)[:, None]
                    * np.array(self.cos_weight)[:, None],
                    axis=0,
                )
                / (self.solar_irradiance[None, :] * self.cos_sun)  # project solar beam
            ).flatten()

        return results

def verify_balance_of_fluxes(aads):
    """

    Verify that the energy coming in is either reflected back, absorbed by the 
    layers or 'lost' at the model boundary.

    Parameters
    ----------
    aads : MultiStreamSolver
        instance of the solver

    """
    
    # flux existing at top 
    flx_back_top = (2 * np.pi
               * np.sum(
                   aads.s_level_rad_up[:, 0, :]
                   * np.array(aads.cos_angle)[:, None]
                   * np.array(aads.cos_weight)[:, None],
                   axis=0)
               )
    
    # flux absorbed at the bottom model boundary: down-up
    flx_abs_bottom = ( 
                   # diffuse down
                   2 * np.pi * np.sum(
                   aads.s_level_rad_down[:, -1, :]
                   * np.array(aads.cos_angle)[:, None]
                   * np.array(aads.cos_weight)[:, None],
                   axis=0)
                   # direct down
                   + aads.solar_irradiance * aads.cos_sun 
                   * np.exp(-aads.total_opt[-1, :] / aads.cos_sun)
                   - 
                   # diffuse up
                   2 * np.pi * np.sum(
                       aads.s_level_rad_up[:, -1, :]
                       * np.array(aads.cos_angle)[:, None]
                       * np.array(aads.cos_weight)[:, None],
                       axis=0)
               )
               
        
    net_flux_layers = ( 
                       # diffuse down
                       2 * np.pi * np.sum(
                       aads.s_level_rad_down
                       * np.array(aads.cos_angle)[:, None, None]
                       * np.array(aads.cos_weight)[:, None, None],
                       axis=0)
                       # direct down
                       + aads.solar_irradiance * aads.cos_sun 
                       * np.exp(-aads.total_opt / aads.cos_sun)
                       - 
                       # diffuse up
                       2 * np.pi * np.sum(
                           aads.s_level_rad_up
                           * np.array(aads.cos_angle)[:, None, None]
                           * np.array(aads.cos_weight)[:, None, None],
                           axis=0)
                   )
    

    flx_abs_layers = - (net_flux_layers[1:, :] - net_flux_layers[:-1, :])
        
    
    flux_balance = (
            aads.cos_sun * aads.solar_irradiance
            - ( np.sum(flx_abs_layers, axis=0) # absorbed flux 
                + flx_abs_bottom # absorbed flux at bottom
                + flx_back_top # flux exiting at top
                )
        )

    
    if sum(flux_balance) > 1e-10:
            raise ValueError("Conservation of fluxes not verified")
    else:
        pass
    
    return None
    
def solve_multi_stream_rt(land, atmosphere, irradiance, output_levels, n_streams):
    """

    This subroutine calculates hemispherical albedo by calling AMOM for each
    layer and combining them with the adding method.

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
    output_levels : string
        Level at which radiance fields mut be returned (TOA/BOA).
    n_streams: int
        Number of discrete ordinates / angles in the gaussian quadrature.
    Returns
    -------
    outputs : dictionary

    """

    aads = _MultiStreamSolver(land, atmosphere, irradiance, output_levels, n_streams)

    if "BOA" in output_levels and atmosphere.use_atmosphere:
        run_downward_loop = True
    else:
        run_downward_loop = False

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

        n = np.arange(aads.n_angles)
        infinite_scattering[:, n, n] += 1

        inv_gamma_t = np.moveaxis(
            np.linalg.solve(
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
                    (refl_down + np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0))[
                        :, :, None
                    ],
                ),
                source=[0, 1, 2],
                destination=[2, 0, 1],
            )[:, 0, :]
        )
        

        refl_trans = np.matmul(
                aads.s_level_refl_up[:, :, :, k + 1],
                aads.s_layer_trans[:, :, :, k],
        )

        aads.s_level_refl_up[:, :, :, k] = (
            aads.s_layer_refl[:, :, :, k] 
            + np.matmul(inv_gamma_t, refl_trans)
            )

    if aads.mth_azi == 0:
        for i in range(len(aads.cos_angle)):
            aads.s_level_rad_up[i, 0, :] += (
                np.sum(aads.s_level_refl_up[i, :, :, 0]) * aads.cosmic_background
            )
        
        
    if run_downward_loop:

        # preserve TOA upward radiance
        aads.s_level_rad_upt[:, 0, :] = aads.s_level_rad_up[:, 0, :].copy()

        if aads.mth_azi == 0:
            for i in range(len(aads.cos_angle)):
                aads.s_level_rad_down[i, 0, :] = aads.cosmic_background
                aads.s_level_rad_downt[i, 0, :] = aads.cosmic_background

        for k in range(0, aads.nbr_lyr):
            infinite_scattering = -np.matmul(
                    aads.s_level_refl_down[:, :, :, k],
                aads.s_layer_refl[:, :, :, k],
            )

            n = np.arange(aads.n_angles)
            infinite_scattering[:, n, n] += 1

            inv_gamma_t = np.moveaxis(
                np.linalg.solve(
                    np.moveaxis(infinite_scattering, 1, 2),
                    np.moveaxis(aads.s_layer_trans[:, :, :, k], 2, 1),
                ),
                1,
                2,
            )

            refl_down = np.matmul(
                    aads.s_level_refl_down[:, :, :, k],
                np.moveaxis(
                    aads.s_layer_source_up[:, k, :], source=[0, 1], destination=[1, 0]
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

            aads.s_level_refl_down[:, :, :, k + 1] = (
                aads.s_layer_refl[:, :, :, k] 
                + np.matmul(inv_gamma_t, refl_trans)
                )

            # finalize upward and downward radiances
            if np.max(np.abs(aads.s_level_refl_down[:, :, :, k + 1])) > 0:

                infinite_scattering = -np.matmul(
                        aads.s_level_refl_down[:, :, :, k + 1],
                        aads.s_level_refl_up[:, :, :, k + 1],

                )

                n = np.arange(aads.n_angles)
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

                aads.s_level_rad_downt[:, k + 1, :] = aads.s_level_rad_down[:, k + 1, :]

                aads.s_level_rad_upt[:, k + 1, :] = (
                    np.matmul(
                            aads.s_level_refl_up[:, :, :, k + 1],

                        np.moveaxis(aads.s_level_rad_down[:, k + 1, :], -1, 0)[
                            :, :, None
                        ],
                    )
                    + np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0)[:, :, None]
                )
            
            
        aads.s_level_rad_down = aads.s_level_rad_downt.copy()
        aads.s_level_rad_up = aads.s_level_rad_upt.copy()
                    
    verify_balance_of_fluxes(aads)

    outputs = aads.get_outputs()

    return outputs

