#!/usr/bin/env python3
"""

Advanced Matrix Operator Method to calculate the transmission and 
reflection matrices of each layer, then Adding Method to combine them.


Quanhua Liu and Fuzhong Weng, 2006
https://doi.org/10.1175/JAS3808.1

Quanhua Liu and Fuzhong Weng, 2013

"""

import numpy as np
from scipy.special import legendre


class _AdvancedDoublingAddingSolver:

    def __init__(self, column, irradiance):
        """
        Initialize all variables required for the solver
        """

        self.mth_azi = 0
        self.planck_surface = 0
        self.solar_irradiance = np.ones_like(irradiance.flx_slr) * 2
        self.solar_flag = True
        self.cos_sun = irradiance.cos_sza
        self.delta_scaling = True
        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999
        self.planck_atmosphere = np.zeros(column.nbr_lyr + 1)
        self.SCATTERING_ALBEDO_tHRESHOLD = 1e-10
        self.cosmic_background = 0
        self.total_opt = np.zeros((column.nbr_lyr + 1, column.nbr_wvl))

        # initialize (not needed once loops vectorized in k)
        self.t_od = np.array(column.tau)
        self.w = np.array(column.ss_alb)
        self.g = np.array(column.asm_prm)

        m = 8  # cf Wiscombe 1977
        self.n_legendre = 50  # Legendre order of expansion = n+1 terms
        self.n_angles = 8

        self.ff = np.zeros(
            (self.n_angles, self.n_angles + 1, column.nbr_lyr, column.nbr_wvl)
        )
        self.bb = np.zeros(
            (self.n_angles, self.n_angles + 1, column.nbr_lyr, column.nbr_wvl)
        )
        self.direct_reflectivity = np.zeros((self.n_angles, column.nbr_wvl))
        self.emissivity = np.zeros_like(self.direct_reflectivity)
        self.reflectivity = np.zeros((self.n_angles, self.n_angles, column.nbr_wvl))
        
        
        ## attributes for adding method
        self.s_level_refl_up = np.zeros(
            (self.n_angles, self.n_angles, column.nbr_lyr + 1, column.nbr_wvl)
        )
        self.s_level_rad_up = np.zeros(
            (self.n_angles, column.nbr_lyr + 1, column.nbr_wvl)
        )

        self.s_layer_source_up = np.zeros(
            (self.n_angles, column.nbr_lyr, column.nbr_wvl)
        )
        self.s_layer_source_down = np.zeros(
            (self.n_angles, column.nbr_lyr, column.nbr_wvl)
        )
        # self.thermal_c = np.zeros((self.n_angles, 
        #                            column.nbr_lyr, 
        #                            column.nbr_wvl))

        ######################################################################
        # SET GAUSSIAN QUADRATURE
        ######################################################################

        nodes, weights = np.polynomial.legendre.leggauss(self.n_angles * 2)
        self.cos_angle = nodes[self.n_angles :]  # only positive
        self.cos_weight = weights[self.n_angles :]

        ######################################################################
        # CALCULATE PHASE COEFFS & PHASE MATRICES WITH DELTA SCALING
        ######################################################################

        if self.delta_scaling:
            orders = np.arange(0, 2 * m - 1)
            phase_coeffs = np.zeros((2 * m - 1, column.nbr_lyr, column.nbr_wvl))

            # Delta truncation: get highest Legendre term following
            # Wicombe 1977 Eq. (15)
            f = self.g ** (2 * m)  # original expansion coeff at 2M

            # Calculate scaled expansion coefficients
            # Wiscombe 1977 Eq. 14
            # Convention is 0.5 * (2l+1) * Bl for the expansion

            phase_coeffs = (
                (2 * orders[:, None, None] + 1)
                * 0.5
                * (
                    (self.g[None, :, :] ** orders[:, None, None] - f[None, :, :])
                    / (1 - f[None, :, :])
                )
            )

            # Wiscombe 1977 Eq. 20(a, b)
            self.t_od = (1.0 - self.w * f) * self.t_od
            self.w = (1.0 - f) * self.w / (1 - self.w * f)

            # Calculate Legendre polynomials
            leg_poly = np.zeros((2 * m - 1, self.n_angles + 1))

            # for all but the last column
            for order in orders:
                leg_poly[order, :-1] = legendre(order)(self.cos_angle)

            # add SZA in the last column
            for order in orders:
                leg_poly[order, self.n_angles] = legendre(order)(self.cos_sun)

            legs = np.arange(self.mth_azi, 2 * m - 1)
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
            
            


        ######################################################################
        # CALCULATE PHASE COEFFS & PHASE MATRICES WITHOUT DELTA SCALING
        ######################################################################
        else:
            orders = np.arange(self.n_legendre + 1)

            self.phase_coeffs = np.zeros((self.n_legendre + 1, column.nbr_lyr))

            # HG Legendre expansion coefficients
            leg_moments = np.arange(0, self.n_legendre + 1)
            self.phase_coeffs = (
                0.5
                * (2 * leg_moments[:, None] + 1)
                * self.g[None, :] ** leg_moments[:, None]
            )

            # Calculate Legendre polynomials (different indexing than with scaling)
            leg_poly = np.zeros((self.n_legendre + 1, self.n_angles + 1))

            # for all but the last column
            for order in orders:
                leg_poly[order, :-1] = legendre(order)(self.cos_angle)

            # add SZA in the last column
            for order in orders:
                leg_poly[order, self.n_angles] = legendre(order)(self.cos_sun)

            legs = np.arange(self.mth_azi, self.n_legendre + 1)
            ifac = (-1) ** (legs - self.mth_azi)

            # Calculate phase matrices
            self.ff = np.sum(
                self.phase_coeffs[:, None, None, :]
                * leg_poly[:, :-1, None, None]  # -1 to exclude SZA
                * leg_poly[:, None, :, None],
                axis=0,
            )

            self.bb = np.sum(
                self.phase_coeffs[:, None, None, :]
                * leg_poly[:, :-1, None, None]  # -1 to exclude SZA
                * leg_poly[:, None, :, None]
                * ifac[:, None, None, None],
                axis=0,
            )

            if np.any(self.ff < -0.1) or np.any(self.bb < -0.1):
                raise ValueError("Negative phase matrix elements")

            self.ff[self.ff < 0] = 0
            self.ff[self.bb < 0] = 0

        return None

    def amom(self, column, k):
        """
        Compute layer transmission, reflection matrices and source
        function at the top and bottom of the layer using the advanced
        matrix operator method (Liu and Weng 2013)

        """

        # EQUATION 6A L&W2013 (without the Kronecker delta)
        pp = (
            self.w[k, None, None, :]
            * self.ff[:, : self.n_angles, k, :]
            * self.cos_weight[None, :, None]
            / self.cos_angle[:, None, None]
        )
        # EQUATION 6A + 7 L&W2013 (apply Kronecker delta to get alpha)
        n = np.arange(self.n_angles)
        pp[n, n, :] -= 1.0 / self.cos_angle[:, None]

        # EQUATION 6B L&W2013
        pm = (
            self.w[k, None, None, :]
            * self.bb[:, : self.n_angles, k, :]
            * self.cos_weight[None, :, None]
            / self.cos_angle[:, None, None]
        )

        # EQUATION 10 L&W2013 [matrix H = (alpha - beta) * (alpha + beta)]
        # moveaxis required as matmul uses the last two axes
        hh = np.matmul(np.moveaxis(pp - pm, -1, 0), np.moveaxis(pp + pm, -1, 0))

        # get eigen values & vectors
        # wavelength dimension at the front
        eig_vals, eig_vecs = np.linalg.eig(hh)

        # why do we take the square roots here?
        eig_value = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)

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
        self.s_layer_trans = trans_t
        self.s_layer_refl = refl_t
        self.s_layer_source_up[:, :, :] = 0.0

        # not included for now since we don't model thermal
        # if self.mth_azi == 0:
        #     print(trans.shape)
        #     thermal_c = trans[:, : column.model_inputs.nb_streams].sum(
        #         axis=1
        #     ) + refl[:, : column.model_inputs.nb_streams].sum(axis=1)

        #     if self.n_angles == (column.model_inputs.nb_streams + 1):
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
            source_up = np.zeros((self.n_angles, column.nbr_wvl))
            source_down = np.zeros((self.n_angles, column.nbr_wvl))

            # solar source
            sfactor = self.w[k, :] * self.solar_irradiance / np.pi

            if self.mth_azi == 0:
                sfactor /= 2.0

            expfactor = np.exp(-self.t_od[k, :] / self.cos_sun)
            s_transmittance = np.exp(-self.total_opt[k, :] / self.cos_sun)
            
            

            solar = np.zeros((n2, column.nbr_wvl))
            v0 = np.zeros((n2, n2, column.nbr_wvl))

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
                np.concatenate([solar1, np.zeros((column.nbr_wvl, 1, 1))], axis=1),
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
                    * np.moveaxis(solar1[ : self.n_angles, :], -1, 0)
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
                    * np.moveaxis(solar1[ : self.n_angles, :], -1, 0)
                ),
                0,
                -1,
            )
            
            

            # Specific treatment for downward source function
            mask = (abs(v0[n2 - 1, n2 - 1, :]) > 1e-4).reshape((1, v0.shape[-1]))
            
            
            if np.sum(mask) == column.nbr_wvl:
                source_down[self.n_angles - 1, :] += (
                    (expfactor - 
                     np.moveaxis(
                         trans_t[:, self.n_angles - 1, self.n_angles - 1], 0, -1))
                    * sfac2
                    / v0[n2 - 1, n2 - 1, :]
                )
            elif np.sum(~mask) == column.nbr_wvl:
                source_down[self.n_angles - 1, :] += (
                        expfactor * sfac2 * self.t_od[k] 
                        / self.cos_angle[self.n_angles - 1]
                    )
            else: 
                source_down[self.n_angles - 1, mask] += (
                    (expfactor[mask] - 
                     np.moveaxis(
                         trans_t[mask, self.n_angles - 1, self.n_angles - 1], 0, -1))
                    * sfac2[mask]
                    / v0[n2 - 1, n2 - 1, mask]
                )
                source_down[self.n_angles - 1, ~mask] += (
                        expfactor[~mask] * sfac2[~mask] * self.t_od[k, ~mask] 
                        / self.cos_angle[self.n_angles - 1]
                    )
            
            source_up *= s_transmittance
            source_down *= s_transmittance

            self.s_layer_source_up[:, k, :] += source_up[:,0,:]
            self.s_layer_source_down[:, k, :] += source_down[:,0,:]
    
                
        return None


def solve_advanced_adding_doubling(column, irradiance):
    """

    This subroutine calculates hemispherical albedo by combining all snow/ice
    layers with the adding method
 
    """

    aads = _AdvancedDoublingAddingSolver(column, irradiance)

    for k in range(1, column.nbr_lyr + 1):
        aads.total_opt[k, :] = aads.total_opt[k - 1, :] + aads.t_od[k - 1, :]

    aads.s_level_refl_up[:, :, -1, :] = aads.reflectivity

    if aads.mth_azi == 0:
        aads.s_level_rad_up[:, -1, :] = aads.emissivity * aads.planck_surface

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

    for k in range(column.nbr_lyr - 1, -1, -1):

        # call AMOM algorithm to compute layer
        # transmission, reflection, and source functions.
        aads.amom(column, k)
        
        
        # Adding method to add the layer to the present level
        # to compute upward radiances and reflection matrix
        # at the new level.
    
        # infinite scattering
        infinite_scattering = -np.matmul(
            np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :],
                        source=[0, 1, 2], 
                        destination=[1, 2, 0]),
            aads.s_layer_refl,
        )

        n = np.arange(aads.n_angles)
        infinite_scattering[:, n, n] += 1


        inv_gamma_t = np.moveaxis(np.linalg.solve(
            np.moveaxis(infinite_scattering, 1, 2), 
            np.moveaxis(aads.s_layer_trans, 2, 1)
        ), 1, 2)
        
        
        refl_down = np.matmul(
            np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :],
                        source=[0, 1, 2], 
                        destination=[1, 2, 0]), 
            np.moveaxis(aads.s_layer_source_down[:, k, :],
                        source=[0, 1], 
                        destination=[1, 0])[:, :, None],
        ).reshape(column.nbr_wvl, aads.n_angles)
        

    
        
        aads.s_level_rad_up[:, k, :] = (
        aads.s_layer_source_up[:, k, :]
        + np.moveaxis(
            np.matmul(
            inv_gamma_t,
            (refl_down + 
        np.moveaxis(aads.s_level_rad_up[:, k + 1, :], -1, 0))[:, :, None],
        ), 
            source=[0, 1, 2], 
            destination=[2, 0, 1])[:,0,:]
        )
        
        
        refl_trans = np.matmul(
            np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :], 
                        source=[0, 1, 2], 
                        destination=[1, 2, 0]), 
            aads.s_layer_trans
        )
        
        aads.s_level_refl_up[:, :, k, :] = np.moveaxis(aads.s_layer_refl + np.matmul(
            inv_gamma_t, refl_trans
        ), source=[0, 1, 2], # wl, i, j
        destination=[2,  0, 1]) # becomes i, j, wl

        
    if aads.mth_azi == 0:
        for i in range(len(aads.cos_angle)):
            aads.s_level_rad_up[i, 0, :] += (
                np.sum(aads.s_level_refl_up[i, :, 0, :]) * aads.cosmic_background
            )
    
    

    albedo = (
        2
        * np.pi
        * np.sum(
            aads.s_level_rad_up[:, 0, :]
            * np.array(aads.cos_angle)[:, None]
            * np.array(aads.cos_weight)[:, None], axis=0
        )
        / (aads.solar_irradiance[None, :] * aads.cos_sun)
    ).flatten()

    return albedo
