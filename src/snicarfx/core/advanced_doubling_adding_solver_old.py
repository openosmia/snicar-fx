#!/usr/bin/env python3
"""

Implementation of the Advanced Doubling Adding Method for Radiative
Transfer in Planetary Atmospheres


Quanhua Liu and Fuzhong Weng, 2006
https://doi.org/10.1175/JAS3808.1

"""

import numpy as np
from scipy.special import legendre

## GENERAL COMMENTS:
# loops in k in __init__ should be vectorized
# maybe we can remove the solar flag? (ie always true)
# maybe we keep emission for now even if we don't use it (ie set at 0)


class _AdvancedDoublingAddingSolver:

    def __init__(self, column, irradiance, wvl):
        """
        Initialize all variables required for the ADA solver
        """

        self.wvl = wvl

        self.mth_azi = 0
        self.planck_surface = 0
        self.solar_irradiance = 2
        self.solar_flag = True
        self.cos_sun = irradiance.cos_sza
        self.delta_scaling = True
        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999
        self.planck_atmosphere = np.zeros(column.nbr_lyr + 1)
        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10
        self.cosmic_background = 0
        self.total_opt = np.zeros(column.nbr_lyr + 1)

        tau_unscaled = np.array(column.tau[:, self.wvl])
        w_unscaled = np.array(column.ss_alb[:, self.wvl])
        g_unscaled = np.array(column.asm_prm[:, self.wvl])

        # initialize (not needed once loops vectorized in k)
        self.t_od = tau_unscaled
        self.w = w_unscaled
        self.g = g_unscaled

        m = 8  # cf Wiscombe 1977
        n_legendre = 50  # Legendre order of expansion = n+1 terms
        n_angles = 8

        self.ff = np.zeros((n_angles, n_angles + 1, column.nbr_lyr))
        self.bb = np.zeros((n_angles, n_angles + 1, column.nbr_lyr))
        self.direct_reflectivity = np.zeros(n_angles)
        self.emissivity = np.zeros_like(self.direct_reflectivity)
        self.reflectivity = np.zeros((n_angles, n_angles))

        ## attributes for adding method
        self.refl_down = np.zeros((n_angles, column.nbr_lyr))
        self.temporal_matrix = np.zeros_like((n_angles, n_angles))
        self.inv_gamma = np.zeros((n_angles, n_angles, column.nbr_lyr))
        self.inv_gamma_t = np.zeros_like(self.inv_gamma)

        self.refl_trans = np.zeros_like(self.inv_gamma)

        self.s_layer_trans = np.zeros_like(self.inv_gamma)

        self.s_layer_refl = np.zeros_like(self.inv_gamma)

        self.s_level_refl_up = np.zeros((n_angles, n_angles, column.nbr_lyr + 1))
        self.s_level_rad_up = np.zeros((n_angles, column.nbr_lyr + 1))
        self.s_layer_source_up = np.zeros((n_angles, column.nbr_lyr))
        self.s_layer_source_down = np.zeros((n_angles, column.nbr_lyr))

        self.thermal_c = np.zeros((n_angles, column.nbr_lyr))

        ######################################################################
        # SET GAUSSIAN QUADRATURE
        ######################################################################

        nodes, weights = np.polynomial.legendre.leggauss(n_angles * 2)
        self.cos_angle = nodes[n_angles:]  # only positive
        self.cos_weight = weights[n_angles:]
        mu = np.array(self.cos_angle)

        ######################################################################
        # CALCULATE PHASE COEFFS & PHASE MATRICES WITH DELTA SCALING
        ######################################################################

        if self.delta_scaling:
            self.phase_coeffs = np.zeros((2 * m - 1, column.nbr_lyr))
            for k in range(column.nbr_lyr):
                g = g_unscaled[k]

                # Delta truncation: get highest Legendre term following
                # Wicombe 1977 Eq. (15)
                f = g ** (2 * m)  # original expansion coeff at 2M

                # Calculate scaled expansion coefficients
                for order in range(2 * m - 1):
                    # Wiscombe 1977 Eq. 14
                    self.phase_coeffs[order, k] = (g**order - f) / (1 - f)
                    # Convention is 0.5 * (2l+1) * Bl for the expansion
                    self.phase_coeffs[order, k] = (
                        (2 * order + 1) * 0.5 * self.phase_coeffs[order, k]
                    )

                # Wiscombe 1977 Eq. 20(a, b)
                self.t_od[k] = (1.0 - w_unscaled[k] * f) * tau_unscaled[k]
                self.w[k] = (1.0 - f) * w_unscaled[k] / (1 - w_unscaled[k] * f)

            # Calculate Legendre polynomials
            leg_poly = np.zeros((2 * m - 1, n_angles + 1))

            for i in range(n_angles):
                mu = self.cos_angle[i]
                for order in range(2 * m - 1):
                    leg_poly[order, i] = legendre(order)(mu)

            # add SZA as last column
            for order in range(2 * m - 1):
                leg_poly[order, n_angles] = legendre(order)(self.cos_sun)

            jn = n_angles + 1 if self.solar_flag else n_angles

            # Calculate phase matrices
            for k in range(column.nbr_lyr):
                for j in range(jn):  # incoming angle
                    for i in range(n_angles):  # outgoing angle
                        off = 0.0
                        obb = 0.0
                        for leg in range(self.mth_azi, 2 * m - 1):
                            ifac = (-1) ** (leg - self.mth_azi)
                            coeff = self.phase_coeffs[leg, k]
                            off += coeff * leg_poly[leg, i] * leg_poly[leg, j]
                            obb += coeff * leg_poly[leg, i] * leg_poly[leg, j] * ifac
                        self.ff[i, j, k] = off
                        self.bb[i, j, k] = obb

                        # if j == jn - 1:
                        #     print(self.ff[i, j, k], self.bb[i, j, k])

                        if self.ff[i, j, k] < 0:
                            if self.ff[i, j, k] < -0.1:
                                raise ValueError("Negative phase matrix elements")
                            else:
                                self.ff[i, j, k] = 0
                        if self.bb[i, j, k] < 0:
                            if self.ff[i, j, k] < -0.1:
                                raise ValueError("Negative phase matrix elements")
                            else:
                                self.bb[i, j, k] = 0

        ######################################################################
        # CALCULATE PHASE COEFFS & PHASE MATRICES WITHOUT DELTA SCALING
        ######################################################################
        else:

            self.phase_coeffs = np.zeros((n_legendre + 1, column.nbr_lyr))
            # HG Legendre expansion coefficients
            for k in range(column.nbr_lyr):
                for leg_moment in range(n_legendre + 1):
                    self.phase_coeffs[leg_moment, k] = (
                        0.5 * (2 * leg_moment + 1) * g**leg_moment
                    )

            leg_poly = np.zeros((n_legendre + 1, n_angles + 1))

            for i in range(n_angles):
                mu = self.cos_angle[i]
                for leg_moment in range(n_legendre + 1):
                    leg_poly[leg_moment, i] = legendre(leg_moment)(mu)

            # add SZA as last column
            for leg_moment in range(n_legendre + 1):
                leg_poly[leg_moment, n_angles] = legendre(leg_moment)(self.cos_sun)

            jn = n_angles + 1 if self.solar_flag else n_angles

            for k in range(column.nbr_lyr):
                for j in range(jn):  # incoming angle
                    for i in range(n_angles):  # outgoing angle
                        off = 0.0
                        obb = 0.0
                        for leg in range(self.mth_azi, n_legendre + 1):
                            ifac = (-1) ** (leg - self.mth_azi)
                            coeff = self.phase_coeffs[leg, k]
                            off += coeff * leg_poly[leg, i] * leg_poly[leg, j]
                            obb += coeff * leg_poly[leg, i] * leg_poly[leg, j] * ifac
                        self.ff[i, j, k] = off
                        self.bb[i, j, k] = obb
                        if (self.bb[i, j, k] < 0) or (self.ff[i, j, k] < 0):
                            print(self.bb[i, j, k])
                            raise ValueError("Negative phase matrix elements")

        return None

    def crtm_anom_layer(self, column, k):
        """Compute layer transmission, reflection matrices and source
        function at the top and bottom of the layer.

        Method and References The transmittance and reflectance
        matrices is further derived from matrix operator method. The
        matrix operator method is referred to the paper by

        Weng, F., and Q. Liu, 2003: Satellite Data Assimilation in
        Numerical Weather Prediction Model: Part 1: Forward Radiative
        Transfer and Jacobian Modeling in Cloudy Atmospheres,
        J. Atmos. Sci., 60, 2633-2646.

        see also ADA method.  Translated by the snicar-fx team from
        the Fortran code of Quanhua Liu Quanhua.Liu@noaa.gov

        """

        pp = np.zeros((len(self.cos_angle), len(self.cos_angle)))
        pm = np.zeros((len(self.cos_angle), len(self.cos_angle)))

        for i in range(len(self.cos_angle)):
            for j in range(len(self.cos_angle)):

                # EQUATION 6A L&W2013 (without the Kronecker delta)
                pp[i, j] = (
                    self.w[k]
                    * self.ff[i, j, k]
                    * self.cos_weight[j]
                    / self.cos_angle[i]
                )
                # EQUATION 6B L&W2013
                pm[i, j] = (
                    self.w[k]
                    * self.bb[i, j, k]
                    * self.cos_weight[j]
                    / self.cos_angle[i]
                )

            # EQUATION 6A + 7 L&W2013 (apply Kronecker delta to get alpha)
            pp[i, i] -= 1.0 / self.cos_angle[i]

        # EQUATION 10 L&W2013 [matrix H = (alpha - beta) * (alpha + beta)]
        hh = np.matmul(pp - pm, pp + pm)

        # get eigen values & vectors
        eig_vals, eig_vecs = np.linalg.eig(hh)

        # why do we take the square roots here?
        eig_value = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)

        # scale eigenvectors by square roots of eigen values
        eig_veva = eig_vecs @ np.diag(eig_value)

        eig_vef = np.linalg.solve(pp - pm, eig_veva)

        # Compute layer reflection (Gp) and transmission (Gm) matrices
        gp = (eig_vecs + eig_vef) / 2.0
        gm = (eig_vecs - eig_vef) / 2.0

        exp_x = np.exp(-eig_value * self.t_od[k])

        a1 = gp * exp_x[np.newaxis, :]
        a4 = gm * exp_x[np.newaxis, :]

        a2 = np.linalg.solve(gm, a1)
        a3 = np.matmul(gp, a2)
        a5 = np.matmul(a1, a2)
        a6 = np.matmul(a4, a2)

        gm_a5 = gm - a5

        trans = np.linalg.solve(gm_a5.T, (a4 - a3).T).T

        refl = np.linalg.solve(gm_a5.T, (gp - a6).T).T

        # post processing
        self.s_layer_trans[:, :, k] = trans
        self.s_layer_refl[:, :, k] = refl
        self.s_layer_source_up[:, k] = 0.0

        if self.mth_azi == 0:
            for i in range(len(self.cos_angle)):
                self.thermal_c[i, k] = 0.0
                for j in range(column.model_inputs.nb_streams):
                    self.thermal_c[i, k] += trans[i, j] + refl[i, j]
                if (i == len(self.cos_angle) - 1) and (
                    len(self.cos_angle) == (column.model_inputs.nb_streams + 1)
                ):
                    self.thermal_c[i, k] += trans[
                        len(self.cos_angle) - 1, len(self.cos_angle) - 1
                    ]
                self.s_layer_source_up[i, k] = (
                    1.0 - self.thermal_c[i, k]
                ) * self.planck_atmosphere[k]

                self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]

        # treatment of solar radiation is not in L&W2013 paper
        if self.solar_flag:
            n2 = 2 * len(self.cos_angle)
            n2_1 = -1
            source_up = np.zeros(len(self.cos_angle))
            source_down = np.zeros(len(self.cos_angle))

            # solar source
            sfactor = self.w[k] * self.solar_irradiance / np.pi
            if self.mth_azi == 0:
                sfactor /= 2.0

            expfactor = np.exp(-self.t_od[k] / self.cos_sun)
            s_transmittance = np.exp(-self.total_opt[k] / self.cos_sun)

            solar = np.zeros(n2)
            v0 = np.zeros((n2, n2))

            for i in range(len(self.cos_angle)):
                solar[i] = -self.bb[i, len(self.cos_angle), k] * sfactor  # bb(i, nZ+1)
                solar[i + len(self.cos_angle)] = (
                    -self.ff[i, len(self.cos_angle), k] * sfactor
                )  # ff(i, nZ+1)

                for j in range(len(self.cos_angle)):
                    v0[i, j] = self.w[k] * self.ff[i, j, k] * self.cos_weight[j]
                    v0[i + len(self.cos_angle), j] = (
                        self.w[k] * self.bb[i, j, k] * self.cos_weight[j]
                    )
                    v0[i, j + len(self.cos_angle)] = v0[i + len(self.cos_angle), j]
                    v0[
                        len(self.cos_angle) + i,
                        j + len(self.cos_angle),
                    ] = v0[i, j]

                v0[i, i] -= 1.0 + self.cos_angle[i] / self.cos_sun
                v0[i + len(self.cos_angle), i + len(self.cos_angle)] -= (
                    1.0 - self.cos_angle[i] / self.cos_sun
                )

            solar1 = np.linalg.solve(v0[:n2_1, :n2_1], solar[:n2_1])
            solar1 = np.append(solar1, 0.0)
            sfac2 = solar[n2 - 1] - np.sum(v0[n2 - 1, :n2_1] * solar1[:n2_1])

            for i in range(len(self.cos_angle)):
                source_up[i] = solar1[i]
                source_down[i] = expfactor * solar1[i + len(self.cos_angle)]

                for j in range(len(self.cos_angle)):
                    source_up[i] -= (
                        refl[i, j] * solar1[j + len(self.cos_angle)]
                        + trans[i, j] * expfactor * solar1[j]
                    )
                    source_down[i] -= (
                        trans[i, j] * solar1[j + len(self.cos_angle)]
                        + refl[i, j] * expfactor * solar1[j]
                    )

            
        
            # Specific treatment for downward source function
            if abs(v0[n2 - 1, n2 - 1]) > 1e-4:
                source_down[len(self.cos_angle) - 1] += (
                    (
                        expfactor
                        - trans[
                            len(self.cos_angle) - 1,
                            len(self.cos_angle) - 1,
                        ]
                    )
                    * sfac2
                    / v0[n2 - 1, n2 - 1]
                )
            else:
                source_down[len(self.cos_angle) - 1] -= (
                    expfactor
                    * sfac2
                    * self.t_od[k]
                    / self.cos_angle[len(self.cos_angle) - 1]
                )
                
            
         
            source_up *= s_transmittance
            source_down *= s_transmittance
            
            

            self.s_layer_source_up[:, k] += source_up
            self.s_layer_source_down[:, k] += source_down

        return None


def solve_advanced_adding_doubling(column, irradiance, wvl):
    """

    This subroutine calculates IR/MW radiance at the top of the atmosphere
    including atmospheric scattering. The scheme will include solar part.
    The ADA algorithm computes layer reflectance and transmittance as well
    as source function by the subroutine CRTM_Doubling_layer, then uses
    an adding method to integrate the layer and surface components.

    Translated by the snicar-fx team from the Fortran code of
    Quanhua Liu (Quanhua.Liu@noaa.gov)

    """

    aads = _AdvancedDoublingAddingSolver(column, irradiance, wvl)

    for k in range(1, column.nbr_lyr + 1):
        aads.total_opt[k] = aads.total_opt[k - 1] + aads.t_od[k - 1]

    aads.s_level_refl_up[:, :, -1] = aads.reflectivity

    if aads.mth_azi == 0:
        aads.s_level_rad_up[:, -1] = aads.emissivity * aads.planck_surface

    # adds a solar reflection term to the upward radiance at the
    # last layer for all viewing angles
    if aads.solar_flag:
        aads.s_level_rad_up[:, -1] += (
            aads.direct_reflectivity
            * aads.cos_sun
            * aads.solar_irradiance
            / np.pi
            * np.exp(-aads.total_opt[-1] / aads.cos_sun)
        )

    for k in range(column.nbr_lyr - 1, -1, -1):

        # call  multiple-stream algorithm for computing layer
        # transmission, reflection, and source functions.
        aads.crtm_anom_layer(column, k)
        # then Adding method to add the layer to the present level
        # to compute upward radiances and reflection matrix
        # at new level.
        
        

        # similar to equation B4 Briegleb and Light 2007
        temporal_matrix = -np.matmul(
            aads.s_level_refl_up[:, :, k + 1],
            aads.s_layer_refl[:, :, k],
        )
        
        
        np.fill_diagonal(temporal_matrix, temporal_matrix.diagonal() + 1.0)
        

        try:
            aads.inv_gamma_t[:, :, k] = np.linalg.solve(
                temporal_matrix.T, aads.s_layer_trans[:, :, k].T
            ).T
        except np.linalg.LinAlgError as e:
            print(f"Error solving temporal_matrix system: {e}")
            raise
        
        

        aads.refl_down[:, k] = np.matmul(
            aads.s_level_refl_up[:, :, k + 1], aads.s_layer_source_down[:, k]
        )
        
        # if k == 0:
        #     print("nv ", np.nanmean(aads.refl_down[:, k]))
        #     return 

        aads.s_level_rad_up[:, k] = aads.s_layer_source_up[:, k] + np.matmul(
            aads.inv_gamma_t[:, :, k],
            aads.refl_down[:, k] + aads.s_level_rad_up[:, k + 1],
        )
        
        # if k == 0:
        #     print("nv ", np.nanmean(aads.s_level_rad_up[:, k]))
        #     return 
        
        

        aads.refl_trans[:, :, k] = np.matmul(
            aads.s_level_refl_up[:, :, k + 1], aads.s_layer_trans[:, :, k]
        )
        
                
        # if k == 0:
        #     print("nv ", np.nanmean(aads.refl_trans[:, :, k]))
        #     return 
        
        

        aads.s_level_refl_up[:, :, k] = aads.s_layer_refl[:, :, k] + np.matmul(
            aads.inv_gamma_t[:, :, k], aads.refl_trans[:, :, k]
        )
        
                        
        # if k == 0:
        #     print("nv ", np.nanmean(aads.s_level_refl_up[:, :, k]))
        #     print("nv ", np.nanmean(aads.inv_gamma_t[:, :, k]))

        #     return 


    if aads.mth_azi == 0:
        for i in range(len(aads.cos_angle)):
            aads.s_level_rad_up[i, 0] += (
                np.sum(aads.s_level_refl_up[i, :, 0]) * aads.cosmic_background
            )

    albedo = (
        2
        * np.pi
        * np.sum(
            aads.s_level_rad_up[:, 0]
            * np.array(aads.cos_angle)
            * np.array(aads.cos_weight)
        )
        / (aads.solar_irradiance * aads.cos_sun)
    )

    return albedo
