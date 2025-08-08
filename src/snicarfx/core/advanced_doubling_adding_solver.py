#!/usr/bin/env python3
"""

Implementation of the Advanced Doubling Adding Method for Radiative
Transfer in Planetary Atmospheres


Quanhua Liu and Fuzhong Weng, 2006
https://doi.org/10.1175/JAS3808.1

"""

import numpy as np
from scipy.special import legendre


class _AdvancedDoublingAddingSolver:

    def __init__(self, column, irradiance, wvl):
        """
        Initialize all variables required for the ADA solver
        """
        
        #################################################### ADDED OR MODIFIED
        self.mth_azi = 0
        self.planck_surface = 0
        self.wvl = wvl
        self.solar_irradiance = 2
        self.solar_flag = True
        self.cos_sun = irradiance.cos_sza
        
        n_angles = 8
        nodes, weights = np.polynomial.legendre.leggauss(n_angles*2)
        self.cos_angle = nodes[n_angles:] # only positive 
        self.cos_weight = weights[n_angles:]
        
        mu = np.array(self.cos_angle)
        
        self.ff = np.zeros((len(self.cos_angle), 
                            len(self.cos_angle)+1,
                            column.nbr_lyr))
        self.bb = np.zeros((len(self.cos_angle), 
                            len(self.cos_angle)+1,
                            column.nbr_lyr))
        
        
        tau_unscaled = column.tau[:, self.wvl] 
        w_unscaled = column.ss_alb[:, self.wvl]
        g_unscaled = column.asm_prm[:, self.wvl] 
        
        m = 8 # Wiscombe
        self.phase_coeffs = np.zeros((2 * m - 1, 
                                      column.nbr_lyr))
        
        self.t_od = tau_unscaled
        self.w = w_unscaled
        self.g = g_unscaled
    

        ######################################################################
        #### calc. phase coefficients + apply delta scaling (CRTM_Atm_Combine)
        #### (Delta-scaling does nothing if g is low (ie low forward scatter))
        ######################################################################
        for k in range(column.nbr_lyr):
            
            g = g_unscaled[k]
            
            # HG Legendre expansion coefficients 
            # for leg_moment in range(n_legendre + 1):
            #     self.phase_coeffs[leg_moment, k] = (
            #         0.5 * (2*leg_moment+1) * g**leg_moment
            #         )
            
            ######################### DELTA CORRECTION ####################### 
            
            # Delta truncation: get highest Legendre term following
            # Wicombe 1977 Eq. (15)
            f = g ** (2*m) # original expansion coeff at 2M
            
            
            # # Wiscombe 1977 Eq. 14
            for order in range(2*m-1):
                self.phase_coeffs[order, k] = (g**order - f) / (1 - f)
                self.phase_coeffs[order,k] = ((2*order + 1) * 0.5 
                                              * self.phase_coeffs[order, k])
                
            # Wiscombe 1977 Eq. 20(a, b)
            self.t_od[k] = (1.0 - w_unscaled[k] * f) * tau_unscaled[k]
            self.w[k] = (1.0 - f) * w_unscaled[k] / (1 - w_unscaled[k] * f)
            
        leg_poly = np.zeros((2*m-1, n_angles+1))
        
        for i in range(n_angles):
            mu = self.cos_angle[i]
            for order in range(2*m-1):
                leg_poly[order, i] = legendre(order)(mu)
        
        # add SZA as last column
        for order in range(2*m-1):
            leg_poly[order, n_angles] = legendre(order)(self.cos_sun)
            
        jn = n_angles + 1 if self.solar_flag else n_angles   
        
        # !!!!!!!! the phase functions are clipped to pos. values in original
        # code but we don't want to clip them - they shouldnt be negative
        # off and obb are "original phase matrices" and pff/pff are clipped.
        
        for k in range(column.nbr_lyr):
            for j in range(jn):  # incoming angle
                for i in range(n_angles):  # outgoing angle
                    off = 0.0
                    obb = 0.0
                    for leg in range(self.mth_azi, 2*m-1):
                        ifac = (-1) ** (leg - self.mth_azi)
                        coeff = self.phase_coeffs[leg, k]  
                        off += coeff * leg_poly[leg, i] * leg_poly[leg, j] 
                        obb += coeff * leg_poly[leg, i] * leg_poly[leg, j] * ifac 
                    self.ff[i, j, k] = off
                    self.bb[i, j, k] = obb
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

            ###################################################################
            
        ######################################################################
        #### calc. legendre polys + calc. phase matrices (CRTM_Phase_Matrix)
        ######################################################################
        
        # n_legendre = 50  # Order of expansion = n+1 terms

        # leg_poly = np.zeros((n_legendre+1, 
        #                 n_angles+1))
        
        # for i in range(n_angles):
        #     mu = self.cos_angle[i]
        #     for leg_moment in range(n_legendre + 1):
        #         leg_poly[leg_moment, i] = legendre(leg_moment)(mu)
        
        # # add SZA as last column
        # for leg_moment in range(n_legendre + 1):
        #     leg_poly[leg_moment, n_angles] = legendre(leg_moment)(self.cos_sun)
            
        # jn = n_angles + 1 if self.solar_flag else n_angles   
        
        # # !!!!!!!! the phase functions are clipped to pos. values in original
        # # code but we don't want to clip them - they shouldnt be negative
        # # off and obb are "original phase matrices" and pff/pff are clipped.
        
        # for k in range(column.nbr_lyr):
        #     for j in range(jn):  # incoming angle
        #         for i in range(n_angles):  # outgoing angle
        #             off = 0.0
        #             obb = 0.0
        #             for leg in range(self.mth_azi, n_legendre + 1):
        #                 ifac = (-1) ** (leg - self.mth_azi)
        #                 coeff = self.phase_coeffs[leg, k]  
        #                 off += coeff * leg_poly[leg, i] * leg_poly[leg, j] 
        #                 obb += coeff * leg_poly[leg, i] * leg_poly[leg, j] * ifac 
        #             self.ff[i, j, k] = off
        #             self.bb[i, j, k] = obb
        #             if (self.bb[i, j, k] < 0) or (self.ff[i, j, k] < 0):
        #                 print(self.bb[i, j, k])
        #                 raise ValueError("Negative phase matrix elements")
        
        ######################################################################
        self.direct_reflectivity = np.zeros(len(self.cos_angle))

        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999
        

        self.planck_atmosphere = np.zeros(column.nbr_lyr + 1)
        
        self.emissivity = np.zeros(len(self.cos_angle))


        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10
        
        self.cosmic_background = 0

        self.total_opt = np.zeros(column.nbr_lyr + 1)
        
        # these are the Legendre matrices built from ff and bb
        self.pp = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.pm = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.ppm = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.i_ppm = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.ppp = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )

        self.reflectivity = np.zeros(
            (len(self.cos_angle), len(self.cos_angle))
        )

        self.temporal_matrix = np.zeros(
            (len(self.cos_angle), len(self.cos_angle))
        )
        self.refl_down = np.zeros((len(self.cos_angle),
                                  column.nbr_lyr))
        
        self.inv_gamma = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.inv_gamma_t = np.zeros_like(
            self.inv_gamma
        )
        
        self.refl_trans = np.zeros_like(
            self.inv_gamma
        )

        self.s_layer_trans = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.s_layer_refl = np.zeros_like(
            self.s_layer_trans
        )
        self.s_level_refl_up = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr + 1)
        )
        self.s_level_rad_up = np.zeros(
            (len(self.cos_angle), column.nbr_lyr + 1)
            )
        self.s_layer_source_up = np.zeros(
            (len(self.cos_angle), column.nbr_lyr)
            )
        self.s_layer_source_down = np.zeros(
            (len(self.cos_angle), column.nbr_lyr)
        )



        self.thermal_c = np.zeros(
            (len(self.cos_angle),
            column.nbr_lyr)
        )


        
        self.hh = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )

        self.eig_value = np.zeros(
            (len(self.cos_angle),
            column.nbr_lyr)
        )
        self.eig_va = np.zeros(
            (len(self.cos_angle),
            column.nbr_lyr)
        )
        self.eig_ve = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.eig_veva = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.eig_vef = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )

        self.gp = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.gm = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.i_gm = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.a1 = np.zeros(
            (len(self.cos_angle),
            len(self.cos_angle),
            column.nbr_lyr)
        )
        self.a2 = np.zeros_like(
            self.a1
        )
        self.a3 = np.zeros_like(
            self.a1
        )
        self.a4 = np.zeros_like(
            self.a1
        )
        self.a5 = np.zeros_like(
            self.a1
        )
        self.a6 = np.zeros_like(
            self.a1
        )
        self.gm_a5 = np.zeros_like(
            self.a1
        )
        self.i_gm_a5 = np.zeros_like(
            self.a1
        )

        self.exp_x = np.zeros(
            (len(self.cos_angle),
            column.nbr_lyr)
        )

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

        for i in range(len(self.cos_angle)):
            c = self.w[k] / self.cos_angle[i]

            for j in range(len(self.cos_angle)):
                # EQUATION 6A AMOM PAPER (without the Kronecker delta)
                self.pp[i, j, k] = c * self.ff[i, j, k] * self.cos_weight[j]
                # EQUATION 6B AMOM PAPER 
                self.pm[i, j, k] = c * self.bb[i, j, k] * self.cos_weight[j]
                
            # EQUATION 6A + 7 (apply Kronecker delta) such that pp = alpha
            self.pp[i, i, k] -= 1.0 / self.cos_angle[i]
        
        # First term in H matrix: alpha - beta 
        self.ppm[:, :, k] = self.pp[:, :, k] - self.pm[:, :, k]
        # First term in H matrix: alpha + beta 
        self.ppp[:, :, k] = self.pp[:, :, k] + self.pm[:, :, k]
        # EQUATION 10 IN AMOM PAPER
        self.hh[:, :, k] = np.matmul(self.ppm[:, :, k], self.ppp[:, :, k])
                
        
        tempo = self.hh[:, :, k].copy()
        
        eig_vals, eig_vecs = np.linalg.eig(tempo)

        self.eig_ve[:, :, k] = eig_vecs
        self.eig_va[:, k] = eig_vals
       
        self.eig_value[:, k] = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)
    
        
        
        # eig_veva[i, j, k] = eig_ve[i, j, k] * eig_value[j, k]
        self.eig_veva[:, :, k] = (
            self.eig_ve[:, :, k] * self.eig_value[np.newaxis, :, k]
        )
        
        self.eig_vef[:, :, k] = np.linalg.solve(self.ppm[:,:,k], 
                                                self.eig_veva[:,:,k])


        # Compute layer reflection (Gp) and transmission (Gm) matrices

        self.gp[:, :, k] = (self.eig_ve[:, :, k] + self.eig_vef[:, :, k]) / 2.0
        self.gm[:, :, k] = (self.eig_ve[:, :, k] - self.eig_vef[:, :, k]) / 2.0


        for i in range(len(self.cos_angle)):
            xx = self.eig_value[i, k] * self.t_od[k]
            self.exp_x[i, k] = np.exp(-xx)

        for i in range(len(self.cos_angle)):
            for j in range(len(self.cos_angle)):
                self.a1[i, j, k] = self.gp[i, j, k] * self.exp_x[j, k]
                self.a4[i, j, k] = self.gm[i, j, k] * self.exp_x[j, k]
                

        self.a2[:, :, k] = np.linalg.solve(self.gm[:, :, k], self.a1[:, :, k])
        self.a3[:, :, k] = np.matmul(self.gp[:, :, k], self.a2[:, :, k])
        self.a5[:, :, k] = np.matmul(self.a1[:, :, k], self.a2[:, :, k])
        self.a6[:, :, k] = np.matmul(self.a4[:, :, k], self.a2[:, :, k])

        self.gm_a5[:, :, k] = self.gm[:, :, k] - self.a5[:, :, k]
        
        trans = np.linalg.solve(self.gm_a5[:, :, k].T, 
                                (self.a4[:, :, k] - self.a3[:, :, k]).T).T

        refl = np.linalg.solve(self.gm_a5[:, :, k].T, 
                                (self.gp[:, :, k] - self.a6[:, :, k]).T).T


        # post processing
        self.s_layer_trans[:, :, k] = trans[:, :]
        self.s_layer_refl[:, :, k] = refl[:, :]
        self.s_layer_source_up[:, k] = 0.0

        if self.mth_azi == 0:
            for i in range(len(self.cos_angle)):
                self.thermal_c[i, k] = 0.0
                for j in range(column.model_inputs.nb_streams):
                    self.thermal_c[i, k] += trans[i, j] + refl[i, j]
                if ((i == len(self.cos_angle) - 1) 
                    and 
                    (len(self.cos_angle) 
                     == (column.model_inputs.nb_streams + 1))):
                    self.thermal_c[i, k] += trans[len(self.cos_angle) - 1, 
                                                  len(self.cos_angle) - 1]
                self.s_layer_source_up[i, k] = (
                    1.0 - self.thermal_c[i, k]
                ) * self.planck_atmosphere[k]
                
                self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]
                
        # compute visible part for visible channels during daytime
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
                solar[i] = (
                    -self.bb[i, len(self.cos_angle), k] * sfactor
                )  # bb(i, nZ+1)
                solar[i + len(self.cos_angle)] = (
                    -self.ff[i, len(self.cos_angle), k] * sfactor
                )  # ff(i, nZ+1)

                for j in range(len(self.cos_angle)):
                    v0[i, j] = self.w[k] * self.ff[i, j, k] * self.cos_weight[j]
                    v0[i + len(self.cos_angle), j] = (
                        self.w[k] * self.bb[i, j, k] * self.cos_weight[j]
                    )
                    v0[i, j + len(self.cos_angle)] = v0[
                        i + len(self.cos_angle), j
                    ]
                    v0[
                        len(self.cos_angle) + i,
                        j + len(self.cos_angle),
                    ] = v0[i, j]
            
            
                v0[i, i] -= 1.0 + self.cos_angle[i] / self.cos_sun
                v0[i + len(self.cos_angle), 
                   i + len(self.cos_angle)] -= (
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


    for k in range(column.nbr_lyr-1, -1, -1):

        # call  multiple-stream algorithm for computing layer
        # transmission, reflection, and source functions.
        aads.crtm_anom_layer(column, k)
        # then Adding method to add the layer to the present level
        # to compute upward radiances and reflection matrix
        # at new level.

        # similar to equation B4 Briegleb and Light 2007
        temporal_matrix = -np.matmul(
            aads.s_level_refl_up[:, :, k+1],
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
            aads.s_level_refl_up[:, :, k+1], aads.s_layer_source_down[:, k]
        )
        
        

        aads.s_level_rad_up[:, k] = aads.s_layer_source_up[
            :, k
        ] + np.matmul(
            aads.inv_gamma_t[:, :, k],
            aads.refl_down[:, k] + aads.s_level_rad_up[:, k+1],
        )
        

        aads.refl_trans[:, :, k] = np.matmul(
            aads.s_level_refl_up[:, :, k+1], aads.s_layer_trans[:, :, k]
        )
        

        aads.s_level_refl_up[:, :, k] = aads.s_layer_refl[
            :, :, k 
        ] + np.matmul(aads.inv_gamma_t[:, :, k], aads.refl_trans[:, :, k])
        


    if aads.mth_azi == 0:
        for i in range(len(aads.cos_angle)):
            aads.s_level_rad_up[i, 0] += (
                np.sum(aads.s_level_refl_up[i, :, 0]) * aads.cosmic_background
            )
    
    albedo = (2 * np.pi 
              * np.sum(aads.s_level_rad_up[:,0] 
                       * np.array(aads.cos_angle) 
                       * np.array(aads.cos_weight))
              /
              (aads.solar_irradiance * aads.cos_sun)
              )

    return albedo
