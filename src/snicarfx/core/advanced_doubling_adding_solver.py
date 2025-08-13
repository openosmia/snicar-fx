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

        tau_unscaled = column.tau[:, self.wvl]
        w_unscaled = column.ss_alb[:, self.wvl]
        g_unscaled = column.asm_prm[:, self.wvl]

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
                            raise ValueError("Negative phase matrix elements")

        return None

    def get_trans_refl_layer(self, k):
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

        self.s_layer_trans[:, :, k] = trans
        self.s_layer_refl[:, :, k] = refl

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

    identity_matrix = np.eye(len(aads.cos_angle))  # identity matrix, size = n_angles

    for k in range(column.nbr_lyr - 1, -1, -1):
        print("layer: ", k)

        #######################################################################
        ## GET LAYER TRANS/REFL MATRICES
        #######################################################################

        # call multiple-stream algorithm for computing symmetric layer
        # transmission & reflection (s_layer_trans[:,:,k] & s_layer_refl[:,:,k])
        aads.get_trans_refl_layer(k)

        # up and down properties are the same w/out Fresnel
        s_layer_refl_up = aads.s_layer_refl[:, :, k]
        s_layer_refl_down = aads.s_layer_refl[:, :, k]
        s_layer_trans_down = aads.s_layer_trans[:, :, k]
        s_layer_trans_up = aads.s_layer_trans[:, :, k]

        #######################################################################
        ## ONCE TRANS/REFL MATRICES ARE CALCULATED,
        # CALC SOURCE TERMS
        #######################################################################

        # INITIALIZATIONS
        aads.s_layer_source_up[:, k] = 0.0
        n = len(aads.cos_angle)
        n2 = 2 * len(aads.cos_angle)
        n2_1 = -1
        source_up = np.zeros(len(aads.cos_angle))
        source_down = np.zeros(len(aads.cos_angle))
        solar = np.zeros(n2)
        v0 = np.zeros((n2, n2))

        # ignore this loop for now as it's for thermal
        if aads.mth_azi == 0:
            # Sum of transmission + reflection across each row
            aads.thermal_c[:n, k] = np.sum(
                s_layer_trans_up[:n, : column.model_inputs.nb_streams]
                + s_layer_refl_down[:n, : column.model_inputs.nb_streams],
                axis=1,
            )

            if n == column.model_inputs.nb_streams + 1:
                aads.thermal_c[n - 1, k] += s_layer_trans_up[n - 1, n - 1]

            aads.s_layer_source_up[:n, k] = (
                1.0 - aads.thermal_c[:n, k]
            ) * aads.planck_atmosphere[k]
            aads.s_layer_source_down[:n, k] = aads.s_layer_source_up[:n, k]

        if aads.solar_flag:

            # SCATTERING OF SOLAR BEAM IN THE LAYER
            # ---> solar describes the injection of energy from the beam
            # into the layer (note bb and ff are indexed at n which is the SZA)
            sfactor = aads.w[k] * aads.solar_irradiance / np.pi
            if aads.mth_azi == 0:
                sfactor /= 2.0
            solar[:n] = -sfactor * aads.bb[:, n, k]
            solar[n:] = -sfactor * aads.ff[:, n, k]

            # SCATTERING WITHIN THE LAYER
            # ---> v0 describes how the scattering behaves within the layer
            # for all angle combinations
            # upward direction
            v0[:n, :n] = aads.w[k] * aads.ff[:, :-1, k] * aads.cos_weight[np.newaxis, :]
            # downward direction
            v0[n:, :n] = aads.w[k] * aads.bb[:, :-1, k] * aads.cos_weight[np.newaxis, :]
            # Symmetry of scattering in azimuth
            v0[:n, n:] = v0[n:, :n]
            v0[n:, n:] = v0[:n, :n]

            # extinction (diag term) + projected solar beam loss
            v0[np.arange(n), np.arange(n)] -= 1.0 + aads.cos_angle / aads.cos_sun
            v0[n + np.arange(n), n + np.arange(n)] -= (
                1.0 - aads.cos_angle / aads.cos_sun
            )

            # RTE is (identity - ss_alb * P * weights) * I = solar_beam
            # which here corresponds to v0 * solar1 = solar
            # so we solve for solar1, which  represents the intensity scattered
            # in all directions due to the injection of the solar beam +
            # multiple scattering within the layer, and is defined for
            # angles in both hemispheres (!)
            solar1 = np.linalg.solve(v0[:n2_1, :n2_1], solar[:n2_1])
            solar1 = np.append(solar1, 0.0)

            # get upward directed emission:
            # we remove the amount of energy that is being reflected &
            # transmitted upwards by the layer
            source_up = solar1[:n]
            source_up -= s_layer_refl_down @ solar1[n:] + s_layer_trans_up @ (
                np.exp(-aads.t_od[k] / aads.cos_sun) * solar1[:n]
            )

            # get upward directed emission:
            # first we multiply by the attenuation as we want the source leaving
            # the bottom of the layer
            # then we remove the amount of energy that is being reflected &
            # transmitted downward by the layer
            source_down = np.exp(-aads.t_od[k] / aads.cos_sun) * solar1[n:]
            source_down -= s_layer_trans_down @ solar1[n:] + s_layer_refl_up @ (
                np.exp(-aads.t_od[k] / aads.cos_sun) * solar1[:n]
            )

            print("n", np.nanmean(source_up), np.nanmean(source_down))
            # calculate the downward emission at the last angle (not incl. in solar1)
            # condition checks to avoid division by 0 and uses a lineralized
            # expression if v0[-1] is too low
            sfac2 = solar[n2 - 1] - np.sum(v0[n2 - 1, :n2_1] * solar1[:n2_1])

            if abs(v0[n2 - 1, n2 - 1]) > 1e-4:
                source_down[-1] += (
                    (np.exp(-aads.t_od[k] / aads.cos_sun) - s_layer_trans_down[-1, -1])
                    * sfac2
                    / v0[n2 - 1, n2 - 1]
                )
            else:
                source_down[-1] -= (
                    np.exp(-aads.t_od[k] / aads.cos_sun)
                    * sfac2
                    * aads.t_od[k]
                    / aads.cos_angle[-1]
                )

            # finally, weigh the source terms by how much solar radiation
            # actually reaches the given layer by using the cumulative
            # optical depth
            source_up *= np.exp(-aads.total_opt[k] / aads.cos_sun)
            source_down *= np.exp(-aads.total_opt[k] / aads.cos_sun)

            # LOCAL SOURCE FUNCTION FOR LAYER K
            aads.s_layer_source_up[:, k] += source_up
            aads.s_layer_source_down[:, k] += source_down

        #######################################################################
        ## APPLY ADDING METHOD
        #######################################################################

        infinite_scattering = identity_matrix - np.matmul(
            aads.s_level_refl_up[:, :, k + 1], s_layer_refl_down
        )

        infinite_scattering_inv = np.linalg.inv(infinite_scattering)

        aads.s_level_refl_up[:, :, k] = s_layer_refl_up + (
            s_layer_trans_down @ infinite_scattering_inv
        ) @ (aads.s_level_refl_up[:, :, k + 1] @ s_layer_trans_up)

        aads.refl_down[:, k] = np.matmul(
            aads.s_level_refl_up[:, :, k + 1], aads.s_layer_source_down[:, k]
        )

        aads.s_level_rad_up[:, k] = aads.s_layer_source_up[:, k] + np.matmul(
            (s_layer_trans_down @ infinite_scattering_inv),
            aads.refl_down[:, k] + aads.s_level_rad_up[:, k + 1],
        )

    #######################################################################
    ## GET HEMISPHERICAL ALBEDO
    #######################################################################
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

    # refl = [(
    #         2
    #         * np.pi
    #         * aads.s_level_rad_up[i, 0]
    #         * aads.cos_angle[i]
    #     / (aads.solar_irradiance * aads.cos_sun)
    # ) for i in range(len(aads.cos_angle))]

    return albedo


def rf(mu, re1, im1, re2, im2):
    """
    This function calculate the Fresnel reflectivity from the refractive
    indices of two media and the angle of incidence, accounting for TIR.
    The critical angle is calculated from the complex RI, while the
    Fresnel coefficients are calculated with the adjusted RI from
    Liou et al. 2002, following the formalism of Whicker et al. 2022.

    Parameters
    ----------
    mu : array or list
        nodes/points of the gaussian integration (in cos(theta))
    re1 : float
        real part of the relative refractive index of medium 1 (incident)
    im1 : float
        imaginary part of the relative refractive index of medium 1
    re2 : float
        real part of the relative refractive index of medium 2 (transm.)
    im2 : float
        imaginary part of the relative refractive index of medium 2
    use_scipy: boolean



    Returns
    -------
    result : array or list
        Fresnel reflectivity coefficient (* mu if scipy used)

    """

    # 1 - calculate values of theta where TIR occurs using complex ref index
    theta_c = np.arcsin((re2 - 1j * im2) / (re1 - 1j * im1))  # critical angle
    mask = np.arccos(mu) >= theta_c  # mask where TIR occurs

    if im1 == 0:  # in this case air is above, rfix of ice is _2
        temp1 = re2**2 - im2**2 + np.sin(np.arccos(mu)) ** 2
        temp2 = re2**2 - im2**2 - np.sin(np.arccos(mu)) ** 2
        nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * (re2**2) * (im2**2)) ** 0.5
        ) ** 0.5

        # angle of transmitted radiation
        mu0n = np.cos(np.arcsin(np.sin(np.arccos(mu)) / nr))

    else:  # in this case air is below, rfix of ice is _1
        temp1 = re1**2 - im1**2 + np.sin(np.arccos(mu)) ** 2
        temp2 = re1**2 - im1**2 - np.sin(np.arccos(mu)) ** 2
        nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * (re1**2) * (im1**2)) ** 0.5
        ) ** 0.5

        # angle of transmitted radiation
        # first clip the few values above 1 due to mu close to 0
        temp = np.clip(np.sin(np.arccos(mu)) * nr, 0, 1)
        mu0n = np.cos(np.arcsin(temp))

    # 3 - calculate reflectivity using Fresnel equations
    # Eq. 22  Briegleb & Light 2007 or from Liou 2002
    r1 = (mu - nr * mu0n) / (
        mu + nr * mu0n
    )  # reflection amplitude factor for perpendicular polarization
    r2 = (nr * mu - mu0n) / (
        nr * mu + mu0n
    )  # reflection amplitude factor for parallel polarization
    rf = 0.5 * (r1**2 + r2**2)

    # 4 - mask reflectivity where TIR occurs
    rf[mask] = 1

    return rf


def rf_times_mu(mu, re1, im1, re2, im2):
    """
    This function calculate the Fresnel reflectivity from the refractive
    indices of two media and the angle of incidence, accounting for TIR.
    The critical angle is calculated from the complex RI, while the
    Fresnel coefficients are calculated with the adjusted RI from
    Liou et al. 2002, following the formalism of Whicker et al. 2022.

    Parameters
    ----------
    mu : array or list
        nodes/points of the gaussian integration (in cos(theta))
    re1 : float
        real part of the relative refractive index of medium 1 (incident)
    im1 : float
        imaginary part of the relative refractive index of medium 1
    re2 : float
        real part of the relative refractive index of medium 2 (transm.)
    im2 : float
        imaginary part of the relative refractive index of medium 2
    use_scipy: boolean



    Returns
    -------
    result : array or list
        Fresnel reflectivity coefficient (* mu if scipy used)

    """

    # 1 - calculate values of theta where TIR occurs using complex ref index
    theta_c = np.arcsin((re2 - 1j * im2) / (re1 - 1j * im1))  # critical angle
    mask = np.arccos(mu) >= theta_c  # mask where TIR occurs

    if im1 == 0:  # in this case air is above, rfix of ice is _2
        temp1 = re2**2 - im2**2 + np.sin(np.arccos(mu)) ** 2
        temp2 = re2**2 - im2**2 - np.sin(np.arccos(mu)) ** 2
        nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * (re2**2) * (im2**2)) ** 0.5
        ) ** 0.5

        # angle of transmitted radiation
        mu0n = np.cos(np.arcsin(np.sin(np.arccos(mu)) / nr))

    else:  # in this case air is below, rfix of ice is _1
        temp1 = re1**2 - im1**2 + np.sin(np.arccos(mu)) ** 2
        temp2 = re1**2 - im1**2 - np.sin(np.arccos(mu)) ** 2
        nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * (re1**2) * (im1**2)) ** 0.5
        ) ** 0.5

        # angle of transmitted radiation
        # first clip the few values above 1 due to mu close to 0
        temp = np.clip(np.sin(np.arccos(mu)) * nr, 0, 1)
        mu0n = np.cos(np.arcsin(temp))

    # 3 - calculate reflectivity using Fresnel equations
    # Eq. 22  Briegleb & Light 2007 or from Liou 2002
    r1 = (mu - nr * mu0n) / (
        mu + nr * mu0n
    )  # reflection amplitude factor for perpendicular polarization
    r2 = (nr * mu - mu0n) / (
        nr * mu + mu0n
    )  # reflection amplitude factor for parallel polarization
    rf = 0.5 * (r1**2 + r2**2)

    # 4 - mask reflectivity where TIR occurs
    rf[mask] = 1

    return rf * mu


def solve_advanced_adding_doubling_fresnel(column, irradiance, wvl):
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

    identity_matrix = np.eye(len(aads.cos_angle))  # identity matrix, size = n_angles

    for k in range(column.nbr_lyr - 1, -1, -1):

        #######################################################################
        ## GET LAYER TRANS/REFL MATRICES
        #######################################################################

        # call multiple-stream algorithm for computing symmetric layer
        # transmission & reflection (s_layer_trans[:,:,k] & s_layer_refl[:,:,k])
        aads.get_trans_refl_layer(k)
        # modify the trans/refl matrices with Fresnel if necessary
        # (!) trans and refl up/down are updated and become asymmetric

        if k == 0:
            ###################################################################
            ######## ATTEMPT 1: SPECULAR COEFFS
            ###################################################################
            # air -> ice interface:
            # reflectance/transmittance to light traveling downwards
            s_layer_fresnel_refl_down_vec = rf(
                aads.cos_angle,
                1,
                0,
                column.ref_idx_re[aads.wvl],
                column.ref_idx_im[aads.wvl],
            )
            # Convert to diagonal reflection/transmission matrices
            s_layer_fresnel_refl_down = np.diag(s_layer_fresnel_refl_down_vec)
            s_layer_fresnel_trans_down = np.diag(1.0 - s_layer_fresnel_refl_down_vec)

            # ice -> air interface:
            # reflectance/transmittance to light traveling upwards
            s_layer_fresnel_refl_up_vec = rf(
                aads.cos_angle,
                column.ref_idx_re[aads.wvl],
                column.ref_idx_im[aads.wvl],
                1,
                0,  # Air below
            )
            # Convert to diagonal reflection/transmission matrices
            s_layer_fresnel_refl_up = np.diag(s_layer_fresnel_refl_up_vec)
            s_layer_fresnel_trans_up = np.diag(1.0 - s_layer_fresnel_refl_up_vec)

            ###################################################################
            ######## ATTEMPT 2: DIFFUSE COEFFS (so NOT at the surface (!))
            ###################################################################
            # from scipy.integrate import fixed_quad

            # s_layer_fresnel_refl_down_vec = fixed_quad(
            #     rf_times_mu, 0, 1,
            #     args = (1,0,
            #             column.ref_idx_re[aads.wvl],
            #             column.ref_idx_im[aads.wvl]),
            #     n=1000)[0] / 0.5 # this give a single value

            # s_layer_fresnel_refl_up_vec = fixed_quad(
            #     rf_times_mu, 0, 1,
            #     args = (column.ref_idx_re[aads.wvl],
            #             column.ref_idx_im[aads.wvl],
            #             1,0),
            #     n=1000)[0] / 0.5 # this give a single value

            # s_layer_fresnel_refl_down = np.diag(s_layer_fresnel_refl_down_vec
            # * np.ones(8))
            # s_layer_fresnel_refl_up =  np.diag(s_layer_fresnel_refl_up_vec
            # * np.ones(8))

            # s_layer_fresnel_trans_up = np.diag(1.0 - s_layer_fresnel_refl_up_vec
            # * np.ones(8))

            # s_layer_fresnel_trans_down = np.diag(1.0 - s_layer_fresnel_refl_down_vec
            # * np.ones(8))

            # Combine reflectance of current layer with pseudo-Fresnel layer
            s_layer_refl_up = aads.s_layer_refl[:, :, k]
            s_layer_refl_down = aads.s_layer_refl[:, :, k]
            s_layer_trans_down = aads.s_layer_trans[:, :, k]
            s_layer_trans_up = aads.s_layer_trans[:, :, k]

            denom1 = np.linalg.inv(
                identity_matrix - s_layer_fresnel_refl_up @ s_layer_refl_down
            )
            denom2 = np.linalg.inv(
                identity_matrix - s_layer_refl_down @ s_layer_fresnel_refl_up
            )

            # light traveling upwards:
            s_layer_refl_up = (
                s_layer_refl_up
                + s_layer_trans_down
                @ denom1
                @ s_layer_fresnel_refl_up
                @ s_layer_trans_up
            )

            s_layer_trans_up = s_layer_fresnel_trans_up @ denom2 @ s_layer_trans_up

            s_layer_refl_down = (
                s_layer_fresnel_refl_down
                + s_layer_fresnel_trans_up
                @ denom2
                @ s_layer_refl_down
                @ s_layer_fresnel_trans_down
            )

            s_layer_trans_down = (
                s_layer_trans_down @ denom1 @ s_layer_fresnel_trans_down
            )

        # if no fresnel layer, the trans/refl properties are symmetric
        else:

            s_layer_refl_up = aads.s_layer_refl[:, :, k]
            s_layer_refl_down = aads.s_layer_refl[:, :, k]
            s_layer_trans_down = aads.s_layer_trans[:, :, k]
            s_layer_trans_up = aads.s_layer_trans[:, :, k]

        #######################################################################
        ## ONCE TRANS/REFL MATRICES ARE CALCULATED,
        # CALC SOURCE TERMS
        #######################################################################

        # INITIALIZATIONS
        aads.s_layer_source_up[:, k] = 0.0
        n = len(aads.cos_angle)
        n2 = 2 * len(aads.cos_angle)
        n2_1 = -1
        source_up = np.zeros(len(aads.cos_angle))
        source_down = np.zeros(len(aads.cos_angle))
        solar = np.zeros(n2)
        v0 = np.zeros((n2, n2))

        # can ignore this loop as it's for thermal
        if aads.mth_azi == 0:
            # Sum of transmission + reflection across each row
            aads.thermal_c[:n, k] = np.sum(
                s_layer_trans_up[:n, : column.model_inputs.nb_streams]
                + s_layer_refl_down[:n, : column.model_inputs.nb_streams],
                axis=1,
            )

            if n == column.model_inputs.nb_streams + 1:
                aads.thermal_c[n - 1, k] += s_layer_trans_up[n - 1, n - 1]

            aads.s_layer_source_up[:n, k] = (
                1.0 - aads.thermal_c[:n, k]
            ) * aads.planck_atmosphere[k]
            aads.s_layer_source_down[:n, k] = aads.s_layer_source_up[:n, k]

        # treatment of solar radiation
        if aads.solar_flag:
            # TRANSMISSION OF DIRECT BEAM
            expfactor = np.exp(-aads.t_od[k] / aads.cos_sun)
            s_transmittance = np.exp(-aads.total_opt[k] / aads.cos_sun)

            # SCATTERING OF SOLAR BEAM IN THE LAYER
            sfactor = aads.w[k] * aads.solar_irradiance / np.pi
            if aads.mth_azi == 0:
                sfactor /= 2.0
            solar[:n] = -sfactor * aads.bb[:, n, k]
            solar[n:] = -sfactor * aads.ff[:, n, k]

            # Top-left block: v0[0:n, 0:n]
            v0[:n, :n] = aads.w[k] * aads.ff[:, :-1, k] * aads.cos_weight[np.newaxis, :]

            # Bottom-left block: v0[n:2n, 0:n]
            v0[n:, :n] = aads.w[k] * aads.bb[:, :-1, k] * aads.cos_weight[np.newaxis, :]

            # Top-right block: symmetric with bottom-left
            v0[:n, n:] = v0[n:, :n]

            # Bottom-right block: symmetric with top-left
            v0[n:, n:] = v0[:n, :n]

            # Diagonal adjustments
            v0[np.arange(n), np.arange(n)] -= 1.0 + aads.cos_angle / aads.cos_sun
            v0[n + np.arange(n), n + np.arange(n)] -= (
                1.0 - aads.cos_angle / aads.cos_sun
            )

            # solve for S term
            solar1 = np.linalg.solve(v0[:n2_1, :n2_1], solar[:n2_1])
            solar1 = np.append(solar1, 0.0)
            sfac2 = solar[n2 - 1] - np.sum(v0[n2 - 1, :n2_1] * solar1[:n2_1])

            source_up = solar1[:n]
            source_down = expfactor * solar1[n:]

            source_up -= s_layer_refl_down @ solar1[n:] + s_layer_trans_up @ (
                expfactor * solar1[:n]
            )

            source_down -= s_layer_trans_down @ solar1[n:] + s_layer_refl_up @ (
                expfactor * solar1[:n]
            )

            if abs(v0[n2 - 1, n2 - 1]) > 1e-4:
                source_down[-1] += (
                    (expfactor - s_layer_trans_down[-1, -1])
                    * sfac2
                    / v0[n2 - 1, n2 - 1]
                )
            else:
                source_down[-1] -= expfactor * sfac2 * aads.t_od[k] / aads.cos_angle[-1]

            source_up *= s_transmittance
            source_down *= s_transmittance

            # LOCAL SOURCE FUNCTION FOR LAYER K
            aads.s_layer_source_up[:, k] += source_up
            aads.s_layer_source_down[:, k] += source_down

        #######################################################################
        ## APPLY ADDING METHOD
        #######################################################################

        # this represents the "infinite" scattering between the two layers
        # combined

        infinite_scattering = identity_matrix - np.matmul(
            aads.s_level_refl_up[:, :, k + 1], s_layer_refl_down
        )

        infinite_scattering_inv = np.linalg.inv(infinite_scattering)

        aads.s_level_refl_up[:, :, k] = s_layer_refl_up + (
            s_layer_trans_down @ infinite_scattering_inv
        ) @ (aads.s_level_refl_up[:, :, k + 1] @ s_layer_trans_up)

        aads.refl_down[:, k] = np.matmul(
            aads.s_level_refl_up[:, :, k + 1], aads.s_layer_source_down[:, k]
        )

        aads.s_level_rad_up[:, k] = aads.s_layer_source_up[:, k] + np.matmul(
            (s_layer_trans_down @ infinite_scattering_inv),
            aads.refl_down[:, k] + aads.s_level_rad_up[:, k + 1],
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
