#!/usr/bin/env python3
"""

Implementation of the Advanced Doubling Adding Method for Radiative
Transfer in Planetary Atmospheres


Quanhua Liu and Fuzhong Weng, 2006
https://doi.org/10.1175/JAS3808.1

"""

import numpy as np


class _AdvancedDoublingAddingSolver:

    def __init__(self, column, irradiance):
        """
        Initialize all variables required for the ADA solver
        """

        self.w = np.zeros(column.nbr_lyr)
        self.t_od = np.zeros(column.nbr_lyr)

        self.emissivity = np.zeros(column.nbr_lyr)
        self.direct_reflectivity = np.zeros(column.nbr_lyr)

        self.reflectivity = np.zeros(
            column.modelinputs.nb_angles, column.modelinputs.nb_angles
        )

        self.cosmic_background = 0

        self.temporal_matrix = np.zeros(
            column.modelinputs.nb_angles, column.modelinputs.nb_angles
        )
        self.refl_down = np.zeros(column.modelinputs.nb_angles)

        self.total_opt = np.zeros(column.nbr_lyr + 1)

        self.s_layer_trans = self.s_level_refl_up = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.s_layer_refl = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.s_level_refl_up = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr + 1,
        )
        self.s_level_rad_up = np.zeros(column.modelinputs.nb_angles, column.nbr_lyr + 1)
        self.s_layer_source_up = np.zeros(column.modelinputs.nb_angles, column.nbr_lyr)
        self.s_layer_source_down = np.zeros(
            column.modelinputs.nb_angles, column.nbr_lyr
        )

        self.emissivity = np.zeros(column.modelinputs.nb_angles)

        self.cos_sun = irradiance.cos_sza

        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10

        self.inv_gamma = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.inv_gamma_t = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        self.cos_angle = np.zeros(column.modelinputs.nb_angles)
        self.cos_weight = np.zeros(column.modelinputs.nb_angles)

        self.solar_flag = True

        self.ff = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles + 1,
            column.nbr_lyr,
        )
        self.bb = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles + 1,
            column.nbr_lyr,
        )

        self.DELTA_OPTICAL_DEPTH = 1e-8

        self.planck_atmosphere = np.zeros(column.nbr_lyr + 1)

        self.thermal_c = np.zeros(
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        self.max_albedo = 0.999999

        self.pp = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.pm = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.ppm = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.i_ppm = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.ppp = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.hh = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        self.eig_value = np.zeros(
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.eig_va = np.zeros(
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.eig_ve = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.eig_veva = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.eig_vef = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        self.gp = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.gm = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.i_gm = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a1 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a2 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a3 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a4 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a5 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.a6 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.gm_a5 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )
        self.i_gm_a5 = np.zeros(
            column.modelinputs.nb_angles,
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        self.exp_x = np.zeros(
            column.modelinputs.nb_angles,
            column.nbr_lyr,
        )

        return None

    def solve_advanced_adding_doubling(self, column, irradiance):
        """

        This subroutine calculates IR/MW radiance at the top of the atmosphere
        including atmospheric scattering. The scheme will include solar part.
        The ADA algorithm computes layer reflectance and transmittance as well
        as source function by the subroutine CRTM_Doubling_layer, then uses
        an adding method to integrate the layer and surface components.

        Translated by the snicar-fx team from the Fortran code of
        Quanhua Liu (Quanhua.Liu@noaa.gov)

        """

        aads = _AdvancedDoublingAddingSolver(column, irradiance)

        aads.total_opt[1:] = np.cumsum(aads.t_od)

        aads.s_level_refl_up[:, :, -1] = aads.reflectivity

        if aads.mth_azi == 0:
            aads.s_level_rad_up[:, -1] = aads.emissivity * aads.planck_surface

        # adds a solar reflection term to the upward radiance at the
        # last layer for all viewing angles
        if aads.solar_flag:
            aads.s_level_rad_up[:, -1] += (
                aads.direct_reflectivity
                * aads.cos_sun
                * irradiance.flx_slr
                / np.pi
                * np.exp(-aads.total_opt[-1] / aads.cos_sun)
            )

        for k in range(column.nbr_lyr, 0, -1):

            if aads.w[k] > aads.SCATTERING_ALBEDO_THRESHOLD:

                # call  multiple-stream algorithm for computing layer
                # transmission, reflection, and source functions.
                aads.crtm_anom_layer(k)

                # then Adding method to add the layer to the present level
                # to compute upward radiances and reflection matrix
                # at new level.

                # similar to equation B4 Briegleb and Light 2007
                temporal_matrix = -np.matmul(
                    aads.s_level_refl_up[:, :, k],
                    aads.s_layer_refl[:, :, k],
                )

                np.fill_diagonal(temporal_matrix, temporal_matrix.diagonal() + 1.0)
                try:
                    aads.Inv_Gamma[:, :, k] = np.linalg.inv(temporal_matrix)
                except np.linalg.LinAlgError as e:
                    print(f"Error in matrix inversion matinv(temporal_matrix): {e}")
                    raise

                aads.inv_gamma_t[:, :, k] = np.matmul(
                    aads.s_layer_trans[:, :, k], aads.inv_gamma[:, :, k]
                )

                aads.refl_down[:, k] = np.matmul(
                    aads.s_level_refl_up[:, :, k], aads.s_layer_source_down[:, k]
                )

                aads.s_level_rad_up[:, k - 1] = aads.s_layer_source_up[
                    :, k
                ] + np.matmul(
                    aads.inv_gamma_t[:, :, k],
                    aads.refl_down[:, k] + aads.s_level_rad_up[:, k],
                )

                aads.refl_trans[:, :, k] = np.matmul(
                    aads.s_level_refl_up[:, :, k], aads.s_layer_trans[:, :, k]
                )

                aads.s_level_refl_up[:, :, k - 1] = aads.s_layer_refl[
                    :, :, k
                ] + np.matmul(aads.inv_gamma_t[:, :, k], aads.refl_trans[:, :, k])

            else:

                for i in range(column.modelinputs.nb_angles):
                    aads.s_layer_trans[i, i, k] = np.exp(
                        -aads.t_od[k] / aads.cos_angle[i]
                    )
                    aads.s_layer_source_up[i, k] = aads.planck_atmosphere[k] * (
                        1.0 - aads.s_layer_trans[i, i, k]
                    )
                    aads.s_layer_source_down[i, k] = aads.s_layer_source_up[i, k]

                for i in range(column.modelinputs.nb_angles):
                    weighted_sum = np.sum(
                        aads.s_level_refl_up[i, :, k] * aads.s_layer_source_down[:, k]
                    )
                    aads.s_level_rad_up[i, k - 1] = aads.s_layer_source_up[
                        i, k
                    ] + aads.s_layer_trans[i, i, k] * (
                        weighted_sum + aads.s_level_rad_up[i, k]
                    )

                for i in range(column.modelinputs.nb_angles):
                    for j in range(column.modelinputs.nb_angles):
                        aads.s_level_refl_up[i, j, k - 1] = (
                            aads.s_layer_trans[i, i, k]
                            * aads.s_level_refl_up[i, j, k]
                            * aads.s_layer_trans[j, j, k]
                        )

        if aads.mth_azi == 0:
            for i in range(aads.n_angles):
                aads.s_level_rad_up[i, 0] += (
                    np.sum(aads.s_level_refl_up[i, :, 0]) * aads.cosmic_background
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
        # for small layer optical depth, single scattering is applied.
        if self.t_od[k] < self.delta_self.t_od[k]:

            s = self.t_od[k] * self.w[k]

            for i in range(column.modelinputs.nb_angles):

                self.thermal_C[i, k] = 0.0
                c = s / self.cos_angle[i]
                for j in range(column.modelinputs.nb_angles):
                    self.s_layer_refl[i, j, k] = c * self.bb[i, j] * self.cos_weight[j]
                    self.s_layer_trans[i, j, k] = c * self.ff[i, j] * self.cos_weight[j]
                    if i == j:
                        self.s_layer_trans[i, i, k] += (
                            1.0 - self.t_od[k] / self.cos_angle[i]
                        )
                    if self.mth_Azi == 0:
                        self.thermal_c[i, k] += (
                            self.s_layer_refl[i, j, k] + self.s_layer_trans[i, j, k]
                        )

                if self.mth_azi == 0:
                    self.s_layer_source_up[i, k] = (
                        1.0 - self.thermal_C[i, k]
                    ) * self.planck_atmosphere
                    self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]

            return None

        # for numerical stability
        s = self.w[k] if self.w[k] < self.max_albedo else self.max_albedo

        # building phase matrices
        for i in range(column.modelinputs.nb_angles):
            c = s / self.cos_angle[i]

            for j in range(column.modelinputs.nb_angles):
                self.pm[i, j, k] = c * self.bb[i, j] * self.cos_weight[j]
                self.pp[i, j, k] = c * self.ff[i, j] * self.cos_weight[j]

            self.pp[i, i, k] -= 1.0 / self.cos_angle[i]

        self.ppm[:, :, k] = self.pp[:, :, k] - self.pm[:, :, k]

        try:
            self.i_ppm[:, :, k] = np.linalg.inv(self.ppm[:, :, k])
        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.ppm[:, :, {k}]): {e}"
            raise np.linalg.LinAlgError from e

        self.ppp[:, :, k] = self.pp[:, :, k] + self.pm[:, :, k]
        self.hh[:, :, k] = np.matmul(self.ppm[:, :, k], self.ppp[:, :, k])

        try:
            tempo = self.hh[:, :, k].copy()

            eig_vals, eig_vecs = np.linalg.eigh(tempo)

            self.eig_ve[:, :, k] = eig_vecs
            self.eig_va[:, k] = eig_vals

            self.eig_value[:, k] = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)
            # eig_veva[i, j, k] = eig_ve[i, j, k] * eig_value[j, k]
            self.eig_veva[:, :, k] = (
                self.eig_ve[:, :, k] * self.eig_value[np.newaxis, :, k]
            )

            # eig_vef[:, :, k] = i_ppm @ eig_veva
            self.eig_vef[:, :, k] = np.matmul(
                self.i_ppm[:, :, k], self.eig_veva[:, :, k]
            )

        except np.linalg.LinAlgError as e:
            print(f"Error in eigenvalue or matmul at layer {k}: {e}")
            raise

        # Compute layer reflection (Gp) and transmission (Gm) matrices
        try:
            self.gp[:, :, k] = (self.eig_ve[:, :, k] + self.eig_vef[:, :, k]) / 2.0
            self.gm[:, :, k] = (self.eig_ve[:, :, k] - self.eig_vef[:, :, k]) / 2.0

            self.i_gm[:, :, k] = np.linalg.inv(self.gm[:, :, k])

        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.gm[:, :, {k}], Error_Status): {e}"
            print(message)
            raise

        for i in range(column.modelinputs.nb_angles):
            xx = self.eig_value[i, k] * self.t_od[k]
            self.exp_x[i, k] = np.exp(-xx)

        for i in range(column.modelinputs.nb_angles):
            for j in range(column.modelinputs.nb_angles):
                self.a1[i, j, k] = self.gp[i, j, k] * self.exp_x[j, k]
                self.a4[i, j, k] = self.gm[i, j, k] * self.exp_x[j, k]

        self.a2[:, :, k] = np.matmul(self.i_gm[:, :, k], self.a1[:, :, k])
        self.a3[:, :, k] = np.matmul(self.gp[:, :, k], self.a2[:, :, k])
        self.a5[:, :, k] = np.matmul(self.a1[:, :, k], self.a2[:, :, k])
        self.a6[:, :, k] = np.matmul(self.a4[:, :, k], self.a2[:, :, k])

        self.gm_a5[:, :, k] = self.gm[:, :, k] - self.a5[:, :, k]

        try:
            self.i_gm_a5[:, :, k] = np.linalg.inv(self.gm_a5[:, :, k])
        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.gm_a5[:, :, {k}], Error_Status): {e}"
            print(message)
            raise

        trans = np.matmul(self.a4[:, :, k] - self.a3[:, :, k], self.i_gm_a5[:, :, k])
        refl = np.matmul(self.gp[:, :, k] - self.a6[:, :, k], self.i_gm_a5[:, :, k])

        # post processing
        self.s_layer_trans[:, :, k] = trans[:, :]
        self.s_layer_refl[:, :, k] = refl[:, :]
        self.s_layer_source_up[:, k] = 0.0

        if self.mth_azi == 0:
            for i in range(column.modelinputs.nb_angles):
                self.thermal_c[i, k] = 0.0
                for j in range(column.modelinputs.nb_streams):
                    self.thermal_c[i, k] += trans[i, j] + refl[i, j]
            if i == self.nz - 1 and self.nz == (self.n_streams + 1):
                self.thermal_c[i, k] += trans[self.nz - 1, self.nz - 1]

            self.s_layer_source_up[i, k] = (
                1.0 - self.thermal_c[i, k]
            ) * self.planck_atmosphere
            self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]

        # compute visible part for visible channels during daytime
        if self.solar_flag_true:
            n2 = 2 * column.modelinputs.nb_angles
            n2_1 = n2 - 1
            source_up = np.zeros(column.modelinputs.nb_angles)
            source_down = np.zeros(column.modelinputs.nb_angles)

            # solar source
            sfactor = self.w[k] * self.solar_irradiance / np.pi
            if self.mth_azi == 0:
                sfactor /= 2.0

            expfactor = np.exp(-self.t_od[k] / self.cos_sun)
            s_transmittance = np.exp(-self.total_opt / self.cos_sun)

            solar = np.zeros(n2)
            v0 = np.zeros((n2, n2))

            for i in range(column.modelinputs.nb_angles):
                solar[i] = (
                    -self.bb[i, column.modelinputs.nb_angles] * sfactor
                )  # bb(i, nZ+1)
                solar[i + column.modelinputs.nb_angles] = (
                    -self.ff[i, column.modelinputs.nb_angles] * sfactor
                )  # ff(i, nZ+1)

                for j in range(column.modelinputs.nb_angles):
                    v0[i, j] = self.w[k] * self.ff[i, j] * self.cos_weight[j]
                    v0[i + column.modelinputs.nb_angles, j] = (
                        self.w[k] * self.bb[i, j] * self.cos_weight[j]
                    )
                    v0[i, j + column.modelinputs.nb_angles] = v0[
                        i + column.modelinputs.nb_angles, j
                    ]
                    v0[
                        column.modelinputs.nb_angles + i,
                        j + column.modelinputs.nb_angles,
                    ] = v0[i, j]

            v0[i, i] -= 1.0 + self.cos_angle[i] / self.cos_sun
            v0[i + column.modelinputs.nb_angles, i + column.modelinputs.nb_angles] -= (
                1.0 - self.cos_angle[i] / self.cos_sun
            )

            # Invert v0 excluding last row/col (0-based: 0 to n2_1-1)
            try:
                v1 = np.linalg.inv(v0[:n2_1, :n2_1])
            except np.linalg.LinAlgError:
                message = (
                    "Error in matrix inversion matinv(v0(1:N2_1,1:N2_1), Error_Status)"
                )
                print(message)
                raise

            solar1 = v1 @ solar[:n2_1]
            solar1 = np.append(solar1, 0.0)  # solar1(N2) = 0.0

            sfac2 = solar[n2 - 1] - np.sum(v0[n2 - 1, :n2_1] * solar1[:n2_1])

            for i in range(column.modelinputs.nb_angles):
                source_up[i] = solar1[i]
                source_down[i] = expfactor * solar1[i + column.modelinputs.nb_angles]

                for j in range(column.modelinputs.nb_angles):
                    source_up[i] -= (
                        refl[i, j] * solar1[j + column.modelinputs.nb_angles]
                        + trans[i, j] * expfactor * solar1[j]
                    )
                    source_down[i] -= (
                        trans[i, j] * solar1[j + column.modelinputs.nb_angles]
                        + refl[i, j] * expfactor * solar1[j]
                    )

            # Specific treatment for downward source function
            if abs(v0[n2 - 1, n2 - 1]) > 1e-4:
                source_down[column.modelinputs.nb_angles - 1] += (
                    (
                        expfactor
                        - trans[
                            column.modelinputs.nb_angles - 1,
                            column.modelinputs.nb_angles - 1,
                        ]
                    )
                    * sfac2
                    / v0[n2 - 1, n2 - 1]
                )
            else:
                source_down[column.modelinputs.nb_angles - 1] -= (
                    expfactor
                    * sfac2
                    * self.t_od[k]
                    / self.cos_angle[column.modelinputs.nb_angles - 1]
                )

            source_up *= s_transmittance
            source_down *= s_transmittance

            self.s_layer_source_up[:, k] += source_up
            self.s_layer_source_down[:, k] += source_down

        return None
