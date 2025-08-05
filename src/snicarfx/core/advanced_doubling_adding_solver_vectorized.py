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

        #################################################### ADDED OR MODIFIED
        self.mth_azi = 0
        self.planck_surface = 0

        self.direct_reflectivity = np.zeros(
            column.model_inputs.nb_angles, column.nbr_wvl
        )
        self.solar_irradiance = 2
        self.cos_sun = irradiance.cos_sza

        # two-streams: gaussian nodes/weights are simplified
        self.cos_angle = [1 / np.sqrt(3)]
        self.cos_weight = [1]

        # ! delta-eddington scale of variables to handle forward peak:
        tautot = column.tau
        wtot = column.ss_alb
        gtot = column.asm_prm
        ftot = gtot * gtot
        self.t_od = (1 - (wtot * ftot)) * tautot
        # layer delta-scaled single scattering albedo
        self.w = ((1 - ftot) * wtot) / (1 - (wtot * ftot))
        # layer delta-scaled asymmetry parameter
        self.g = gtot / (1 + gtot)

        # compute fowards and backward scattering phase matrix

        a = np.array(
            [
                [
                    0.5 + 1.5 * self.g * self.cos_angle[0] ** 2,
                    0.5 + 1.5 * self.g * self.cos_angle[0] * self.cos_sun,
                ]
            ]
        )
        self.ff = a

        b = np.array(
            [
                [
                    0.5 - 1.5 * self.g * self.cos_angle[0] ** 2,
                    0.5 - 1.5 * self.g * self.cos_angle[0] * self.cos_sun,
                ]
            ]
        )
        self.bb = b

        ######################################################################
        self.DELTA_OPTICAL_DEPTH = 1e-8
        self.max_albedo = 0.999999

        self.planck_atmosphere = np.zeros(column.nbr_lyr + 1)

        self.emissivity = np.zeros(column.model_inputs.nb_angles, column.nbr_wvl)

        self.SCATTERING_ALBEDO_THRESHOLD = 1e-10

        self.cosmic_background = 0

        self.solar_flag = True

        self.total_opt = np.zeros((column.nbr_lyr + 1, column.nbr_wvl))

        self.reflectivity = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )

        self.temporal_matrix = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        self.refl_down = np.zeros(
            (column.model_inputs.nb_angles, column.nbr_lyr, column.nbr_wvl)
        )

        self.inv_gamma = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_lyr,
                column.nbr_wvl,
            )
        )
        self.inv_gamma_t = np.zeros_like(self.inv_gamma)

        self.refl_trans = np.zeros_like(self.inv_gamma)

        self.s_layer_trans = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_lyr,
                column.nbr_wvl,
            )
        )
        self.s_layer_refl = np.zeros_like(self.s_layer_trans)
        self.s_level_refl_up = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_lyr + 1,
                column.nbr_wvl,
            )
        )
        self.s_level_rad_up = np.zeros(
            (column.model_inputs.nb_angles, column.nbr_lyr + 1, column.nbr_wvl)
        )
        self.s_layer_source_up = np.zeros(
            (column.model_inputs.nb_angles, column.nbr_lyr, column.nbr_wvl)
        )
        self.s_layer_source_down = np.zeros(
            (column.model_inputs.nb_angles, column.nbr_lyr, column.nbr_wvl)
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

        hh = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )

        eig_value = np.zeros((column.model_inputs.nb_angles, column.nbr_wvl))
        eig_va = np.zeros((column.model_inputs.nb_angles, column.nbr_wvl))
        eig_ve = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        eig_veva = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        eig_vef = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )

        gp = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        gm = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        i_gm = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        a1 = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        a2 = np.zeros_like(a1)
        a3 = np.zeros_like(a1)
        a4 = np.zeros_like(a1)
        a5 = np.zeros_like(a1)
        a6 = np.zeros_like(a1)
        gm_a5 = np.zeros_like(a1)
        i_gm_a5 = np.zeros_like(a1)

        exp_x = np.zeros(column.model_inputs.nb_angles)

        thermal_c = np.zeros((column.model_inputs.nb_angles, column.nbr_wvl))

        # these are the Legendre matrices built from ff and bb
        pp = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        pm = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        ppm = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        i_ppm = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )
        ppp = np.zeros(
            (
                column.model_inputs.nb_angles,
                column.model_inputs.nb_angles,
                column.nbr_wvl,
            )
        )

        # for small layer optical depth, single scattering is applied.
        if self.t_od[k] < self.DELTA_OPTICAL_DEPTH:

            s = self.t_od[k] * self.w[k]

            for i in range(column.model_inputs.nb_angles):

                thermal_c[i] = 0.0
                c = s / self.cos_angle[i]
                for j in range(column.model_inputs.nb_angles):

                    self.s_layer_refl[i, j, k] = (
                        c * self.bb[i, j, k] * self.cos_weight[j]
                    )
                    self.s_layer_trans[i, j, k] = (
                        c * self.ff[i, j, k] * self.cos_weight[j]
                    )

                    if i == j:
                        self.s_layer_trans[i, i, k] += (
                            1.0 - self.t_od[k] / self.cos_angle[i]
                        )
                    if self.mth_azi == 0:
                        thermal_c[i] += (
                            self.s_layer_refl[i, j, k] + self.s_layer_trans[i, j, k]
                        )

                if self.mth_azi == 0:
                    self.s_layer_source_up[i, k] = (
                        1.0 - thermal_c[i]
                    ) * self.planck_atmosphere
                    self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]

            return None

        # for numerical stability
        s = self.w[k] if self.w[k] < self.max_albedo else self.max_albedo

        # building phase matrices
        for i in range(column.model_inputs.nb_angles):
            c = s / self.cos_angle[i]

            for j in range(column.model_inputs.nb_angles):
                pm[i, j] = c * self.bb[i, j, k] * self.cos_weight[j]
                pp[i, j] = c * self.ff[i, j, k] * self.cos_weight[j]

            pp[i, i] -= 1.0 / self.cos_angle[i]

        ppm[:, :] = pp[:, :] - pm[:, :]

        try:
            i_ppm[:, :] = np.linalg.inv(ppm[:, :])
        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.ppm[:, :, {k}]): {e}"
            raise np.linalg.LinAlgError from e

        ppp[:, :] = pp[:, :] + pm[:, :]
        hh[:, :] = np.matmul(ppm[:, :], ppp[:, :])

        try:
            tempo = hh[:, :].copy()

            eig_vals, eig_vecs = np.linalg.eigh(tempo)

            eig_ve[:, :] = eig_vecs
            eig_va[:] = eig_vals

            eig_value[:] = np.where(eig_vals > 0.0, np.sqrt(eig_vals), 0.0)
            # eig_veva[i, j, k] = eig_ve[i, j, k] * eig_value[j, k]
            eig_veva[:, :] = eig_ve[:, :] * eig_value[np.newaxis, :]

            # eig_vef[:, :, k] = i_ppm @ eig_veva
            eig_vef[:, :] = np.matmul(i_ppm[:, :], eig_veva[:, :])

        except np.linalg.LinAlgError as e:
            print(f"Error in eigenvalue or matmul at layer {k}: {e}")
            raise

        # Compute layer reflection (Gp) and transmission (Gm) matrices
        try:
            gp[:, :] = (eig_ve[:, :] + eig_vef[:, :]) / 2.0
            gm[:, :] = (eig_ve[:, :] - eig_vef[:, :]) / 2.0

            i_gm[:, :] = np.linalg.inv(gm[:, :])

        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.gm[:, :, {k}], Error_Status): {e}"
            print(message)
            raise

        for i in range(column.model_inputs.nb_angles):
            xx = eig_value[i] * self.t_od[k]
            exp_x[i] = np.exp(-xx)

        for i in range(column.model_inputs.nb_angles):
            for j in range(column.model_inputs.nb_angles):
                a1[i, j] = gp[i, j] * exp_x[j]
                a4[i, j] = gm[i, j] * exp_x[j]

        a2[:, :] = np.matmul(i_gm[:, :], a1[:, :])
        a3[:, :] = np.matmul(gp[:, :], a2[:, :])
        a5[:, :] = np.matmul(a1[:, :], a2[:, :])
        a6[:, :] = np.matmul(a4[:, :], a2[:, :])

        gm_a5[:, :] = gm[:, :] - a5[:, :]

        try:
            i_gm_a5[:, :] = np.linalg.inv(gm_a5[:, :])
        except np.linalg.LinAlgError as e:
            message = f"Error in matinv(self.gm_a5[:, :, {k}], Error_Status): {e}"
            print(message)
            raise

        trans = np.matmul(a4[:, :] - a3[:, :], i_gm_a5[:, :])
        refl = np.matmul(gp[:, :] - a6[:, :], i_gm_a5[:, :])

        # post processing
        self.s_layer_trans[:, :, k] = trans[:, :]
        self.s_layer_refl[:, :, k] = refl[:, :]
        self.s_layer_source_up[:, k] = 0.0

        if self.mth_azi == 0:
            for i in range(column.model_inputs.nb_angles):
                thermal_c[i] = 0.0
                for j in range(column.model_inputs.nb_streams):
                    thermal_c[i] += trans[i, j] + refl[i, j]
                if (i == column.model_inputs.nb_angles - 1) and (
                    column.model_inputs.nb_angles
                    == (column.model_inputs.nb_streams + 1)
                ):
                    thermal_c[i] += trans[
                        column.model_inputs.nb_angles - 1,
                        column.model_inputs.nb_angles - 1,
                    ]
                self.s_layer_source_up[i, k] = (
                    1.0 - thermal_c[i]
                ) * self.planck_atmosphere[k]

                self.s_layer_source_down[i, k] = self.s_layer_source_up[i, k]

        # compute visible part for visible channels during daytime
        if self.solar_flag:
            n2 = 2 * column.model_inputs.nb_angles
            n2_1 = -1
            source_up = np.zeros(column.model_inputs.nb_angles)
            source_down = np.zeros(column.model_inputs.nb_angles)

            # solar source
            sfactor = self.w[k] * self.solar_irradiance / np.pi
            if self.mth_azi == 0:
                sfactor /= 2.0

            expfactor = np.exp(-self.t_od[k] / self.cos_sun)
            s_transmittance = np.exp(-self.total_opt[k] / self.cos_sun)

            solar = np.zeros(n2)
            v0 = np.zeros((n2, n2))

            for i in range(column.model_inputs.nb_angles):
                solar[i] = (
                    -self.bb[i, column.model_inputs.nb_angles, k] * sfactor
                )  # bb(i, nZ+1)
                solar[i + column.model_inputs.nb_angles] = (
                    -self.ff[i, column.model_inputs.nb_angles, k] * sfactor
                )  # ff(i, nZ+1)

                for j in range(column.model_inputs.nb_angles):
                    v0[i, j] = self.w[k] * self.ff[i, j, k] * self.cos_weight[j]
                    v0[i + column.model_inputs.nb_angles, j] = (
                        self.w[k] * self.bb[i, j, k] * self.cos_weight[j]
                    )
                    v0[i, j + column.model_inputs.nb_angles] = v0[
                        i + column.model_inputs.nb_angles, j
                    ]
                    v0[
                        column.model_inputs.nb_angles + i,
                        j + column.model_inputs.nb_angles,
                    ] = v0[i, j]

            v0[i, i] -= 1.0 + self.cos_angle[i] / self.cos_sun
            v0[
                i + column.model_inputs.nb_angles, i + column.model_inputs.nb_angles
            ] -= (1.0 - self.cos_angle[i] / self.cos_sun)

            # Invert v0 excluding last row/col (0-based: 0 to n2_1=-1)
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

            for i in range(column.model_inputs.nb_angles):
                source_up[i] = solar1[i]
                source_down[i] = expfactor * solar1[i + column.model_inputs.nb_angles]

                for j in range(column.model_inputs.nb_angles):
                    source_up[i] -= (
                        refl[i, j] * solar1[j + column.model_inputs.nb_angles]
                        + trans[i, j] * expfactor * solar1[j]
                    )
                    source_down[i] -= (
                        trans[i, j] * solar1[j + column.model_inputs.nb_angles]
                        + refl[i, j] * expfactor * solar1[j]
                    )

            # Specific treatment for downward source function
            if abs(v0[n2 - 1, n2 - 1]) > 1e-4:
                source_down[column.model_inputs.nb_angles - 1] += (
                    (
                        expfactor
                        - trans[
                            column.model_inputs.nb_angles - 1,
                            column.model_inputs.nb_angles - 1,
                        ]
                    )
                    * sfac2
                    / v0[n2 - 1, n2 - 1]
                )
            else:
                source_down[column.model_inputs.nb_angles - 1] -= (
                    expfactor
                    * sfac2
                    * self.t_od[k]
                    / self.cos_angle[column.model_inputs.nb_angles - 1]
                )

            source_up *= s_transmittance
            source_down *= s_transmittance

            self.s_layer_source_up[:, k] += source_up
            self.s_layer_source_down[:, k] += source_down

        return None


def solve_advanced_adding_doubling(column, irradiance):
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

    if True:
        return aads

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
            * aads.solar_irradiance
            / np.pi
            * np.exp(-aads.total_opt[-1, :] / aads.cos_sun)
        )

    for k in range(column.nbr_lyr - 1, -1, -1):

        if aads.w[k] > aads.SCATTERING_ALBEDO_THRESHOLD:

            # call  multiple-stream algorithm for computing layer
            # transmission, reflection, and source functions.
            aads.crtm_anom_layer(column, k)
            # then Adding method to add the layer to the present level
            # to compute upward radiances and reflection matrix
            # at new level.

            # similar to equation B4 Briegleb and Light 2007
            # moveaxis back and forth because matmul expects fourth
            # dimension at the front
            temporal_matrix = -np.moveaxis(
                np.matmul(
                    np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :], -1, 0),
                    np.moveaxis(aads.s_layer_refl[:, :, k, :], -1, 0),
                ),
                0,
                -1,
            )

            for b in range(temporal_matrix.shape[2]):
                np.fill_diagonal(
                    temporal_matrix[:, :, b], temporal_matrix[:, :, b].diagonal() + 1.0
                )

            try:
                # here again, moveaxis back and forth because function
                # expects batch dimension to be first
                aads.inv_gamma[:, :, k, :] = np.moveaxis(
                    np.linalg.inv(np.moveaxis(temporal_matrix, -1, 0)), 0, -1
                )
            except np.linalg.LinAlgError as e:
                print(f"Error in matrix inversion matinv(temporal_matrix): {e}")
                raise

            aads.inv_gamma_t[:, :, k, :] = np.moveaxis(
                np.matmul(
                    np.moveaxis(aads.s_layer_trans[:, :, k, :], -1, 0),
                    np.moveaxis(aads.inv_gamma[:, :, k, :], -1, 0),
                ),
                0,
                -1,
            )

            aads.refl_down[:, k, :] = np.moveaxis(
                np.matmul(
                    np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :], -1, 0),
                    np.moveaxis(aads.s_layer_source_down[:, k, :], -1, 0)[..., None],
                ),
                0,
                -1,
            )[..., 0]

            aads.s_level_rad_up[:, k, :] = (
                aads.s_layer_source_up[:, k, :]
                + np.moveaxis(
                    np.matmul(
                        np.moveaxis(aads.inv_gamma_t[:, :, k, :], -1, 0),
                        np.moveaxis(
                            aads.refl_down[:, k, :] + aads.s_level_rad_up[:, k + 1, :],
                            -1,
                            0,
                        )[..., None],
                    ),
                    0,
                    -1,
                )[..., 0]
            )

            aads.refl_trans[:, :, k, :] = np.moveaxis(
                np.matmul(
                    np.moveaxis(aads.s_level_refl_up[:, :, k + 1, :], -1, 0),
                    np.moveaxis(aads.s_layer_trans[:, :, k, :], -1, 0),
                ),
                0,
                -1,
            )

            aads.s_level_refl_up[:, :, k, :] = aads.s_layer_refl[
                :, :, k, :
            ] + np.moveaxis(
                np.matmul(
                    np.moveaxis(aads.inv_gamma_t[:, :, k, :], -1, 0),
                    np.moveaxis(aads.refl_trans[:, :, k, :], -1, 0),
                ),
                0,
                -1,
            )

        else:  # we dont enter in this loop anyway
            print("a")

    if aads.mth_azi == 0:
        for i in range(column.model_inputs.nb_angles):
            aads.s_level_rad_up[i, 0, :] += (
                np.sum(aads.s_level_refl_up[i, :, 0, :], axis=0)
                * aads.cosmic_background
            )

    return aads
