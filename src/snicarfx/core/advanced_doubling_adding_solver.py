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
        """ """

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

        return None

    def solve_advanced_adding_doubling(self, column, irradiance):
        """
        Solve the ADA equations
        """

        aads = _AdvancedDoublingAddingSolver(column, irradiance)

        aads.total_opt[1:] = np.cumsum(aads.t_od)

        aads.s_level_refl_up[:, :, -1] = aads.reflectivity

        if not aads.mth_azi:
            aads.s_level_rad_up[:, -1] = aads.emissivity * aads.planck_surface

        # adds a solar reflection term to the upward radiance at the
        # last layer for all viewing angles
        aads.s_level_rad_up[:, -1] += (
            aads.direct_reflectivity
            * aads.cos_sun
            * irradiance.flx_slr
            / np.pi
            * np.exp(-aads.total_opt[-1] / aads.cos_sun)
        )

        for k in range(column.nbr_lyr, 0, -1):

            if aads.w[k] > aads.SCATTERING_ALBEDO_THRESHOLD:
                aads.crtm_anom_layer(k)

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

        return None

    def crtm_anom_layer(self):
        return None
