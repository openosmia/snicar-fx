#!/usr/bin/python
"""Radiative transfer solver using the adding-doubling method.

Here the ice/impurity optical properties and irradiance conditions
are used to calculate energy fluxes between the ice, atmosphere and
underlying substrate.

Typically this function would be called from snicar_driver() because
it takes as inputs intermediates that are calculated elsewhere.
Specifically, the functions setup_snicar(), get_layer_OPs() and
mix_in_impurities() are called to generate tau, ssa, g and L_snw,
which are then passed as inputs to adding_doubling_solver().

The adding-doubling routine implemented here originates in Brieglib
and Light (2007) and was coded up in Matlab by Chloe Whicker and Mark
Flanner to be published in Whicker (2022: The Cryosphere). Their
scripts were the jumping off point for this script and their code is still
used to benchmark this script against.

The adding-doubling solver implemented here has been shown to do an excellent
job at simulating solid glacier column. This solver can either treat ice as a
"granular" material with a bulk medium of air with discrete ice grains, or
as a bulk medium of ice with air inclusions. In the latter case, the upper
boundary is a Fresnel reflecting surface. Total internal reflection is accounted
for if the irradiance angles exceed the critical angle.

This is always the appropriate solver to use in any  model configuration where
solid ice layers and fresnel reflection are included.

"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Outputs:
    """

    Output data from radiative transfer calculations.

    Attributes:
        heat_rt: Heating rate in each layer.
        BBAVIS: Broadband albedo in visible range.
        BBANIR: Broadband albedo in NIR range.
        BBA: Broadband albedo across solar spectrum.
        abs_slr_btm: Absorbed solar energy at bottom surface.
        abs_vis_btm: Absorbed visible energy at bottom surface.
        abs_nir_btm: Absorbed NIR energy at bottom surface.
        albedo: Albedo of ice column.
        total_insolation: Energy arriving from atmosphere.
        abs_slr_tot: Total absorbed energy across solar spectrum.
        abs_vis_tot: Total absorbed energy across visible spectrum.
        abs_nir_tot: Total absorbed energy across NIR spectrum.
        absorbed_flux_per_layer: Total absorbed flux per layer.
    """

    heat_rt: Any | None = None
    BBAVIS: float | None = None
    BBANIR: float | None = None
    BBA: float | None = None
    abs_slr_btm: float | None = None
    abs_vis_btm: float | None = None
    abs_nir_btm: float | None = None
    albedo: float | None = None
    total_insolation: float | None = None
    abs_slr_tot: float | None = None
    abs_vis_tot: float | None = None
    abs_nir_tot: float | None = None
    absorbed_flux_per_layer: Any | None = None


class _AddingDoublingSolver:

    def __init__(self, column, irradiance):
        """
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class
        tau0: initial optical thickness (m/m)
        g0: initial asymmetry parameter (dimensionless)
        ssa0: initial single scatterign albedo (dimensionless)
        epsilon: small number to avoid singularity
        exp_min: small number to avoid zero calcs
        nr: real part of refractive index
        mu0: cosine of direct beam zenith angle
        mu0n: adjusted cosine of direct beam zenith angle after refraction
        trnlay: direct transmission of solar beam through layer
        rdif_a: reflectivity to diffuse irradiance coming from above
        rdif_b: reflectivity to diffuse irradiance coming from bloe
        tdif_a: transmissivity to diffuse irradiance coming from above
        tdif_b: transmissivity to diffuse irradiance coming from below
        rdir: reflectivity to direct beam
        tdir: total transmission of the direct beam (direct + diffuse)
        lyrfrsnl: index of uppermost fresnel reflecting layer in ice column

        ws: layer delta-scaled single scattering albedo
        gs:layer delta-scaled asymmetry parameter

        smt: accumulator for tdif gaussian integration
        smr: accumulator for rdif gaussian integration
        swt: sum of gaussian weights

        albedo: ratio of upwards fluxes to incoming irradiance
        F_abs: absorbed flux in each layer
        F_btm_net: net fluxes at bottom surface
        F_top_pls: upwards flux from upper surface
        """

        self.column = column
        self.irradiance = irradiance

        # read and transpose tau
        self.tau0 = column.tau.T

        # read and transpose g
        self.g0 = column.asm_prm.T

        # read and transpose ssa
        self.ssa0 = column.ss_alb.T

        # to deal with singularity
        self.epsilon = 1e-5

        # exp(-500)  # min value > 0 to avoid error
        self.exp_min = 1e-5

        self.nr = np.zeros(shape=480)

        # cos beam angle = incident beam
        self.mu0 = irradiance.mu_not * np.ones(480)

        # ice-adjusted real refractive index
        temp1 = (
            column.ref_idx_re**2
            - column.ref_idx_im**2
            + np.sin(np.arccos(irradiance.mu_not)) ** 2
        )
        temp2 = (
            column.ref_idx_re**2
            - column.ref_idx_im**2
            - np.sin(np.arccos(irradiance.mu_not)) ** 2
        )
        self.nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * column.ref_idx_re**2 * column.ref_idx_im**2) ** 0.5
        ) ** 0.5

        # . Eq. 20: Briegleb and Light 2007: adjusts beam angle
        # (i.e. this is Snell's Law for refraction at interface between media)
        # mu0n = -1 represents light travelling vertically upwards and mu0n = +1
        # represents light travellign vertically downwards
        # mu0n = np.sqrt(1-((1-mu0**2)/(ref_indx*ref_indx)))  (original,
        # before update for diffuse Fresnel reflection)
        # this version accounts for diffuse fresnel reflection:
        self.mu0n = np.cos(np.arcsin(np.sin(np.arccos(self.mu0)) / self.nr))

        # solar beam transm for layer (direct beam only)
        self.trnlay = np.zeros(shape=[column.nbr_wvl, column.nbr_lyr + 1])

        # layer reflectivity to diffuse radiation from above
        self.rdif_a = np.zeros_like(self.trnlay)

        # layer reflectivity to diffuse radiation from below
        self.rdif_b = np.zeros_like(self.trnlay)

        # layer transmittivity to diffuse radiation from above
        self.tdif_a = np.zeros_like(self.trnlay)

        # layer transmittivity to diffuse radiation from below
        self.tdif_b = np.zeros_like(self.trnlay)

        # layer reflectivity to direct radiation (solar beam + diffuse)
        self.rdir = np.zeros_like(self.trnlay)

        # layer transmittivity to direct radiation (solar beam + diffuse)
        self.tdir = np.zeros_like(self.trnlay)

        # reflection of diffuse radiation for layers above
        self.rdndif = np.zeros_like(self.trnlay)

        # total transmission from layers above
        self.trntdr = np.zeros_like(self.trnlay)

        # diffuse transmission for layers above
        self.trndif = np.zeros_like(self.trnlay)

        # solar beam down transmission from top
        self.trndir = np.zeros_like(self.trnlay)

        # reflectivity to diffuse radiation
        self.rupdif = np.zeros_like(self.trnlay)

        # reflectivity to direct radiation
        self.rupdir = np.zeros_like(self.trnlay)

        self.fdirup = np.zeros_like(self.trnlay)
        self.fdifup = np.zeros_like(self.trnlay)
        self.fdirdn = np.zeros_like(self.trnlay)
        self.fdifdn = np.zeros_like(self.trnlay)
        self.dfdir = np.zeros_like(self.trnlay)
        self.dfdif = np.zeros_like(self.trnlay)
        self.F_up = np.zeros_like(self.trnlay)
        self.F_dwn = np.zeros_like(self.trnlay)

        self.F_abs = np.zeros(shape=[column.nbr_wvl, column.nbr_lyr])
        self.F_abs_vis = np.zeros(shape=[column.nbr_lyr])
        self.F_abs_nir = np.zeros(shape=[column.nbr_lyr])

        # if there are non zeros in layer type, grab the index of the
        # first fresnel layer and load in the precalculated diffuse fresnel
        # reflection
        # (precalculated as large no. of gaussian points required for convergence)
        if np.sum(np.array(column.layer_type) == 1) > 0:
            self.lyrfrsnl = column.layer_type.index(1)

        else:
            self.lyrfrsnl = 9999999

        # gaussian angles (radians)
        self.GAUSPT = [
            0.9894009,
            0.9445750,
            0.8656312,
            0.7554044,
            0.6178762,
            0.4580168,
            0.2816036,
            0.0950125,
        ]
        # gaussian weights
        self.GAUSWT = [
            0.0271525,
            0.0622535,
            0.0951585,
            0.1246290,
            0.1495960,
            0.1691565,
            0.1826034,
            0.1894506,
        ]

        return None

    def calculate_reflectivity_transmittivity_delta_eddington(self, lyr):
        """Calculates multiple scattering within a given layer to yield
        reflectivity and transmissivity of the layer to
        direct and diffuse radiation, using the Delta-Eddington solution.
        Eq. A24, A26, A30, A31 Briegleb and Light 2007

        Sets up new variables, applies delta transformation and makes
        initial calculations of direct reflectivity and transmissivity in each
        layer.
        """

        # calculation over layers with penetrating radiation
        # includes optical thickness, single scattering albedo,
        # asymmetry parameter and total flux
        tautot = self.tau0[:, lyr]
        wtot = self.ssa0[:, lyr]
        gtot = self.g0[:, lyr]
        ftot = self.g0[:, lyr] * self.g0[:, lyr]

        # coefficient for delta eddington solution for all layers
        # Eq. 50: Briegleb and Light 2007
        # layer delta-scaled extinction optical depth
        self.ts = (1 - (wtot * ftot)) * tautot

        # layer delta-scaled single scattering albedo
        self.ws = ((1 - ftot) * wtot) / (1 - (wtot * ftot))

        # layer delta-scaled asymmetry parameter
        self.gs = gtot / (1 + gtot)

        # lambda
        lm = np.sqrt(3 * (1 - self.ws) * (1 - self.ws * self.gs))

        # u equation, term in diffuse reflectivity and transmissivity
        ue = 1.5 * (1 - self.ws * self.gs) / lm

        # extinction, MAX function lyr keeps from getting an error
        # if the exp(-lm*ts) is < 1e-5
        extins = np.maximum(
            np.full((self.column.nbr_wvl,), self.exp_min), np.exp(-lm * self.ts)
        )

        # N equation, term in diffuse reflectivity and transmissivity
        ne = (ue + 1) ** 2 / extins - (ue - 1) ** 2 * extins

        # calculation of rdif, tdif using Delta-Eddington formulas
        # Eq. A30 and A31 Briegleb and Light 2007
        # note that rdif and tdif are used for rdir, tdir calculations here
        # but are recalculated with gaussian integration later
        self.rdif_a[:, lyr] = (ue**2 - 1) * (1 / extins - extins) / ne
        self.tdif_a[:, lyr] = 4 * ue / ne

        # evaluate rdir, tdir for direct beam
        # transmission from TOA to interface
        self.trnlay[:, lyr] = np.maximum(
            np.full((self.column.nbr_wvl,), self.exp_min), np.exp(-self.ts / self.mu0n)
        )

        #  Eq. 50: Briegleb and Light 2007  alpha and gamma for direct radiation
        # alp = alpha(ws,mu0n,gs,lm)
        alp = (
            (0.75 * self.ws * self.mu0n)
            * (1 + self.gs * (1 - self.ws))
            / (1 - lm**2 * self.mu0n**2 + self.epsilon)
        )
        # gam = gamma(ws,mu0n,gs,lm)
        gam = (0.5 * self.ws) * (
            (1 + 3 * self.gs * self.mu0n**2 * (1 - self.ws))
            / (1 - lm**2 * self.mu0n**2 + self.epsilon)
        )

        # apg = alpha plus gamma
        # amg = alpha minus gamma
        apg = alp + gam
        amg = alp - gam

        # layer reflectivity to DIRECT radiation
        self.rdir[:, lyr] = apg * self.rdif_a[:, lyr] + amg * (
            self.tdif_a[:, lyr] * self.trnlay[:, lyr] - 1
        )

        # layer transmissivity to DIRECT radiation
        self.tdir[:, lyr] = (
            apg * self.tdif_a[:, lyr]
            + (amg * self.rdif_a[:, lyr] - apg + 1) * self.trnlay[:, lyr]
        )

        return None

    def apply_gaussian_integral(self, lyr):
        """Applies gaussian integral to integrate over angles.

        Uses gaussien integration to integrate fluxes hemispherically from
        N of reference angles where N = len(gauspt) (default is 8).
        """

        self.swt = 0
        self.smr = 0
        self.smt = 0

        # use r1 as temporary
        r1 = self.rdif_a[:, lyr].copy()

        # use t1 as temporary
        t1 = self.tdif_a[:, lyr].copy()

        for ng in np.arange(0, len(self.GAUSPT), 1):

            # solar zenith angles
            mu = self.GAUSPT[ng]

            # gaussian weight
            gwt = self.GAUSWT[ng]

            # sum of weights
            self.swt = self.swt + mu * gwt

            # transmission
            trn = np.maximum(
                np.full((self.column.nbr_wvl,), self.exp_min), np.exp(-self.ts / mu)
            )
            lm = np.sqrt(3 * (1 - self.ws) * (1 - self.ws * self.gs))

            # alp = alpha(ws,mu0n,gs,lm)
            alp = (
                (0.75 * self.ws * mu)
                * (1 + self.gs * (1 - self.ws))
                / (1 - lm**2 * mu**2 + self.epsilon)
            )
            gam = (
                (0.5 * self.ws)
                * (1 + 3 * self.gs * mu**2 * (1 - self.ws))
                / (1 - lm**2 * mu**2 + self.epsilon)
            )  # gam = gamma(ws,mu0n,gs,lm)

            apg = alp + gam
            amg = alp - gam
            rdr = apg * r1 + amg * t1 * trn - amg
            tdr = apg * t1 + amg * r1 * trn - apg * trn + trn

            # accumulator for rdif gaussian integration
            self.smr = self.smr + mu * rdr * gwt

            # accumulator for tdif gaussian integration
            self.smt = self.smt + mu * tdr * gwt

        return None

    def calculate_diff_transmittivity_reflectivity(self, lyr):
        """
        Calculate transmissivity and reflectivity to DIFFUSE radiation
        after gaussian integration, eq. A33 Briegleb and Light 2007.
        """
        self.rdif_a[:, lyr] = self.smr / self.swt
        self.tdif_a[:, lyr] = self.smt / self.swt

        # homogeneous layer (all layers are except the fresnel layer, so the
        # combination of layers including a fresnel layer becomes unhomogeneous, hence
        # why we need to compute rdif/tdif above and below for all layers)
        self.rdif_b[:, lyr] = self.rdif_a[:, lyr]
        self.tdif_b[:, lyr] = self.tdif_a[:, lyr]

        return None

    def calculate_correction_fresnel_layer(self, lyr):
        """Update diffuse and direct reflectivity and transmittivity of current
            layer by integrating effect of Fresnel boundary above, i.e. merging
            the reflectivity & transmittivity of current layer + fresnel layer.

        Corrects fluxes for Fresnel reflection in cases where total
            internal reflection does and does not occur (angle > critical_angle).
            In the diffuse radiation, coefficients are precalculated because
            ~256 gaussian points required for convergence.
        """

        ref_indx = self.column.ref_idx_re + 1j * self.column.ref_idx_im
        critical_angle = np.arcsin(ref_indx)

        for wl in np.arange(0, self.column.nbr_wvl, 1):
            if np.arccos(self.irradiance.mu_not) < critical_angle[wl]:
                # in this case, no total internal reflection

                # compute fresnel reflection and transmission amplitudes
                # for two polarizations: 1=perpendicular and 2=parallel to
                # the plane containing incident, reflected and refracted rays.

                # Eq. 22  Briegleb & Light 2007
                # Inputs to equation 21 (i.e. Fresnel formulae for R and T)

                # reflection amplitude factor for perpendicular polarization
                r1 = (self.mu0[wl] - self.nr[wl] * self.mu0n[wl]) / (
                    self.mu0[wl] + self.nr[wl] * self.mu0n[wl]
                )

                # reflection amplitude factor for parallel polarization
                r2 = (self.nr[wl] * self.mu0[wl] - self.mu0n[wl]) / (
                    self.nr[wl] * self.mu0[wl] + self.mu0n[wl]
                )

                # transmission amplitude factor for perpendicular polarization
                t1 = 2 * self.mu0[wl] / (self.mu0[wl] + self.nr[wl] * self.mu0n[wl])
                # transmission amplitude factor for parallel polarization
                t2 = 2 * self.mu0[wl] / (self.nr[wl] * self.mu0[wl] + self.mu0n[wl])

                # unpolarized light for direct beam
                # Eq. 21  Brigleb and light 2007
                rf_dif_a = 0.5 * (r1**2 + r2**2)
                tf_dir_a = (
                    0.5 * (t1**2 + t2**2) * self.nr[wl] * self.mu0n[wl] / self.mu0[wl]
                )

            # in this case, total internal reflection occurs
            else:
                tf_dir_a = 0
                rf_dif_a = 1

            # precalculated diffuse reflectivities and transmissivities
            # for incident radiation above and below fresnel layer, using
            # the direct albedos and accounting for complete internal
            # reflection from below. Precalculated because high order
            # number of gaussian points (~256) is required for convergence:

            # Eq. 25  Briegleb and light 2007
            # diffuse reflection of flux arriving from above

            # reflection from diffuse unpolarized radiation
            rf_dif_a = self.column.fl_r_dif_a[wl]
            tf_dif_a = 1 - rf_dif_a  # transmission from diffuse unpolarized radiation

            # diffuse reflection of flux arriving from below
            rf_dif_b = self.column.fl_r_dif_b[wl]
            tif_dif_b = 1 - rf_dif_b

            # the lyr = lyrfrsnl layer properties are updated to combine
            # the fresnel (refractive) layer, always taken to be above
            # the present layer lyr (i.e. be the top interface):

            # save fluxes of lyr before merging with frsnl layer
            rdif_a_0 = self.rdif_a[wl, lyr].copy()
            rdif_b_0 = self.rdif_b[wl, lyr].copy()
            tdif_a_0 = self.tdif_a[wl, lyr].copy()
            tdif_b_0 = self.tdif_b[wl, lyr].copy()
            tdir_0 = self.tdir[wl, lyr].copy()
            rdir_0 = self.rdir[wl, lyr].copy()

            # combined layer transmissivity to DIRECT radiation
            # Eq. B7  Briegleb & Light 2007
            self.tdir[wl, lyr] = tf_dir_a * tdir_0 + tf_dir_a * self.rdir[
                wl, lyr
            ] * rf_dif_b * self.tdif_a[wl, lyr] * 1 / (1 - rf_dif_b * rdif_a_0)

            # combined layer reflectivity to DIRECT radiation
            # Eq. B7  Briegleb & Light 2007
            self.rdir[wl, lyr] = rf_dif_a + tf_dir_a * rdir_0 * tif_dif_b * 1 / (
                1 - rf_dif_b * rdif_a_0
            )

            # combined layer reflectivity to DIFFUSE radiation (above)
            # Eq. B9  Briegleb & Light 2007
            self.rdif_a[wl, lyr] = rf_dif_a + tf_dif_a * rdif_a_0 * tif_dif_b * 1 / (
                1 - rf_dif_b * rdif_a_0
            )

            # combined layer reflectivity to DIFFUSE radiation (below)
            # Eq. B10  Briegleb & Light 2007
            self.rdif_b[wl, lyr] = rdif_b_0 + tdif_b_0 * rf_dif_b * tdif_a_0 * 1 / (
                1 - rf_dif_b * rdif_b_0
            )

            # combined layer transmissivity to DIFFUSE radiation (above)
            # Eq. B9  Briegleb & Light 2007
            self.tdif_a[wl, lyr] = tdif_a_0 * tf_dif_a * 1 / (1 - rf_dif_b * rdif_a_0)

            # Eq. B10  Briegleb & Light 2007
            self.tdif_b[wl, lyr] = tdif_b_0 * tif_dif_b * 1 / (1 - rf_dif_b * rdif_b_0)

            # update trnlay to include fresnel transmission (Eq. B8)
            self.trnlay[wl, lyr] = tf_dir_a * self.trnlay[wl, lyr]

        return None

    def combine_layers_downward(self, lyr):
        """Calculate energy going downward in the ice column:
        solar beam transmission, total transmission, diffuse transmission,
        and reflectivity to diffuse radiation arriving from below.
        The loop starts at the upper layer, working downwards.
        Equations are B2 & B5 from Briegleb & Light 2007.
        """

        # term below represents 1 / multiple scattering between layers
        # rdndif is the combined reflectivity from all layers
        # above current layer to diffuse radiation coming from above
        # (1 - RBAr1 * RBAr2) in Eq. B2 from B&L 2007
        refkm1 = 1 / (1 - self.rdndif[:, lyr] * self.rdif_a[:, lyr])

        # transmission of solar beam (direct)
        # trnlay = exp(-ts/mu_not), with ts changing every layer,
        # mu0 is mu0n under fresnel lr
        self.trndir[:, lyr + 1] = self.trndir[:, lyr] * self.trnlay[:, lyr]

        # total down diffuse = total transmission - direct transmission
        self.tdndif = self.trntdr[:, lyr] - self.trndir[:, lyr]

        # Eq. B2  Briegleb and Light 2007
        # total transmission for layers above
        self.trntdr[:, lyr + 1] = (
            self.trndir[:, lyr] * self.tdir[:, lyr]
            + (
                self.tdndif
                + self.trndir[:, lyr] * self.rdir[:, lyr] * self.rdndif[:, lyr]
            )
            * refkm1
            * self.tdif_a[:, lyr]
        )

        # reflectivity to diffuse radiation from below for layers above
        self.rdndif[:, lyr + 1] = self.rdif_b[:, lyr] + (
            self.tdif_b[:, lyr] * self.rdndif[:, lyr] * refkm1 * self.tdif_a[:, lyr]
        )

        # Eq. B5 (!! layers are not homogeneous so a =! b)
        self.trndif[:, lyr + 1] = self.trndif[:, lyr] * self.tdif_a[:, lyr] * refkm1

        return None

    def combine_layers_upward(self, lyr):
        """Combine energy going upward in the ice column:
        Compute reflectivity to direct (rupdir) and diffuse (rupdif) radiation
        arriving from above, for layers below current layer.
        The loop starts from the second to last interface, working upwards.
        Equations are B2-B4 from Briegleb & Light 2007.
        """

        # starts at the bottom and works its way up to the top layer
        # interface scattering
        refkp1 = 1 / (1 - self.rdif_b[:, lyr] * self.rupdif[:, lyr + 1])

        # Eq. B2
        self.rupdir[:, lyr] = (
            self.rdir[:, lyr]
            + (
                self.trnlay[:, lyr] * self.rupdir[:, lyr + 1]
                + (self.tdir[:, lyr] - self.trnlay[:, lyr]) * self.rupdif[:, lyr + 1]
            )
            * refkp1
            * self.tdif_b[:, lyr]
        )

        # Eq. B4 (!! layers are not homogeneous so a =! b)
        self.rupdif[:, lyr] = (
            self.rdif_a[:, lyr]
            + self.tdif_a[:, lyr]
            * self.rupdif[:, lyr + 1]
            * refkp1
            * self.tdif_b[:, lyr]
        )
        return None

    def calculate_fluxes_at_interfaces(self, lyr):
        """
        Calculates up and down fluxes at layer interfaces.
        Equation B6 from Briegleb & Light 2007.
        """

        puny = 1e-10  # not sure how should we define this

        # Eq. 52  Briegleb and Light 2007
        # interface scattering
        refk = 1 / (1 - self.rdndif[:, lyr] * self.rupdif[:, lyr])

        # Eq B6
        # dir tran ref from below times interface scattering, plus diff
        # tran and ref from below times interface scattering
        self.fdirup[:, lyr] = (
            self.trndir[:, lyr] * self.rupdir[:, lyr]
            + (self.trntdr[:, lyr] - self.trndir[:, lyr]) * self.rupdif[:, lyr]
        ) * refk

        # dir tran plus total diff trans times interface scattering plus
        # dir tran with up dir ref and down dif ref times interface scattering
        self.fdirdn[:, lyr] = (
            self.trndir[:, lyr]
            + (
                self.trntdr[:, lyr]
                - self.trndir[:, lyr]
                + self.trndir[:, lyr] * self.rupdir[:, lyr] * self.rdndif[:, lyr]
            )
            * refk
        )

        # diffuse tran ref from below times interface scattering
        self.fdifup[:, lyr] = self.trndif[:, lyr] * self.rupdif[:, lyr] * refk

        # diffuse tran times interface scattering
        self.fdifdn[:, lyr] = self.trndif[:, lyr] * refk

        # dfdir = fdirdn - fdirup
        self.dfdir[:, lyr] = (
            self.trndir[:, lyr]
            + (self.trntdr[:, lyr] - self.trndir[:, lyr])
            * (1 - self.rupdif[:, lyr])
            * refk
            - self.trndir[:, lyr]
            * self.rupdir[:, lyr]
            * (1 - self.rdndif[:, lyr])
            * refk
        )

        if np.max(self.dfdir[:, lyr]) < puny:
            # dfdif = fdifdn - fdifup
            self.dfdir[:, lyr] = np.zeros((self.column.nbr_wvl,), dtype=int)

        self.dfdif[:, lyr] = self.trndif[:, lyr] * (1 - self.rupdif[:, lyr]) * refk

        if np.max(self.dfdif[:, lyr]) < puny:
            self.dfdif[:, lyr] = np.zeros((self.column.nbr_wvl,), dtype=int)

        return None

    def calculate_bulk_fluxes(self):
        """
        Calculates total fluxes in each layer and for entire column.
        """

        for n in np.arange(0, self.column.nbr_lyr + 1, 1):
            self.F_up[:, n] = (
                self.fdirup[:, n]
                * (self.irradiance.Fs * self.irradiance.mu_not * np.pi)
                + self.fdifup[:, n] * self.irradiance.Fd
            )
            self.F_dwn[:, n] = (
                self.fdirdn[:, n]
                * (self.irradiance.Fs * self.irradiance.mu_not * np.pi)
                + self.fdifdn[:, n] * self.irradiance.Fd
            )

        self.F_net = self.F_up - self.F_dwn

        # Absorbed flux in each layer
        self.F_abs[:, :] = self.F_net[:, 1:] - self.F_net[:, :-1]

        # Upward flux at upper model boundary
        self.F_top_pls = self.F_up[:, 0]

        # Net flux at lower model boundary = bulk transmission through entire
        # media = absorbed radiation by underlying surface:
        self.F_btm_net = -self.F_net[:, self.column.nbr_lyr]

        self.albedo = self.F_up[:, 0] / self.F_dwn[:, 0]

        return None

    def conservation_of_energy_check(self):
        """
        Checks there is no conservation of energy violation.
        Raises:
            ValueError is conservation of energy error is detected

        """
        # Incident direct+diffuse radiation equals (absorbed+transmitted+bulk_reflected)
        energy_sum = (
            (self.irradiance.mu_not * np.pi * self.irradiance.Fs)
            + self.irradiance.Fd
            - (np.sum(self.F_abs, axis=1) + self.F_btm_net + self.F_top_pls)
        )

        energy_conservation_error = sum(abs(energy_sum))

        if energy_conservation_error > 1e-10:
            raise ValueError(f"energy conservation error: {energy_conservation_error}")
        else:
            pass

        return None

    def get_outputs(self):
        """
        Assimilates useful data into instance of Outputs class.

        Returns:
            outputs: instance of Outputs class

        """

        outputs = Outputs()

        # Radiative heating rate:
        f_abs_slr = np.sum(self.F_abs, axis=0)
        # [K/s] 2117 = specific heat column (J kg-1 K-1)
        heat_rt = f_abs_slr / (np.array(self.column.layer_mass) * 2117)
        outputs.heat_rt = heat_rt * 3600  # [K/hr]

        # Spectral albedo
        outputs.albedo = self.albedo

        # Spectrally-integrated solar, visible, and NIR albedos:
        outputs.BBA = np.sum(self.irradiance.flx_slr * self.albedo) / np.sum(
            self.irradiance.flx_slr
        )

        # Total incident insolation( Wm - 2)
        outputs.total_insolation = np.sum(
            (self.irradiance.mu_not * np.pi * self.irradiance.Fs) + self.irradiance.Fd
        )

        # Spectrally-integrated absorption by underlying surface:
        outputs.abs_slr_btm = np.sum(self.F_btm_net, axis=0)

        # Spectrally-integrated absorption by entire snow/column column
        outputs.abs_slr_tot = np.sum(f_abs_slr)

        # Spectrally-integrated absorption by each layer
        outputs.absorbed_flux_per_layer = f_abs_slr

        return outputs


def solve_adding_doubling(column, irradiance):
    """control function for the adding-doubling solver.

    Makes function calls in sequence to generate, then return, an instance of
    Outputs class.

    Args:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    Returns:
        outputs: Instance of Outputs class

    Raises:
        ValueError if violation of conservation of energy detected

    """

    ads = _AddingDoublingSolver(column, irradiance)

    # initialize reflection and transmission at top interface
    ads.trntdr[:, 0] = 1
    ads.trndif[:, 0] = 1
    ads.rdndif[:, 0] = 0
    ads.trndir[:, 0] = 1

    # initialize reflection to direct & diffuse radiation from lowest interface
    ads.rupdif[:, column.nbr_lyr] = column.sfc
    ads.rupdir[:, column.nbr_lyr] = column.sfc

    # loop through layers
    for lyr in np.arange(0, column.nbr_lyr, 1):

        # condition: if current layer is above fresnel layer or the
        # top layer is a Fresnel layer
        # else: within or below fl
        ads.mu0n = ads.mu0 if lyr < ads.lyrfrsnl else ads.mu0n

        # 1 - calculate reflectivity & transmittivity of the layer to
        # direct & diffuse radiation with dEdd method
        ads.calculate_reflectivity_transmittivity_delta_eddington(lyr)

        # 2 - re calculate reflectivity & transmittivity to diffuse radiation
        # using direct angular integration over rdir and tdir,
        # since Delta-Eddington diffuse formula is not well-behaved
        # (it is usually biased low and can even be negative)
        ads.apply_gaussian_integral(lyr)

        # so far the layer is homogeneous, i.e. the transmittivity and
        # reflectivity to radiation from above (_a) & below (_b) are the same
        ads.calculate_diff_transmittivity_reflectivity(lyr)

        # 3 - if fresnel boundary, a pseudo non-absorbing layer is added and
        # merged to the current layer, so reflectivity & transmittivity are
        # recalculated.
        # (!!) the layer becomes inhomogeneous,  so the reflectivity and
        # transmittivity to radiation from above (_a) are different from the
        # transmittivity to radiation radiation from below (_b)
        if lyr == ads.lyrfrsnl:
            ads.calc_correction_fresnel_layer(lyr)

        # combine layers from up down: calculate total & direct
        # transmission as well as reflection/transmission of diffuse radiation
        # coming from below
        ads.combine_layers_downward(lyr)

    # combine layers from down up: calculate reflectivity to diffuse & direct
    # radiation coming from above
    for lyr in np.arange(column.nbr_lyr - 1, -1, -1):
        ads.combine_layers_upward(lyr)

    # calculate fluxes at interfaces, from up down
    for lyr in np.arange(0, column.nbr_lyr + 1, 1):
        ads.calculate_fluxes_at_interfaces(lyr)

    ads.calculate_bulk_fluxes()

    ads.conservation_of_energy_check()

    outputs = ads.get_outputs()

    return outputs
