"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np


class _TwoStreamSolverAD:
    """
    This class loads and initialize the variables necessary to solve the radiative
    transfer equation using the Delta-Eddington Adding-Doubling two-stream solver 
    from Briegleb and Light 2007, later modified by Whicker et al. 2022. 
    The solver is identical to that of SNICAR-ADv4 
    (https://github.com/chloewhicker/SNICAR-ADv4).

    References:
    Briegleb and Light 2007: https://doi.org/10.5065/D6B27S71
    Whicker et al. 2022: https://doi.org/10.5194/tc-16-1197-2022

    Attributes
    ----------
    column : LandColumn
        An instance of the LandColumn class.
    irradiance : SolarIrradiance
        An instance of the SolarIrradiance class.
    cos_sza : float
        Cosine of solar zenith angle.
    mu0 : array
        Cosine of solar zenith angle repeated for every wavelength.
    epsilon : float
        Small number to prevent numerical singularities.
    exp_min : float
        Minimum exponent value to prevent underflow.
    nbr_wvl : int
        Number of wavelengths.
    nr : array
        Modified refractive index adjusted for the imaginary component.
    mu0n : ndarray
        Cosine of the refraction angle after applying Snell's law.
    trnlay: array
        Transmission of direct solar beam (exponential term).
    rdif_a, rdif_b, tdif_a, tdif_b : array
        Layer reflectivities and transmittivities to diffuse radiation.
    rdir, tdir : ndarray
        Layer reflectivity and transmittivity to direct radiation.
    rupdif, rupdir : array
        Upward reflection of diffuse and direct radiation.
    rdndif : array
        Combined reflectivity from all layers above current layer to
        diffuse radiation coming from above.
    trntdr, trndif, trndir : array
        Spectral transmission (total, diffuse, direct).
    fdirup, fdirdn : ndarray
        Upward/downward direct solar fluxes.
    fdifup, fdifdn : ndarray
        Upward/downward diffuse fluxes.
    dfdir, dfdif : ndarray
        Differences between up/down for direct and diffuse fluxes.
    F_up, F_dwn, F_abs : ndarray
        Total upward, downward, and absorbed fluxes.
    GAUSPT : list of float
        Gaussian quadrature points (cosine of angle).
    GAUSWT : list of float
        Corresponding Gaussian quadrature weights.
    lyrfrsnl : int
        Index of first layer with refractive boundary (layer_type == 1).
    ts : array
        Spectral optical thickness of each layer (Delta-scaled).
    ws : array
        Spectral single scattering albedo of each layer (Delta-scaled).
    gs : array
        Spectral asymmetry parameter of each layer (Delta-scaled).
    albedo : array
        Hemispherical spectral albedo.

    """

    def __init__(self, column, irradiance):
        """
        Initialize the radiative transfer solver.

        Sets up the internal state and pre-allocates arrays based on the
        provided snow/ice column and incoming solar irradiance parameters.

        Parameters
        ----------
        column : LandColumn
            An instance of the LandColumn class containing the
            properties of the snow/ice column.
        irradiance : SolarIrradiance
            An instance of the SolarIrradiance class providing the incoming
            solar flux and the cosine of the solar zenith angle.
        """

        self.column = column

        self.irradiance = irradiance

        self.cos_sza = np.cos(np.deg2rad(np.rint(irradiance.sza)))
        
        self.wavelengths = self.column._wavelengths

        self.nbr_wvl = column.nbr_wvl

        # cos beam angle = incident beam
        self.mu0 = self.cos_sza * np.ones(self.nbr_wvl)

        # to deal with singularity
        self.epsilon = 1e-5

        # exp(-500)  # min value > 0 to avoid error
        self.exp_min = 1e-5

        self.nr = np.zeros(shape=self.nbr_wvl)

        # ice-adjusted real refractive index
        temp1 = (
            column.ref_idx_re**2
            - column.ref_idx_im**2
            + np.sin(np.arccos(self.cos_sza)) ** 2
        )
        temp2 = (
            column.ref_idx_re**2
            - column.ref_idx_im**2
            - np.sin(np.arccos(self.cos_sza)) ** 2
        )
        self.nr = (np.sqrt(2) / 2) * (
            temp1 + (temp2**2 + 4 * column.ref_idx_re**2 * column.ref_idx_im**2) ** 0.5
        ) ** 0.5

        # refraction angle (Snell's law)
        self.mu0n = np.cos(np.arcsin(np.sin(np.arccos(self.mu0)) / self.nr))

        # solar beam transmission for layer (direct beam only)
        self.trnlay = np.zeros(shape=[self.nbr_wvl, column.nbr_lyr + 1])

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

        # reflectivity to diffuse radiation
        self.rupdif = np.zeros_like(self.trnlay)

        # reflectivity to direct radiation
        self.rupdir = np.zeros_like(self.trnlay)

        # reflection of diffuse radiation for layers above
        self.rdndif = np.zeros_like(self.trnlay)

        # total transmission from layers above
        self.trntdr = np.zeros_like(self.trnlay)

        # diffuse transmission for layers above
        self.trndif = np.zeros_like(self.trnlay)

        # solar beam down transmission from top
        self.trndir = np.zeros_like(self.trnlay)

        # direct flux up
        self.fdirup = np.zeros_like(self.trnlay)

        # diffuse flux up
        self.fdifup = np.zeros_like(self.trnlay)

        # direct flux down
        self.fdirdn = np.zeros_like(self.trnlay)

        # diffuse flux up
        self.fdifdn = np.zeros_like(self.trnlay)

        # difference between up and down (direct)
        self.dfdir = np.zeros_like(self.trnlay)

        # difference between up and down (diffuse)
        self.dfdif = np.zeros_like(self.trnlay)

        # total flux up
        self.F_up = np.zeros_like(self.trnlay)

        # total flux down
        self.F_dwn = np.zeros_like(self.trnlay)

        # absorbed flux
        self.F_abs = np.zeros(shape=[self.nbr_wvl, column.nbr_lyr])

        # find interface with Fresnel boundary (i.e. layer type > 0)
        if np.sum(np.array(column.layer_type) == 1) > 0:
            self.lyrfrsnl = column.layer_type.index(1)

        else:
            self.lyrfrsnl = 9999999

        # gaussian angles (in cos(theta)) to integrate for diffuse fluxes
        # cf. Table A. p. 66 Briegleb and Light 2007
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
        # gaussian weights to integrate for diffuse fluxes
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
        """
        Compute multiple scattering in each layer via the Delta Eddington
        approximation to yield reflectivity and transmissivity of the layer to
        direct and diffuse radiation. Use equations A24, A26, A30, A31 from
        Briegleb and Light 2007.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the optical properties are calculated.

        """

        tautot = self.column.tau.T[:, lyr]
        wtot = self.column.ss_alb.T[:, lyr]
        gtot = self.column.asm_prm.T[:, lyr]
        ftot = self.column.asm_prm.T[:, lyr] * self.column.asm_prm.T[:, lyr]

        # layer delta-scaled extinction optical depth
        self.ts = (1 - (wtot * ftot)) * tautot

        # layer delta-scaled single scattering albedo
        self.ws = ((1 - ftot) * wtot) / (1 - (wtot * ftot))

        # layer delta-scaled asymmetry parameter
        self.gs = gtot / (1 + gtot)

        # lambda
        lm = np.sqrt(3 * (1 - self.ws) * (1 - self.ws * self.gs))

        # u, term in diffuse reflectivity and transmissivity
        ue = 1.5 * (1 - self.ws * self.gs) / lm

        # extinction, MAX function lyr keeps from getting an error
        # if the exp(-lm*ts) is < 1e-5
        extins = np.maximum(
            np.full((self.nbr_wvl,), self.exp_min), np.exp(-lm * self.ts)
        )

        # N, term in diffuse reflectivity and transmissivity
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
            np.full((self.nbr_wvl,), self.exp_min), np.exp(-self.ts / self.mu0n)
        )

        #  Eq. 50: Briegleb and Light 2007  alpha and gamma for direct radiation
        # alp = alpha(ws,mu0n,gs,lm)
        alp = (
            (0.75 * self.ws * self.mu0n)
            * (1 + self.gs * (1 - self.ws))
            / (1 - lm**2 * self.mu0n**2 + self.epsilon)
        )

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
        """
        Integrate reflectivity and transmissivity to direct radiation using
        Gaussian quadrature to get diffuse reflectivity and transmittivity.

        This method performs angular integration with a fixed number
        of discrete zenith angles and corresponding weights (default N=8).

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
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
                np.full((self.nbr_wvl,), self.exp_min), np.exp(-self.ts / mu)
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
        Calculate transmissivity and reflectivity to diffuse radiation
        after gaussian integration, eq. A33 Briegleb and Light 2007.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
        """
        self.rdif_a[:, lyr] = self.smr / self.swt
        self.tdif_a[:, lyr] = self.smt / self.swt

        # homogeneous layer (all layers are except the fresnel layer, so the
        # combination of layers including a fresnel layer becomes unhomogeneous,
        # hence why we need to compute rdif/tdif above and below for all layers)
        self.rdif_b[:, lyr] = self.rdif_a[:, lyr]
        self.tdif_b[:, lyr] = self.tdif_a[:, lyr]

        return None

    def calculate_correction_fresnel_layer(self, lyr):
        """
        Update diffuse + direct reflectivity and transmittivity of current
            layer by integrating effect of Fresnel boundary, i.e. merging
            the reflectivity & transmittivity of current layer + fresnel layer.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
        """

        # Eq. 22  Briegleb & Light 2007
        # reflection amplitude factor for perpendicular polarization
        r1 = (self.mu0 - self.nr * self.mu0n) / (self.mu0 + self.nr * self.mu0n)
        # reflection amplitude factor for parallel polarization
        r2 = (self.nr * self.mu0 - self.mu0n) / (self.nr * self.mu0 + self.mu0n)

        # transmission amplitude factor for perpendicular polarization
        t1 = 2 * self.mu0 / (self.mu0 + self.nr * self.mu0n)

        # transmission amplitude factor for parallel polarization
        t2 = 2 * self.mu0 / (self.nr * self.mu0 + self.mu0n)

        # Eq. 21  Brigleb and light 2007
        rf_dir_a = 0.5 * (r1**2 + r2**2)
        tf_dir_a = 0.5 * (t1**2 + t2**2) * self.nr * self.mu0n / self.mu0

        # mask where total internal reflection occurs
        ref_indx = self.column.ref_idx_re + 1j * self.column.ref_idx_im
        critical_angle = np.arcsin(ref_indx)
        mask = np.arccos(self.cos_sza) >= critical_angle
        rf_dir_a[mask] = 1
        tf_dir_a[mask] = 0

        # Eq. 25  Briegleb and light 2007
        # diffuse reflection of flux arriving from above
        rf_dif_a = self.column.fl_r_dif_a
        tf_dif_a = 1 - rf_dif_a
        # diffuse reflection of flux arriving from below
        rf_dif_b = self.column.fl_r_dif_b
        tif_dif_b = 1 - rf_dif_b

        # save fluxes of lyr before merging with frsnl layer
        rdif_a_0 = self.rdif_a[:, lyr].copy()
        rdif_b_0 = self.rdif_b[:, lyr].copy()
        tdif_a_0 = self.tdif_a[:, lyr].copy()
        tdif_b_0 = self.tdif_b[:, lyr].copy()
        tdir_0 = self.tdir[:, lyr].copy()
        rdir_0 = self.rdir[:, lyr].copy()

        # combined layer transmissivity to DIRECT radiation
        # Eq. B7  Briegleb & Light 2007
        self.tdir[:, lyr] = tf_dir_a * tdir_0 + tf_dir_a * self.rdir[
            :, lyr
        ] * rf_dif_b * self.tdif_a[:, lyr] * 1 / (1 - rf_dif_b * rdif_a_0)

        # combined layer reflectivity to DIRECT radiation
        # Eq. B7  Briegleb & Light 2007
        self.rdir[:, lyr] = rf_dir_a + tf_dir_a * rdir_0 * tif_dif_b * 1 / (
            1 - rf_dif_b * rdif_a_0
        )

        # combined layer reflectivity to DIFFUSE radiation (above)
        # Eq. B9  Briegleb & Light 2007
        self.rdif_a[:, lyr] = rf_dif_a + tf_dif_a * rdif_a_0 * tif_dif_b * 1 / (
            1 - rf_dif_b * rdif_a_0
        )

        # combined layer reflectivity to DIFFUSE radiation (below)
        # Eq. B10  Briegleb & Light 2007
        self.rdif_b[:, lyr] = rdif_b_0 + tdif_b_0 * rf_dif_b * tdif_a_0 * 1 / (
            1 - rf_dif_b * rdif_b_0
        )

        # combined layer transmissivity to DIFFUSE radiation (above)
        # Eq. B9  Briegleb & Light 2007
        self.tdif_a[:, lyr] = tdif_a_0 * tf_dif_a * 1 / (1 - rf_dif_b * rdif_a_0)

        # Eq. B10  Briegleb & Light 2007
        self.tdif_b[:, lyr] = tdif_b_0 * tif_dif_b * 1 / (1 - rf_dif_b * rdif_b_0)

        # update trnlay to include fresnel transmission (Eq. B8)
        self.trnlay[:, lyr] = tf_dir_a * self.trnlay[:, lyr]

        return None

    def combine_layers_downward(self, lyr):
        """
        Calculate energy going downward in the ice column:
        solar beam transmission, total transmission, diffuse transmission,
        and reflectivity to diffuse radiation arriving from below.
        The loop starts at the upper layer, working downwards.
        Equations are B2 & B5 from Briegleb & Light 2007.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
        """

        # term below represents 1 / multiple scattering between layers
        # rdndif is the combined reflectivity from all layers
        # above current layer to diffuse radiation coming from above
        # (1 - RBAr1 * RBAr2) in Eq. B2 from B&L 2007
        refkm1 = 1 / (1 - self.rdndif[:, lyr] * self.rdif_a[:, lyr])

        # transmission of solar beam (direct)
        # trnlay = exp(-ts/cos_sza), with ts changing every layer,
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
        """
        Combine energy going upward in the ice column:
        Compute reflectivity to direct (rupdir) and diffuse (rupdif) radiation
        arriving from above, for layers below current layer.
        The loop starts from the second to last interface, working upwards.
        Equations are B2-B4 from Briegleb & Light 2007.

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
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

        Parameters
        ----------
        lyr : int
            Index of the layer for which the integration is applied.
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
            self.dfdir[:, lyr] = np.zeros((self.nbr_wvl,), dtype=int)

        self.dfdif[:, lyr] = self.trndif[:, lyr] * (1 - self.rupdif[:, lyr]) * refk

        if np.max(self.dfdif[:, lyr]) < puny:
            self.dfdif[:, lyr] = np.zeros((self.nbr_wvl,), dtype=int)

        return None

    def calculate_bulk_fluxes(self):
        """Calculate total fluxes in each layer and for entire column."""

        for n in np.arange(0, self.column.nbr_lyr + 1, 1):
            self.F_up[:, n] = (
                self.fdirup[:, n] * (self.irradiance.direct_beam * self.cos_sza)
                + self.fdifup[:, n] * self.irradiance.diffuse * np.pi
            )
            self.F_dwn[:, n] = (
                self.fdirdn[:, n] * (self.irradiance.direct_beam * self.cos_sza)
                + self.fdifdn[:, n] * self.irradiance.diffuse * np.pi
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
        Perform conservation of energy validation.

        This method verifies that the total incident solar energy (direct + diffuse)
        is equal to the sum of absorbed, transmitted, and reflected energy across
        the entire snow/ice column. If an imbalance is detected beyond a small
        tolerance (1e-10), an error is raised.

        Raises
        ------
        ValueError
            If the energy balance check fails (i.e., energy conservation is violated).


        """
        # Incident direct+diffuse radiation equals (absorbed+transmitted+bulk_reflected)
        energy_sum = (
            (self.irradiance.direct_beam * self.cos_sza)
            + self.irradiance.diffuse * np.pi
            - (np.sum(self.F_abs, axis=1) + self.F_btm_net + self.F_top_pls)
        )

        energy_conservation_error = sum(abs(energy_sum))

        if energy_conservation_error > 1e-10:
            raise ValueError(f"energy conservation error: {energy_conservation_error}")

        return None

    def get_outputs(self):
        """
        Compile and return radiative transfer results as dictionary

        Returns
        -------
        results : dictionary
            Two-stream solver results.

        """

        results = {}

        # Radiative heating rate:
        f_abs_slr = np.sum(self.F_abs, axis=0)

        # Spectrally-integrated solar, visible, and NIR albedos:
        # BBA = np.sum(self.irradiance.total_irradiance * self.albedo) / np.sum(
        #     self.irradiance.total_irradiance
        # )
        
        BBA = np.trapezoid(
            self.F_up[:, 0], x=self.wavelengths
        ) / np.trapezoid(self.F_dwn[:, 0], x=self.wavelengths)
        
        results["broadband_albedo_boa"] = BBA
        results["albedo_boa"] = self.albedo

        # Spectrally-integrated absorption by underlying surface:
        abs_slr_btm = np.sum(self.F_btm_net, axis=0)

        results["absorbed_flux_fraction"] = f_abs_slr
        results["absorbed_flux_fraction_bottom"] = f_abs_slr

        return results


def solve_two_stream_rt_ad(column, irradiance):
    """
    Solve radiative transfer through a layered snow/ice column using the
    two-stream Delta-Eddington adding doubling solver from Briegleb and Light
    2007, with updates from Whicker et al. 2022.

    Computes upward and downward fluxes in a layered snow or ice column based
    on the column optical properties and incoming solar irradiance.
    Makes function calls in sequence to generate, then return, an instance of
    Outputs class storing the results generated by the solver.


    Parameters
    ----------
    column : LandColumn
        An instance of the `LandColumn` class containing the optical and
        physical properties of the snow/ice column.

    irradiance : SolarIrradiance
        An instance of the `SolarIrradiance` class providing spectral solar fluxes
        (direct and diffuse) and the cosine of the solar zenith angle.

    Returns
    -------
        outputs: _TwoStreamSolverResults


    Raises
    ------
        ValueError if violation of conservation of energy detected.

    """

    ads = _TwoStreamSolverAD(column, irradiance)

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
            ads.calculate_correction_fresnel_layer(lyr)

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
