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

import numpy as np

from snicarfx.rt_solvers.adding_doubling_solver_oo import Outputs


def adding_doubling_solver(column, irradiance):
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

    # DEFINE CONSTANTS AND ARRAYS
    (
        tau0,
        g0,
        ssa0,
        epsilon,
        exp_min,
        nr,
        mu0,
        mu0n,
        trnlay,
        rdif_a,
        rdif_b,
        tdif_a,
        tdif_b,
        rdir,
        tdir,
        lyrfrsnl,
        trnlay,
        rdif_a,
        rdif_b,
        tdif_a,
        tdif_b,
        rdir,
        tdir,
        rdndif,
        trntdr,
        trndir,
        trndif,
        fdirup,
        fdifup,
        fdirdn,
        fdifdn,
        dfdir,
        dfdif,
        F_up,
        F_dwn,
        F_abs,
        F_abs_vis,
        F_abs_nir,
        rupdif,
        rupdir,
    ) = define_constants_arrays(irradiance, column)
    
    # initialize reflection and transmission at top interface 
    trntdr[:, 0] = 1
    trndif[:, 0] = 1
    rdndif[:, 0] = 0
    trndir[:, 0] = 1
    
    # initialize reflection to direct & diffuse radiation from lowest interface
    rupdif[:, column.nbr_lyr] = column.sfc
    rupdir[:, column.nbr_lyr] = column.sfc

    for lyr in np.arange(0, column.nbr_lyr, 1):  # loop through layers
    
        # condition: if current layer is above fresnel layer or the
        # top layer is a Fresnel layer
        if lyr < lyrfrsnl:
            mu0n = mu0

        else:
            # within or below fl
            mu0n = mu0n
        
        # 1 - calculate reflectivity & transmittivity of the layer to 
        # direct & diffuse radiation with dEdd method 
        rdir, tdir, ts, ws, gs, rdif_a, tdif_a = calc_reflectivity_transmittivity_delta_eddington(
            column,
            tau0,
            ssa0,
            g0,
            lyr,
            exp_min,
            trnlay,
            mu0n,
            epsilon,
            rdir,
            tdir,
            rdif_a, 
            tdif_a
        )
        
        # 2 - re calculate reflectivity & transmittivity to diffuse radiation
        # using direct angular integration over rdir and tdir,
        # since Delta-Eddington diffuse formula is not well-behaved 
        # (it is usually biased low and can even be negative)
        
        smt, smr, swt = apply_gaussian_integral(
            column, exp_min, ts, ws, gs, epsilon, lyr, rdif_a, tdif_a
        )
        
        # so far the layer is homogeneous, i.e. the transmittivity and 
        # reflectivity to radiation from above (_a) & below (_b) are the same 
        rdif_a, tdif_a, rdif_b, tdif_b = calc_diff_transmittivity_reflectivity(
            swt, smr, smt, lyr, rdif_a, tdif_a, rdif_b, tdif_b
        )

        # 3 - if fresnel boundary, a pseudo non-absorbing layer is added and
        # merged to the current layer, so reflectivity & transmittivity are 
        # recalculated. 
        # (!!) the layer becomes inhomogeneous,  so the reflectivity and
        # transmittivity to radiation from above (_a) are different from the 
        # transmittivity to radiation radiation from below (_b)
        if lyr == lyrfrsnl:
            (
                rdif_a,
                rdif_b,
                tdif_a,
                tdif_b,
                trnlay,
                rdir,
                tdir,
            ) = calc_correction_fresnel_layer(
                column,
                irradiance,
                mu0n,
                mu0,
                nr,
                rdif_a,
                rdif_b,
                tdif_a,
                tdif_b,
                trnlay,
                lyr,
                rdir,
                tdir,
            )
                
        
        # combine layers from up down: calculate total & direct
        # transmission as well as reflection/transmission of diffuse radiation
        # coming from below
        trndir, trntdr, rdndif, trndif = combine_layers_downward(
            lyr,
            trnlay,
            rdif_a,
            rdir,
            tdif_a,
            rdif_b,
            tdir,
            tdif_b,
            trndir,
            rdndif,
            trntdr,
            trndif,
        )

    # combine layers from down up: calculate reflectivity to diffuse & direct 
    # radiation coming from above
    for lyr in np.arange(
        column.nbr_lyr - 1, -1, -1
    ):  
        rupdif, rupdir = combine_layers_upward(
            lyr,
            rdif_a,
            rdif_b,
            tdif_a,
            tdif_b,
            trnlay,
            rdir,
            tdir,
            rupdif,
            rupdir
        )
 
        
    # calculate fluxes at interfaces, from up down
    for lyr in np.arange(0, column.nbr_lyr + 1, 1):
        
        fdirup, fdifup, fdirdn, fdifdn = calculate_fluxes_at_interfaces(
            column,
            lyr,
            rupdif,
            rupdir,
            rdndif,
            trndir,
            trndif,
            trntdr,
            fdirup,
            fdirdn,
            fdifup,
            fdifdn,
            dfdir,
            dfdif,
        )

    albedo, F_abs, F_btm_net, F_top_pls = calculate_bulk_fluxes(
        column,
        irradiance,
        fdirup,
        fdifup,
        fdirdn,
        fdifdn,
        F_up,
        F_dwn,
        F_abs,
        F_abs_vis,
        F_abs_nir,
    )

    conservation_of_energy_check(irradiance, F_abs, F_btm_net, F_top_pls)

    outputs = get_outputs(column, irradiance, albedo,  F_abs, F_btm_net)

    return outputs




def define_constants_arrays(irradiance, column):
    """defines and instantiates constants required for a-d calculations.

    Defines and instantiates all variables required for calculating energy fluxes
    using the adding-doubling method.

    Args:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    Returns:
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
        trnlay:
        rdif_a:
        rdif_b:
        tdif_a:
        tdif_b:
        rdir:
        tdir:
        rdndif:
        trntdr:
        trndir:
        trndif:
        fdirup:
        fdifup:
        fdirdn:
        fdifdn:
        dfdir:
        dfdif:
        F_up:
        F_dwn:
        F_abs:
        F_abs_vis:
        F_abs_nir:
        rupdif:
        rupdir:

    """

    tau0 = column.tau.T  # read and transpose tau
    g0 = column.asm_prm.T  # read and transpose g
    ssa0 = column.ss_alb.T  # read and transpose ssa
    epsilon = 1e-5  # to deal with singularity
    exp_min = 1e-5  # exp(-500)  # min value > 0 to avoid error
    mu0 = irradiance.mu_not * np.ones(column.nbr_wvl)  # cos beam angle = incident beam

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
    nr = (np.sqrt(2) / 2) * (
        temp1 + (temp2**2 + 4 * column.ref_idx_re**2 * column.ref_idx_im**2) ** 0.5
    ) ** 0.5

    # . Eq. 20: Briegleb and Light 2007: adjusts beam angle
    # (i.e. this is Snell's Law for refraction at interface between media)
    # mu0n = -1 represents light travelling vertically upwards and mu0n = +1
    # represents light travellign vertically downwards
    # mu0n = np.sqrt(1-((1-mu0**2)/(ref_indx*ref_indx)))  (original,
    # before update for diffuse Fresnel reflection)
    # this version accounts for diffuse fresnel reflection:
    mu0n = np.cos(np.arcsin(np.sin(np.arccos(mu0)) / nr))

    # solar beam transm for layer (direct beam only)
    trnlay = np.zeros(shape=[column.nbr_wvl, column.nbr_lyr + 1])
    # layer reflectivity to diffuse radiation from above
    rdif_a = np.zeros_like(trnlay)
    # layer reflectivity to diffuse radiation from below
    rdif_b = np.zeros_like(trnlay)
    # layer transmittivity to diffuse radiation from above
    tdif_a = np.zeros_like(trnlay)
    # layer transmittivity to diffuse radiation from below
    tdif_b = np.zeros_like(trnlay)
    # layer reflectivity to direct radiation (solar beam + diffuse)
    rdir = np.zeros_like(trnlay)
    # layer transmittivity to direct radiation (solar beam + diffuse)
    tdir = np.zeros_like(trnlay)

    # reflection of diffuse radiation for layers above
    rdndif = np.zeros_like(trnlay)
    # total transmission from layers above
    trntdr = np.zeros_like(trnlay)
    # diffuse transmission for layers above
    trndif = np.zeros_like(trnlay)
    # solar beam down transmission from top
    trndir = np.zeros_like(trnlay)
    # reflectivity to diffuse radiation
    rupdif = np.zeros_like(trnlay)
    # reflectivity to direct radiation
    rupdir = np.zeros_like(trnlay)

    fdirup = np.zeros_like(trnlay)
    fdifup = np.zeros_like(trnlay)
    fdirdn = np.zeros_like(trnlay)
    fdifdn = np.zeros_like(trnlay)
    dfdir = np.zeros_like(trnlay)
    dfdif = np.zeros_like(trnlay)
    F_up = np.zeros_like(trnlay)
    F_dwn = np.zeros_like(trnlay)
    
    
    F_abs = np.zeros(shape=[column.nbr_wvl, column.nbr_lyr])
    F_abs_vis = np.zeros(shape=[column.nbr_lyr])
    F_abs_nir = np.zeros(shape=[column.nbr_lyr])



    # if there are non zeros in layer type, grab the index of the
    # first fresnel layer and load in the precalculated diffuse fresnel
    # reflection
    # (precalculated as large no. of gaussian points required for convergence)
    if np.sum(np.array(column.layer_type) == 1) > 0:
        lyrfrsnl = column.layer_type.index(1)

    else:
        lyrfrsnl = 9999999

    return (
        tau0,
        g0,
        ssa0,
        epsilon,
        exp_min,
        nr,
        mu0,
        mu0n,
        trnlay,
        rdif_a,
        rdif_b,
        tdif_a,
        tdif_b,
        rdir,
        tdir,
        lyrfrsnl,
        trnlay,
        rdif_a,
        rdif_b,
        tdif_a,
        tdif_b,
        rdir,
        tdir,
        rdndif,
        trntdr,
        trndir,
        trndif,
        fdirup,
        fdifup,
        fdirdn,
        fdifdn,
        dfdir,
        dfdif,
        F_up,
        F_dwn,
        F_abs,
        F_abs_vis,
        F_abs_nir,
        rupdif,
        rupdir,
    )


def calc_reflectivity_transmittivity_delta_eddington(
    column,
    tau0,
    ssa0,
    g0,
    lyr,
    exp_min,
    trnlay,
    mu0n,
    epsilon,
    rdir,
    tdir,
    rdif_a, 
    tdif_a
):
    
    
    
    """Calculates multiple scattering within a given layer to yield
    reflectivity and transmissivity of the layer to
    direct and diffuse radiation, using the Delta-Eddington solution.
    Eq. A24, A26, A30, A31 Briegleb and Light 2007

    Sets up new variables, applies delta transformation and makes
    initial calculations of direct reflectivity and transmissivity in each
    layer.

    Args:
        tau0: initial optical thickness
        ssa0: initial single scattering albedo
        g0: initial asymmetry parameter
        lyr: index of current layer
        exp_min: small number to avoid /zero error
        trnlay: transmission through layer
        mu0n: incident beam angle adjusted for refraction
        epsilon: small number to avoid singularity
        rdir: reflectivity to direct beam
        tdir: transmissivity to direct beam

    Returns:
        rdir: layer reflectivity to direct beam
        tdir: layer transmissivity to direct beam
        ts: layer delta-scaled extinction optical depth
        ws: layer delta-scaled single scattering albedo
        gs:layer delta-scaled asymmetry parameter
        rdif_a: layer reflectivity to diffuse radiation from above
        tdif_a: layer transmittivity to diffuse radiation from below

    """
    # calculation over layers with penetrating radiation
    # includes optical thickness, single scattering albedo,
    # asymmetry parameter and total flux
    tautot = tau0[:, lyr]
    wtot = ssa0[:, lyr]
    gtot = g0[:, lyr]
    ftot = g0[:, lyr] * g0[:, lyr]

    # coefficient for delta eddington solution for all layers
    # Eq. 50: Briegleb and Light 2007
    # layer delta-scaled extinction optical depth
    ts = (1 - (wtot * ftot)) * tautot
    ws = ((1 - ftot) * wtot) / (
        1 - (wtot * ftot)
    )  # layer delta-scaled single scattering albedo
    gs = gtot / (1 + gtot)  # layer delta-scaled asymmetry parameter
    lm = np.sqrt(3 * (1 - ws) * (1 - ws * gs))  # lambda
    ue = (
        1.5 * (1 - ws * gs) / lm
    )  # u equation, term in diffuse reflectivity and transmissivity
    extins = np.maximum(
        np.full((column.nbr_wvl,), exp_min), np.exp(-lm * ts)
    )  # extinction, MAX function lyr keeps from getting an error
    # if the exp(-lm*ts) is < 1e-5
    ne = (ue + 1) ** 2 / extins - (
        ue - 1
    ) ** 2 * extins  # N equation, term in diffuse reflectivity and transmissivity

    # calculation of rdif, tdif using Delta-Eddington formulas
    # Eq. A30 and A31 Briegleb and Light 2007
    # note that rdif and tdif are used for rdir, tdir calculations here 
    # but are recalculated with gaussian integration later

    rdif_a[:, lyr] = (
        (ue**2 - 1) * (1 / extins - extins) / ne
    )
    tdif_a[:, lyr] = 4 * ue / ne

    # evaluate rdir, tdir for direct beam
    trnlay[:, lyr] = np.maximum(
        np.full((column.nbr_wvl,), exp_min), np.exp(-ts / mu0n)
    )  # transmission from TOA to interface

    #  Eq. 50: Briegleb and Light 2007  alpha and gamma for direct radiation
    alp = (
        (0.75 * ws * mu0n) * (1 + gs * (1 - ws)) / (1 - lm**2 * mu0n**2 + epsilon)
    )  # alp = alpha(ws,mu0n,gs,lm)
    gam = (0.5 * ws) * (
        (1 + 3 * gs * mu0n**2 * (1 - ws)) / (1 - lm**2 * mu0n**2 + epsilon)
    )  # gam = gamma(ws,mu0n,gs,lm)

    # apg = alpha plus gamma
    # amg = alpha minus gamma
    apg = alp + gam
    amg = alp - gam

    rdir[:, lyr] = apg * rdif_a[:, lyr] + amg * (
        tdif_a[:, lyr] * trnlay[:, lyr] - 1
    )  # layer reflectivity to DIRECT radiation
    tdir[:, lyr] = (
        apg * tdif_a[:, lyr] + (amg * rdif_a[:, lyr] - apg + 1) * trnlay[:, lyr]
    )  # layer transmissivity to DIRECT radiation

    return rdir, tdir, ts, ws, gs, rdif_a, tdif_a


def apply_gaussian_integral(
    column, exp_min, ts, ws, gs, epsilon, lyr, rdif_a, tdif_a
):
    """Applies gaussian integral to integrate over angles.

    Uses gaussien integration to integrate fluxes hemispherically from
    N of reference angles where N = len(gauspt) (default is 8).

    Args:
        exp_min: small number for avoiding div/0 error
        ts: delta-scaled extinction optical depth for lyr
        ws: delta-scaled single scattering albedo for lyr
        gs: delta-scaled asymmetry parameter for lyr
        epsilon: small number to avoid singularity
        lyr: integer representing index of current layer (0==top)
        rdif_a: layer reflectivity to diffuse radiation from above
        tdif_a: layer transmittivity to diffuse radiation from below

    Returns:
        smt: accumulator for tdif gaussian integration
        smr: accumulator for rdif gaussian integration
        swt: sum of gaussian weights

    """
    # gaussian angles (radians)
    gauspt = [
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
    gauswt = [
        0.0271525,
        0.0622535,
        0.0951585,
        0.1246290,
        0.1495960,
        0.1691565,
        0.1826034,
        0.1894506,
    ]
    
    # nodes, weights = np.polynomial.legendre.leggauss(8)
    
    # # change of variable in the interval a=0 to b=1 (equivalent to [-1; 0])
    # a, b = 0, 1
    # gauspt = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    # gauswt = (b - a) / 2 * weights
    
    swt = 0
    smr = 0
    smt = 0
    
    R1 = rdif_a[:,lyr].copy() # use R1 as temporary
    T1 = tdif_a[:,lyr].copy() # use T1 as temporary

    for ng in np.arange(0, len(gauspt), 1):
        mu = gauspt[ng]  # solar zenith angles
        gwt = gauswt[ng]  # gaussian weight
        swt = swt + mu * gwt  # sum of weights
        trn = np.maximum(
            np.full((column.nbr_wvl,), exp_min), np.exp(-ts / mu)
        )  # transmission
        lm = np.sqrt(3 * (1 - ws) * (1 - ws * gs))
        alp = (
            (0.75 * ws * mu) * (1 + gs * (1 - ws)) / (1 - lm**2 * mu**2 + epsilon)
        )  # alp = alpha(ws,mu0n,gs,lm)
        gam = (
            (0.5 * ws)
            * (1 + 3 * gs * mu**2 * (1 - ws))
            / (1 - lm**2 * mu**2 + epsilon)
        )  # gam = gamma(ws,mu0n,gs,lm)

        apg = alp + gam
        amg = alp - gam
        rdr = apg * R1 + amg * T1 * trn - amg
        tdr = apg * T1 + amg * R1 * trn - apg * trn + trn
        smr = smr + mu * rdr * gwt  # accumulator for rdif gaussian integration
        smt = smt + mu * tdr * gwt  # accumulator for tdif gaussian integration

    return smt, smr, swt


def calc_diff_transmittivity_reflectivity(
    swt, smr, smt, lyr, rdif_a, tdif_a, rdif_b, tdif_b
):
    """ calculate transmissivity and reflectivity to DIFFUSE radiation
    after gaussian integration, eq. A33 Briegleb and Light 2007.

    Args:
        swt: sum of gaussian weights (for integrating over angle)
        smr: accumulator for rdif gaussian integration
        smt: accumulator for tdif gaussian integration
        lyr: integer representign index of current layer (0 == top)
        rdif_a: layer reflectivity to diffuse radiation from above
        rdif_b: layer reflectivity to diffuse radiation from below
        tdif_a: layer transmittivity to diffuse radiation from above
        tdif_b: layer transmittivity to diffuse radiation from above

    Returns:
        rdif_a: layer reflectivity to diffuse radiation from above
        rdif_b: layer reflectivity to diffuse radiation from below
        tdif_a: layer transmittivity to diffuse radiation from above
        tdif_b: layer transmittivity to diffuse radiation from above
    """
    rdif_a[:, lyr] = smr / swt
    tdif_a[:, lyr] = smt / swt

    # homogeneous layer (all layers are except the fresnel layer, so the 
    # combination of layers including a fresnel layer becomes unhomogeneous, hence 
    # why we need to compute rdif/tdif above and below for all layers)
    rdif_b[:, lyr] = rdif_a[:, lyr]
    tdif_b[:, lyr] = tdif_a[:, lyr]

    return rdif_a, tdif_a, rdif_b, tdif_b


def calc_correction_fresnel_layer(
    column,
    irradiance,
    mu0n,
    mu0,
    nr,
    rdif_a,
    rdif_b,
    tdif_a,
    tdif_b,
    trnlay,
    lyr,
    rdir,
    tdir,
):
    """Update diffuse and direct reflectivity and transmittivity of current
    layer by integrating effect of Fresnel boundary above, i.e. merging
    the reflectivity & transmittivity of current layer + fresnel layer.
    
    Corrects fluxes for Fresnel reflection in cases where total
    internal reflection does and does not occur (angle > critical_angle).
    In the diffuse radiation, coefficients are precalculated because 
    ~256 gaussian points required for convergence.

    Args:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class
        mu0n: incidence angle for direct beam adjusted for refraction
        mu0: incidence angle of direct beam at upper surface
        nr: real part of refractive index
        rdif_a: layer reflectivity to diffuse radiation from above
        rdif_b: layer reflectivity to diffuse radiation from below
        tdif_a: layer transmittivity to diffuse radiation from above
        tdif_b: layer transmittivity to diffuse radiation from above
        trnlay: transmission of layer == lyr
        lyr: current layer (0 ==top)
        rdir: layer reflectivity to direct beam
        tdir: layer transmittivity to direct beam


    Returns:
        rdif_a: layer reflectivity to diffuse radiation from above
        rdif_b: layer reflectivity to diffuse radiation from below
        tdif_a: layer transmittivity to diffuse radiation from above
        tdif_b: layer transmittivity to diffuse radiation from above
        trnlay: transmission of layer == lyr
        rdir: layer reflectivity to direct beam
        tdir: layer transmittivity to direct beam
    """

    ref_indx = column.ref_idx_re + 1j * column.ref_idx_im
    critical_angle = np.arcsin(ref_indx)

    for wl in np.arange(0, column.nbr_wvl, 1):
        if np.arccos(irradiance.mu_not) < critical_angle[wl]:
            # in this case, no total internal reflection

            # compute fresnel reflection and transmission amplitudes
            # for two polarizations: 1=perpendicular and 2=parallel to
            # the plane containing incident, reflected and refracted rays.

            # Eq. 22  Briegleb & Light 2007
            # Inputs to equation 21 (i.e. Fresnel formulae for R and T)
            R1 = (mu0[wl] - nr[wl] * mu0n[wl]) / (
                mu0[wl] + nr[wl] * mu0n[wl]
            )  # reflection amplitude factor for perpendicular polarization
            R2 = (nr[wl] * mu0[wl] - mu0n[wl]) / (
                nr[wl] * mu0[wl] + mu0n[wl]
            )  # reflection amplitude factor for parallel polarization
            T1 = (
                2 * mu0[wl] / (mu0[wl] + nr[wl] * mu0n[wl])
            )  # transmission amplitude factor for perpendicular polarization
            T2 = (
                2 * mu0[wl] / (nr[wl] * mu0[wl] + mu0n[wl])
            )  # transmission amplitude factor for parallel polarization

            # unpolarized light for direct beam
            # Eq. 21  Brigleb and light 2007
            Rf_dir_a = 0.5 * (R1**2 + R2**2)
            Tf_dir_a = 0.5 * (T1**2 + T2**2) * nr[wl] * mu0n[wl] / mu0[wl]

        else:  # in this case, total internal reflection occurs
            Tf_dir_a = 0
            Rf_dir_a = 1

        # precalculated diffuse reflectivities and transmissivities
        # for incident radiation above and below fresnel layer, using
        # the direct albedos and accounting for complete internal
        # reflection from below. Precalculated because high order
        # number of gaussian points (~256) is required for convergence:

        # Eq. 25  Briegleb and light 2007
        # diffuse reflection of flux arriving from above

        # reflection from diffuse unpolarized radiation
        Rf_dif_a = column.fl_r_dif_a[wl]
        Tf_dif_a = 1 - Rf_dif_a  # transmission from diffuse unpolarized radiation

        # diffuse reflection of flux arriving from below
        Rf_dif_b = column.fl_r_dif_b[wl]
        Tf_dif_b = 1 - Rf_dif_b

        # -----------------------------------------------------------------------
        # the lyr = lyrfrsnl layer properties are updated to combine
        # the fresnel (refractive) layer, always taken to be above
        # the present layer lyr (i.e. be the top interface):

        # save fluxes of lyr before merging with frsnl layer
        rdif_a_0 = rdif_a[wl, lyr].copy()
        rdif_b_0 = rdif_b[wl, lyr].copy()
        tdif_a_0 = tdif_a[wl, lyr].copy()
        tdif_b_0 = tdif_b[wl, lyr].copy()
        tdir_0 = tdir[wl, lyr].copy()
        rdir_0 = rdir[wl, lyr].copy()

        # combined layer transmissivity to DIRECT radiation
        # Eq. B7  Briegleb & Light 2007
        tdir[wl, lyr] = (
            Tf_dir_a * tdir_0
            + Tf_dir_a * rdir[wl, lyr] * Rf_dif_b * tdif_a[wl, lyr]
            * 1 / (1 - Rf_dif_b * rdif_a_0)
        )

        # combined layer reflectivity to DIRECT radiation
        # Eq. B7  Briegleb & Light 2007
        rdir[wl, lyr] = (Rf_dir_a 
                          + Tf_dir_a 
                          * rdir_0
                          * Tf_dif_b 
                          * 1 / (1 - Rf_dif_b * rdif_a_0)
                          )
                         
        # combined layer reflectivity to DIFFUSE radiation (above)
        # Eq. B9  Briegleb & Light 2007
        rdif_a[wl, lyr] = (Rf_dif_a 
                            + Tf_dif_a 
                            * rdif_a_0
                            * Tf_dif_b
                            * 1 / (1 - Rf_dif_b * rdif_a_0)
                            )
        
        # combined layer reflectivity to DIFFUSE radiation (below)
        # Eq. B10  Briegleb & Light 2007
        rdif_b[wl, lyr] = (
            rdif_b_0
            + tdif_b_0 
            * Rf_dif_b 
            * tdif_a_0
            * 1 / (1 - Rf_dif_b * rdif_b_0)
        )
        
        # combined layer transmissivity to DIFFUSE radiation (above)
        # Eq. B9  Briegleb & Light 2007
        tdif_a[wl, lyr] = (tdif_a_0
                            * Tf_dif_a
                            * 1 / (1 - Rf_dif_b * rdif_a_0)
                            )

        # Eq. B10  Briegleb & Light 2007
        tdif_b[wl, lyr] = (tdif_b_0 
                            * Tf_dif_b
                            * 1 / (1 - Rf_dif_b * rdif_b_0)
                            )

        # update trnlay to include fresnel transmission (Eq. B8)
        trnlay[wl, lyr] = Tf_dir_a * trnlay[wl, lyr]

    return rdif_a, rdif_b, tdif_a, tdif_b, trnlay, rdir, tdir

def combine_layers_downward(
    lyr,
    trnlay,
    rdif_a,
    rdir,
    tdif_a,
    rdif_b,
    tdir,
    tdif_b,
    trndir,
    rdndif,
    trntdr,
    trndif,
):

    """Calculate energy going downward in the ice column:
        solar beam transmission, total transmission, diffuse transmission,
        and reflectivity to diffuse radiation arriving from below.
        The loop starts at the upper layer, working downwards.
        Equations are B2 & B5 from Briegleb & Light 2007.
    

    Args:
        lyr: integer representing the index of the current layer (0 at top)
        trnlay: transmissivity of current layer
        rdif_a: reflectivity to diffuse irradiance for radiation above
        rdir: reflectivity to direct beam
        tdif_a: transmissivity to diffuse irradiance for radiation above
        rdif_b: reflectivity to diffuse irradiance for radiation below
        tdir: transmissivity to direct beam
        tdif_b: transmissivity to diffuse irradiance for radiation below
        trndir: transmission of direct beam
        rdndif: reflectivity to diffuse radiation for all layers above current layer
        trntdr: total transmission of direct beam (diffuse + direct)
        trndif: diffuse transmission

    Returns:
        trndir: transmission of direct beam
        trntdr: total transmission of direct beam (diffuse + direct)
        rdndif: downwards diffuse reflectance
        trndif: diffuse transmission

    """
    
    # term below represents 1 / multiple scattering between layers 
    # rdndif is the combined reflectivity from all layers
    # above current layer to diffuse radiation coming from above
    # (1 - RBAR1 * RBAR2) in Eq. B2 from B&L 2007
    refkm1 = 1 / (1 - rdndif[:, lyr] * rdif_a[:, lyr])
    
    # transmission of solar beam (direct) 
    # trnlay = exp(-ts/mu_not), with ts changing every layer, mu0 is mu0n under fresnel lr 
    trndir[:, lyr + 1] = (
        trndir[:, lyr] * trnlay[:, lyr]
    ) 
    
    # total down diffuse = total transmission - direct transmission
    tdndif = trntdr[:, lyr] - trndir[:, lyr]
    
    # Eq. B2  Briegleb and Light 2007
    # total transmission for layers above
    trntdr[:, lyr + 1] = (
        trndir[:, lyr] * tdir[:, lyr]
        + (tdndif + trndir[:, lyr] * rdir[:, lyr] * rdndif[:, lyr]) 
        * refkm1 * tdif_a[:, lyr]
    )
    
    # reflectivity to diffuse radiation from below for layers above
    rdndif[:, lyr + 1] = rdif_b[:, lyr] + (
        tdif_b[:, lyr] * rdndif[:, lyr] * refkm1 * tdif_a[:, lyr]
    )
 
    # Eq. B5 (!! layers are not homogeneous so a =! b)
    trndif[:, lyr + 1] = trndif[:, lyr] * tdif_a[:, lyr] * refkm1

    return trndir, trntdr, rdndif, trndif

def combine_layers_upward(
    lyr,
    rdif_a,
    rdif_b,
    tdif_a,
    tdif_b,
    trnlay,
    rdir,
    tdir,
    rupdif,
    rupdir,
):
    """Combine energy going upward in the ice column:
        Compute reflectivity to direct (rupdir) and diffuse (rupdif) radiation
        arriving from above, for layers below current layer. 
        The loop starts from the second to last interface, working upwards.
        Equations are B2-B4 from Briegleb & Light 2007. 

    Args:
        ice: instance of Ice class
        rdif_a: reflectance to diffuse energy from above
        rdif_b: reflectance to diffuse energy from below
        tdif_a: transmittance to diffuse energy from above
        tdif_b: transmittance to diffuse energy from below
        trnlay: transmission of layer == lyr
        rdir: reflectance to direct beam
        tdir: transmission of direct beam
        rupdif: reflectivity to diffuse radiation coming from above for the combined layers
        rupdir: reflectivity to direct radiation coming from above for the combined layers

    Returns:
        rupdif: reflectivity to diffuse radiation coming from above for the combined layers
        rupdir: reflectivity to direct radiation coming from above for the combined layers

    """

    # starts at the bottom and works its way up to the top layer
    # interface scattering
    refkp1 = 1 / (1 - rdif_b[:, lyr] * rupdif[:, lyr + 1])

    # Eq. B2
    rupdir[:, lyr] = (
        rdir[:, lyr]
        + (
            trnlay[:, lyr] * rupdir[:, lyr + 1]
            + (tdir[:, lyr] - trnlay[:, lyr]) * rupdif[:, lyr + 1]
        )
        * refkp1
        * tdif_b[:, lyr]
    )

    # Eq. B4 (!! layers are not homogeneous so a =! b)
    rupdif[:, lyr] = (
        rdif_a[:, lyr]
        + tdif_a[:, lyr] * rupdif[:, lyr + 1] * refkp1 * tdif_b[:, lyr]
    )
    return rupdif, rupdir


def calculate_fluxes_at_interfaces(
            column,
            lyr,
            rupdif,
            rupdir,
            rdndif,
            trndir,
            trndif,
            trntdr,
            fdirup,
            fdirdn,
            fdifup,
            fdifdn,
            dfdir,
            dfdif,
):
    """Calculates up and down fluxes at layer interfaces.
    Equation B6 from Briegleb & Light 2007. 

    Args:
        column: instance of ColumnProperties class
        rupdif: total diffuse radiation reflected upwards
        rupdir: total direct radiation reflected upwards
        rdndif: downwards reflection of diffuse radiation
        trndir: transmission of direct radiation
        trndif: transmission of diffuse radiation
        trntdr: total transmission
        fdirup:
        fdirdn:
        fdifup:
        fdifdn:
        dfdir:
        dfdif:

    Returns:
        fdirup: upwards flux of direct radiation
        fdifup: upwards flux of diffuse radiation
        fdirdn: downwards flux of direct radiation
        fdifdn: downwards flux of diffuse radiation

    """

    puny = 1e-10  # not sure how should we define this

    # Eq. 52  Briegleb and Light 2007
    # interface scattering
    refk = 1 / (1 - rdndif[:, lyr] * rupdif[:, lyr])

    # Eq B6
    # dir tran ref from below times interface scattering, plus diff
    # tran and ref from below times interface scattering
    fdirup[:, lyr] = (
        trndir[:, lyr] * rupdir[:, lyr]
        + (trntdr[:, lyr] - trndir[:, lyr]) * rupdif[:, lyr]
    ) * refk

    # dir tran plus total diff trans times interface scattering plus
    # dir tran with up dir ref and down dif ref times interface scattering
    fdirdn[:, lyr] = (
        trndir[:, lyr]
        + (
            trntdr[:, lyr]
            - trndir[:, lyr]
            + trndir[:, lyr] * rupdir[:, lyr] * rdndif[:, lyr]
        )
        * refk
    )

    # diffuse tran ref from below times interface scattering
    fdifup[:, lyr] = trndif[:, lyr] * rupdif[:, lyr] * refk

    # diffuse tran times interface scattering
    fdifdn[:, lyr] = trndif[:, lyr] * refk

    # dfdir = fdirdn - fdirup
    dfdir[:, lyr] = (
        trndir[:, lyr]
        + (trntdr[:, lyr] - trndir[:, lyr]) * (1 - rupdif[:, lyr]) * refk
        - trndir[:, lyr] * rupdir[:, lyr] * (1 - rdndif[:, lyr]) * refk
    )

    if np.max(dfdir[:, lyr]) < puny:
        dfdir[:, lyr] = np.zeros(
            (column.nbr_wvl,), dtype=int
        )  # echmod necessary?
        # dfdif = fdifdn - fdifup

    dfdif[:, lyr] = trndif[:, lyr] * (1 - rupdif[:, lyr]) * refk

    if np.max(dfdif[:, lyr]) < puny:
        dfdif[:, lyr] = np.zeros(
            (column.nbr_wvl,), dtype=int
        )  # !echmod necessary?

    return fdirup, fdifup, fdirdn, fdifdn


def calculate_bulk_fluxes(
    column,
    irradiance,
    fdirup,
    fdifup,
    fdirdn,
    fdifdn,
    F_up,
    F_dwn,
    F_abs,
    F_abs_vis,
    F_abs_nir,
):
    """Calculates total fluxes in each layer and for entire column.

    Args:
        ice: instance of Ice class
        irradiance: instance of SolarFluxes class
        fdirup: upwards flux of direct radiation
        fdifup: upwards flux od diffuse radiation
        fdirdn: downwards flux of direct radiation
        fdifdn: downwards flux of diffuse radiation
        F_up:
        F_dwn:
        F_abs:
        F_abs_vis:
        F_abs_nir:

    Returns:
        albedo: ratio of upwards fluxes to incoming irradiance
        F_abs: absorbed flux in each layer
        F_btm_net: net fluxes at bottom surface
        F_top_pls: upwards flux from upper surface

    """

    for n in np.arange(0, column.nbr_lyr + 1, 1):
        F_up[:, n] = (
            fdirup[:, n] * (irradiance.Fs * irradiance.mu_not * np.pi)
            + fdifup[:, n] * irradiance.Fd
        )
        F_dwn[:, n] = (
            fdirdn[:, n] * (irradiance.Fs * irradiance.mu_not * np.pi)
            + fdifdn[:, n] * irradiance.Fd
        )

    F_net = F_up - F_dwn

    # Absorbed flux in each layer
    F_abs[:, :] = F_net[:, 1:] - F_net[:, :-1]

    # Upward flux at upper model boundary
    F_top_pls = F_up[:, 0]

    # Net flux at lower model boundary = bulk transmission through entire
    # media = absorbed radiation by underlying surface:
    F_btm_net = -F_net[:, column.nbr_lyr]

    albedo = F_up[:, 0] / F_dwn[:, 0]

    return albedo, F_abs, F_btm_net, F_top_pls


def conservation_of_energy_check(irradiance, F_abs, F_btm_net, F_top_pls):
    """Checks there is no conservation of energy violation.

    Args:
        irradiance: instance of irradiance class
        F_abs: absorbed flux in each layer
        F_btm_net: net flux at bottom surface
        F_top_pls: upwards flux from upper boundary

    Returns:
        None

    Raises:
        ValueError is conservation of energy error is detected

    """
    # Incident direct+diffuse radiation equals (absorbed+transmitted+bulk_reflected)
    energy_sum = (
        (irradiance.mu_not * np.pi * irradiance.Fs)
        + irradiance.Fd
        - (np.sum(F_abs, axis=1) + F_btm_net + F_top_pls)
    )

    energy_conservation_error = sum(abs(energy_sum))

    if energy_conservation_error > 1e-10:
        raise ValueError(f"energy conservation error: {energy_conservation_error}")
    else:
        pass


def get_outputs(column, irradiance, albedo, F_abs, F_btm_net):
    """Assimilates useful data into instance of Outputs class.

    Args:
        irradiance: instance of irradiance class
        albedo: ratio of upwwards fluxes and irradiance
        F_abs: absorbed flux in each layer
        F_btm_net: net flux at bottom surface

    Returns:
        outputs: instance of Outputs class

    """
    outputs = Outputs()

    # Radiative heating rate:
    F_abs_slr = np.sum(F_abs, axis=0)
    # [K/s] 2117 = specific heat column (J kg-1 K-1)
    heat_rt = F_abs_slr / (np.array(column.layer_mass) * 2117)
    outputs.heat_rt = heat_rt * 3600  # [K/hr]

    # Spectral albedo
    outputs.albedo = albedo

    # Spectrally-integrated solar, visible, and NIR albedos:
    outputs.BBA = np.sum(irradiance.flx_slr * albedo) / np.sum(irradiance.flx_slr)

    # Total incident insolation( Wm - 2)
    outputs.total_insolation = np.sum(
        (irradiance.mu_not * np.pi * irradiance.Fs) + irradiance.Fd
    )

    # Spectrally-integrated absorption by underlying surface:
    outputs.abs_slr_btm = np.sum(F_btm_net, axis=0)

    # Spectrally-integrated absorption by entire snow/column column
    outputs.abs_slr_tot = np.sum(F_abs_slr)

    # Spectrally-integrated absorption by each layer
    outputs.absorbed_flux_per_layer = F_abs_slr

    return outputs




if __name__ == "__main__":
    pass
