"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from PythonicDISORT.subroutines import interpolate as interp_u
from PythonicDISORT._assemble_intensity_and_fluxes import _assemble_intensity_and_fluxes
from PythonicDISORT import pydisort
import numpy as np


class _MultiStreamSolverDISORT:
    """
    Compute and store the variables necessary to solve the
    unpolarized radiative transfer equation using the DISORT algorithm via
    the PythonicDISORT package.

    Attributes
    ----------
    direct_irradiance : ndarray
        Direct spectral solar irradiance.
    diffuse_irradiance : ndarray
        Diffuse spectral solar irradiance.
    solar_flag : bool
        Controls whether to add a solar source.
    mu0 : float
        Cosine of the solar zenith angle.
    wavelengths : ndarray
        Wavelength array at which RTE is solved.
    nbr_wvl : ndarray
        Number of wavelengths.
    n_fourier : int
        Number of Fourier modes used in the expansion of the intensity in azimuth.
    n_streams : int
        Number of streams used for the Gaussian quadrature.
    nbr_lyr : int
        Number of layers for the total column.
    cos_angle : ndarray
        Nodes of the gaussian quadrature.
    cos_weight : ndarray
        Weights of the gaussian quadrature.
    output_polar_angles : array
        Array of polar angles for output.
    n_expansion : int
        Expansion order of Legendre series.
    only_fourier_m0 : bool
        Encodes whether the 0th moment is required only.
    output_levels : str
        Levels in the column at which to return the results.
    phi0 : float
        Solar azimuth angle.
    azimuth_angles : array
        Array of azimuth angles (degrees).
    relative_azimuths : ndarray
        Relative azimuth angle (viewing - solar).
    relative_azimuths_rad : ndarray
        Relative azimuth angle in radians (viewing - solar).
    t_od : ndarray
        Delta scaled spectral optical depth for all layers.
    w : ndarray
        Delta scaled spectral single scattering albedo for all layers.
    legendre_moments : ndarray
        Delta scaled Legendre moments for all layers.
    unscaled_tau : ndarray
        Unscaled spectral optical depth for all layers.
    unscaled_w : ndarray
        Unscaled spectral single scattering albedo for all layers.
    unscaled_legendre_moments : ndarray
        unscaled Legendre moments for all layers.
    surface_idx : int
        Index in the column at the interface atmosphere-land.
    scale_tau : ndarray
        Internal variable for PythonicDISORT
    albedo_toa : array
        Top of Atmosphere (TOA) albedo.
    albedo_boa : array
        Bottom of Atmosphere (BOA) albedo.
    flux_up_boa : array
        Flux upward at BOA.
    flux_down_boa : array
        Flux downward at BOA.
    directional_reflectance_toa_m0 : array
        Directional reflectance at TOA for 0th fourier moment.
    directional_radiance_toa_m0 : array
        Directional radiance at TOA for 0th fourier moment.
    directional_reflectance_toa_m0 : array
        Directional reflectance at BOA for 0th fourier moment.
    directional_radiance_toa_m0 : array
        Directional radiance at BOA for 0th fourier moment.
    directional_radiance_toa : array
        Directional radiance at TOA for all fourier moments.
    directional_reflectance_toa : array
        Directional reflectance at TOA for all fourier moments.
    directional_radiance_boa : array
        Directional radiance at BOA for all fourier moments.
    directional_reflectance_boa : array
        Directional reflectance at BOA for all fourier moments.
    """

    def __init__(
        self,
        land,
        atmosphere,
        irradiance,
        config,
    ):
        """
        Initialize all variables required for the solver and applies delta
        scaling to the single scattering properties of the ice/snow column.

        Parameters
        ----------
        land : LandColumn
            Instance of the LandColumn class, storing the physical
            and optical properties of the ice/snow column.
        atmosphere : AtmosphereColumn
            Instance of the AtmosphereColumn class, storing the physical
            and optical properties of the atmosphere column.
        irradiance : SolarIrradiance
            Instance of the SolarIrradiance class, storing the properties of the
            incoming solar irradiance.
        config : dictionary
            Solver parameters set in the input Yaml file.
        """

        self.direct_irradiance = irradiance.total_irradiance
        self.diffuse_irradiance = 0 * irradiance.diffuse / (np.pi * 4)
        self.mu0 = np.cos(np.deg2rad(irradiance.sza))
        self.n_streams = config.SOLVER.N_STREAMS
        self.n_fourier = config.SOLVER.N_FOURIER_MODES
        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.only_fourier_m0 = self.n_fourier == 1
        self.output_levels = config.SOLVER.OUTPUT_LEVELS

        self.wavelengths = config._wavelengths
        self.nbr_wvl = len(self.wavelengths)

        self.set_gaussian_quadrature()
        self.output_polar_angles = np.cos(
            np.deg2rad(np.arange(*config.SOLVER.POLAR_ANGLES))
        )

        if not self.only_fourier_m0:
            self.phi0 = np.deg2rad(irradiance.saa)
            self.azimuth_angles = np.arange(*config.SOLVER.AZIMUTH_ANGLES)
            self.relative_azimuths = np.abs(self.azimuth_angles - irradiance.saa)
            self.relative_azimuths_rad = np.deg2rad(self.relative_azimuths)
        else:
            self.phi0 = 0  # not used for n_fourier = 1 but must be prescribed

        if config.SOLVER.DELTA_SCALING == "M" or config.SOLVER.DELTA_SCALING == "M+":
            
            (
                tau_land,
                ss_alb_land,
                legendre_moments_land,
                tau_atm,
                ss_alb_atm,
                legendre_moments_atm,
                scale_factor,
            ) = self.apply_delta_scaling(atmosphere, land, config.SOLVER)

            self.scale_tau = scale_factor

        else:
            tau_land = np.array(land.tau)
            ss_alb_land = np.array(land.ss_alb)
            legendre_moments_land = np.array(
                land.legendre_moments[: land.n_expansion, :, :]
            )
            tau_atm = np.array(atmosphere.tau)
            ss_alb_atm = np.array(atmosphere.ss_alb)
            legendre_moments_atm = np.array(
                atmosphere.legendre_moments[: atmosphere.n_expansion, :, :]
            )

        if not atmosphere.use_atmosphere:
            self.nbr_lyr = land.nbr_lyr
            self.t_od = np.cumsum(tau_land, axis=0)
            self.w = ss_alb_land
            self.legendre_moments = legendre_moments_land
            self.unscaled_tau = np.cumsum(land.tau, axis=0)
            self.unscaled_w = np.array(land.ss_alb)
            self.unscaled_legendre_moments = np.array(land.legendre_moments)
            self.tau_surface = np.zeros(self.nbr_wvl)

        else:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.cumsum(np.vstack([tau_atm, tau_land]), axis=0)
            self.w = np.vstack([ss_alb_atm, ss_alb_land])
            self.legendre_moments = np.hstack(
                [legendre_moments_atm, legendre_moments_land]
            )
            self.unscaled_tau = np.cumsum(np.vstack([atmosphere.tau, land.tau]), axis=0)
            self.unscaled_w = np.vstack([atmosphere.ss_alb, land.ss_alb])
            self.unscaled_legendre_moments = np.hstack(
                [
                    atmosphere.legendre_moments,
                    land.legendre_moments,
                ]
            )
            self.tau_surface = self.unscaled_tau[-land.nbr_lyr - 1, :]

        self.albedo_toa = np.zeros(self.nbr_wvl)
        self.flux_up_boa = np.zeros(self.nbr_wvl)
        self.flux_down_boa = np.zeros(self.nbr_wvl)

        self.directional_reflectance_toa_m0 = np.zeros(
            (len(self.output_polar_angles), self.nbr_wvl)
        )
        self.directional_radiance_toa_m0 = np.zeros_like(
            self.directional_reflectance_toa_m0
        )

        self.albedo_boa = np.zeros(self.nbr_wvl)
        self.directional_reflectance_boa_m0 = np.zeros(
            (len(self.output_polar_angles), self.nbr_wvl)
        )
        self.directional_radiance_boa_m0 = np.zeros_like(
            self.directional_reflectance_boa_m0
        )

        if not self.only_fourier_m0:
            self.directional_radiance_toa = np.zeros(
                (
                    len(self.output_polar_angles),
                    self.nbr_wvl,
                    len(self.relative_azimuths),
                )
            )
            self.directional_reflectance_toa = np.zeros_like(
                self.directional_radiance_toa
            )
            self.directional_radiance_boa = np.zeros(
                (
                    len(self.output_polar_angles),
                    self.nbr_wvl,
                    len(self.relative_azimuths),
                )
            )
            self.directional_reflectance_boa = np.zeros_like(
                self.directional_radiance_boa
            )

    def set_gaussian_quadrature(self):
        """
        Set nodes and weights of gaussian quadrature in [0-1] (cos polar angle).
        """

        # generate nodes / weights in [-1:1] and then remap to [0-1]
        nodes, weights = np.polynomial.legendre.leggauss(int(self.n_streams // 2))
        self.cos_angle = 0.5 * (nodes + 1.0)
        self.cos_weight = 0.5 * weights


    def apply_delta_scaling(self, atmosphere, land, SOLVER):
        """
        Apply optional Delta-M or Delta-M+ scaling to expansion coefficients
        to truncate strongly forward-scattering phase functions.

        Parameters
        ----------
        land : LandColumn
            Instance of the LandColumn class, storing the physical
            and optical properties of the ice/snow column.
        atmosphere : AtmosphereColumn
            Instance of the AtmosphereColumn class, storing the physical
            and optical properties of the atmosphere column.
        irradiance : SolarIrradiance
            Instance of the SolarIrradiance class, storing the properties of the
            incoming solar irradiance.
        SOLVER : dictionary
            Solver parameters set in the input Yaml file.

        """

        # initialize and set in case atmosphere is not used
        tau_atm = None
        ss_alb_atm = None
        legendre_moments_atm = None
        
        if SOLVER.DELTA_SCALING == "M+":
            # Check from DISORT v.4.0.98
            if atmosphere.use_atmosphere: 
                if (
                        (atmosphere.legendre_moments[atmosphere.n_expansion, :, :] < 1e-4).any()
                        or (land.legendre_moments[land.n_expansion, :, :] < 1e-4).any()
                        or (
                            atmosphere.legendre_moments[atmosphere.n_expansion + 1, :, :]
                            < 0.7 * atmosphere.legendre_moments[atmosphere.n_expansion, :, :]
                            ).any()
                        or (
                            land.legendre_moments[land.n_expansion + 1, :, :]
                            < 0.7 * land.legendre_moments[land.n_expansion, :, :]
                            ).any()
                        ):
                    
                    raise ValueError("Delta-M+ scaling cannot be applied to Legendre moments - select Delta-M instead.")
            else: 
                if (
                        (land.legendre_moments[land.n_expansion, :, :] < 1e-4).any()
                        or (
                            land.legendre_moments[land.n_expansion + 1, :, :]
                            < 0.7 * land.legendre_moments[land.n_expansion, :, :]
                            ).any()
                        ):
                    
                    raise ValueError("Delta-M+ scaling cannot be applied to Legendre moments - select Delta-M instead.")
            
            sigma_sq = ((land.n_expansion + 1) ** 2 - land.n_expansion**2) / (
                np.log((land.legendre_moments[land.n_expansion]) ** 2)
                - np.log((land.legendre_moments[land.n_expansion + 1]) ** 2)
            )
            f = np.array(
                land.legendre_moments[land.n_expansion]
                * np.exp(land.n_expansion**2 / (2 * sigma_sq))
            )
            scale_factor = 1.0 - land.ss_alb * f
            legendre_moments_land = np.array(
                (
                    land.legendre_moments[: land.n_expansion, :, :]
                    - f[None, :, :]
                    * np.exp(
                        -(np.arange(land.n_expansion) ** 2)[:, None, None]
                        / (2 * sigma_sq)
                    )
                )
                / (1 - f[None, :, :])
            )
            tau_land = np.array((1.0 - land.ss_alb * f) * land.tau)
            ss_alb_land = np.array((1.0 - f) * land.ss_alb / (1 - land.ss_alb * f))

            if atmosphere.use_atmosphere:

                sigma_sq = (
                    (atmosphere.n_expansion + 1) ** 2 - atmosphere.n_expansion**2
                ) / (
                    np.log(
                        (
                            atmosphere.legendre_moments[
                                atmosphere.n_expansion, :, :
                            ]
                        )
                        ** 2
                    )
                    - np.log(
                        (
                            atmosphere.legendre_moments[
                                atmosphere.n_expansion + 1,
                                :,
                                :,
                            ]
                        )
                        ** 2
                    )
                )
                f = atmosphere.legendre_moments[
                    atmosphere.n_expansion,
                    :,
                ] * np.exp(atmosphere.n_expansion**2 / (2 * sigma_sq))

                legendre_moments_scaled = np.array(
                    (
                        atmosphere.legendre_moments[
                            : atmosphere.n_expansion, :, :
                        ]
                        - f[None, :, :]
                        * np.exp(
                            -(np.arange(atmosphere.n_expansion) ** 2)[:, None, None]
                            / (2 * sigma_sq)
                        )
                    )
                    / (1 - f[None, :, :])
                )
                tau_scaled = np.array(
                    (1.0 - atmosphere.ss_alb[:, :] * f)
                    * atmosphere.tau[:, :]
                )
                ss_alb_scaled = np.array(
                    (1.0 - f)
                    * atmosphere.ss_alb[:, :]
                    / (1 - atmosphere.ss_alb[:, :] * f)
                )

                legendre_moments_atm = np.hstack(
                    [
                        atmosphere.legendre_moments[
                            : atmosphere.n_expansion, :, :
                        ],
                        legendre_moments_scaled,
                    ]
                )

                tau_atm = np.vstack(
                    [atmosphere.tau[:, :], tau_scaled]
                )
                ss_alb_atm = np.vstack(
                    [atmosphere.ss_alb[:, :], ss_alb_scaled]
                )

                scale_factor = np.vstack(
                    [
                        np.ones((atmosphere.nbr_lyr, self.nbr_wvl)),
                        (1.0 - atmosphere.ss_alb[:, :] * f),
                        scale_factor,
                    ]
                )
                
        elif SOLVER.DELTA_SCALING == "M":
            # Delta truncation: get highest Legendre term following
            # Wicombe 1977 Eq. (15) - 2M = N_MOMENTS
            f = np.array(land.legendre_moments[land.n_expansion])
            scale_factor = 1.0 - land.ss_alb * f
            legendre_moments_land = np.array(
                (land.legendre_moments[: land.n_expansion, :, :] - f[None, :, :])
                / (1 - f[None, :, :])
            )
            tau_land = np.array((1.0 - land.ss_alb * f) * land.tau)
            ss_alb_land = np.array((1.0 - f) * land.ss_alb / (1 - land.ss_alb * f))

            if atmosphere.use_atmosphere:
                # could be applied only from aerosol boundary down as no effect in rayleigh layers
                f = np.array(atmosphere.legendre_moments[atmosphere.n_expansion])
                legendre_moments_atm = np.array(
                    (
                        atmosphere.legendre_moments[: atmosphere.n_expansion, :, :]
                        - f[None, :, :]
                    )
                    / (1 - f[None, :, :])
                )
                tau_atm = np.array((1.0 - atmosphere.ss_alb * f) * atmosphere.tau)
                ss_alb_atm = np.array(
                    (1.0 - f) * atmosphere.ss_alb / (1 - atmosphere.ss_alb * f)
                )

                # update scale factor
                scale_factor = np.vstack([(1.0 - atmosphere.ss_alb * f), scale_factor])

                
        return (
            tau_land,
            ss_alb_land,
            legendre_moments_land,
            tau_atm,
            ss_alb_atm,
            legendre_moments_atm,
            scale_factor,
        )
    
    def fill_outputs(self, outputs_wl, wl_idx):
        """
        Fill outputs at a given wavelength depending on user
        inputs, in preparation for get_outputs.

        """

        if len(outputs_wl) == 5:
            mu, flux_up, flux_down, u0, u = outputs_wl

        elif len(outputs_wl) == 4:
            if self.n_fourier == 1:
                mu, flux_up, flux_down, u0 = outputs_wl
            elif self.n_fourier > 1:
                flux_up, flux_down, u0, u = outputs_wl

        elif len(outputs_wl) == 3:
            flux_up, flux_down, u0 = outputs_wl

        if "TOA" in self.output_levels:

            # calculate all fluxes at TOA ie tau = 0
            # sum downward flux (diff + dir)
            self.albedo_toa[wl_idx] = flux_up(0) / np.sum(flux_down(0))

            # intensity function only in upward angles
            # radiance in Wm-2nm-1sr-1
            self.directional_radiance_toa_m0[:, wl_idx] = interp_u(u0)(
                self.output_polar_angles,  # interpolate at user angles
                0,  # TOA
            )

            self.directional_reflectance_toa_m0[:, wl_idx] = (
                self.directional_radiance_toa_m0[:, wl_idx]
                * np.pi
                / np.sum(flux_down(0))
            )

            if self.n_fourier > 1:
                self.directional_radiance_toa[:, wl_idx, :] = interp_u(u)(
                    self.output_polar_angles,  # interpolate at user angles
                    0,  # TOA
                    self.phi0 + self.relative_azimuths_rad,
                )  # phi = phi0 + delta_phi

                self.directional_reflectance_toa[:, wl_idx, :] = (
                    self.directional_radiance_toa[:, wl_idx, :]
                    * np.pi
                    / np.sum(flux_down(0))
                )

        if "BOA" in self.output_levels:
            
            self.flux_up_boa[wl_idx] = flux_up(self.tau_surface[wl_idx])
            
            self.flux_down_boa[wl_idx] = np.sum(flux_down(self.tau_surface[wl_idx]))

            # intensity function only in upward angles
            self.directional_radiance_boa_m0[:, wl_idx] = interp_u(u0)(
                self.output_polar_angles,  # interpolate at user angles
                self.tau_surface[wl_idx],
            )

            self.directional_reflectance_boa_m0[:, wl_idx] = (
                self.directional_radiance_boa_m0[:, wl_idx]
                * np.pi
                / (self.flux_down_boa[wl_idx])
            )

            # calculate all fluxes at BOA
            # sum downward flux (diff + dir)
            if self.n_fourier > 1:
                self.directional_radiance_boa[:, wl_idx, :] = interp_u(u)(
                    self.output_polar_angles,  # interpolate at user angles
                    self.tau_surface[wl_idx],
                    self.phi0 + self.relative_azimuths_rad,
                )  # phi = phi0 + delta_phi
                self.directional_reflectance_boa[:, wl_idx, :] = (
                    self.directional_radiance_boa[:, wl_idx, :]
                    * np.pi
                    / np.sum(flux_down(self.flux_down_boa[wl_idx]))
                )

    def get_outputs(self):
        """
        Compile and return radiative transfer results at required output levels.

        Returns
        -------
        results : dictionary
            Multi-stream solver results.

        """

        # dictionnary with outputs depending on user inputs
        results = {}

        results["polar_angle"] = np.rad2deg(np.arccos(self.output_polar_angles))

        if self.n_fourier > 1:
            results["azimuth_angle"] = self.azimuth_angles

        if "BOA" in self.output_levels:
            results["albedo_boa"] = self.flux_up_boa / self.flux_down_boa

            results["bba_boa"] = np.trapezoid(
                self.flux_up_boa, x=self.wavelengths
            ) / np.trapezoid(self.flux_down_boa, x=self.wavelengths)

            results["directional_reflectance_boa_m0"] = (
                self.directional_reflectance_boa_m0
            )
            results["directional_radiance_boa_m0"] = self.directional_radiance_boa_m0
            
            if self.n_fourier > 1:
                
                # radiance as a func of phi & mu at the bottom of the atmosphere (BOA)
                results["directional_reflectance_boa"] = (
                    self.directional_reflectance_boa
                )
                results["directional_radiance_boa"] = self.directional_radiance_boa

        if "TOA" in self.output_levels:
            results["albedo_toa"] = self.albedo_toa

            results["directional_radiance_toa_m0"] = self.directional_radiance_toa_m0

            results["directional_reflectance_toa_m0"] = (
                self.directional_reflectance_toa_m0
            )
            
            if self.n_fourier > 1:

                # radiance as a func of phi & mu at the top of the atmosphere (TOA)
                results["directional_radiance_toa"] = self.directional_radiance_toa
    
                # reflectance as a func of phi & mu at the top of the atmosphere (TOA)
                results["directional_reflectance_toa"] = self.directional_reflectance_toa

        return results


def solve_multi_stream_rt_disort(land, atmosphere, irradiance, config):
    """

    Compute upward and downward radiances for a column of homogeneous layers.

    Parameters
    ----------
    land : LandColumn
        Instance of the LandColumn class, storing the physical
        and optical properties of the ice/snow column.
    atmosphere : AtmosphereColumn
        Instance of the AtmosphereColumn class, storing the physical
        and optical properties of the atmosphere column.
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class, storing the properties of the
        incoming solar irradiance.
    config : dictionary
        Solver parameters set in the input Yaml file.

    Returns
    -------
    outputs : dictionary
        Results of the solvers (radiance/reflectance/albedo at TOA/BOA).
    """

    mssd = _MultiStreamSolverDISORT(land, atmosphere, irradiance, config)

    for wl_idx in range(mssd.nbr_wvl):

        rescale_factor = np.max(
            (mssd.direct_irradiance[wl_idx], mssd.diffuse_irradiance[wl_idx])
        )
        if rescale_factor != 0:
            I0 = (mssd.direct_irradiance[wl_idx] / rescale_factor).copy()
            b_neg = (mssd.diffuse_irradiance[wl_idx] / rescale_factor).copy()

        outputs_wl = _assemble_intensity_and_fluxes(
            scaled_omega_arr=mssd.w[:, wl_idx],
            scale_tau=mssd.scale_tau[:, wl_idx],  # factor scaling tau
            tau_arr=mssd.unscaled_tau[:, wl_idx],  # unscaled tau
            scaled_tau_arr_with_0=np.insert(mssd.t_od, 0, 0, axis=0)[
                :, wl_idx
            ],  # scaled cumsum tau with 0 at TOA
            use_banded_solver_NLayers=10,  # default is 10
            mu_arr_pos=mssd.cos_angle,
            M_inv=1 / mssd.cos_angle,
            W=mssd.cos_weight,
            N=int(mssd.n_streams // 2),
            NQuad=mssd.n_streams,
            NLeg=mssd.n_expansion,
            NFourier=mssd.n_fourier,
            NLayers=mssd.nbr_lyr,
            is_atmos_multilayered=(mssd.nbr_lyr > 1),
            weighted_scaled_Leg_coeffs=(
                mssd.legendre_moments
                * (2 * np.arange(mssd.n_expansion) + 1)[:, None, None]
            )[:, :, wl_idx].T,
            mu0=mssd.mu0,
            I0=I0,  # direct beam scaled
            I0_div_4pi=I0 / (4 * np.pi),
            rescale_factor=rescale_factor,  # internal pythonic disort var
            b_neg=b_neg,  # fisot
            phi0=mssd.phi0,
            there_is_beam_source=True,
            only_flux=mssd.only_fourier_m0,
            # unused arguments
            BDRF_Fourier_modes=[],  # not used
            NBDRF=0,  # not used = len(BDRF_Fourier_modes)
            b_pos=0,  # not used
            b_pos_is_scalar=True,  # not used
            b_neg_is_scalar=True,  # not used
            b_pos_is_vector=False,  # not used
            b_neg_is_vector=False,  # not used
            Nscoeffs=0,  # not used
            scaled_s_poly_coeffs=np.atleast_2d([]),  # not used
            there_is_iso_source=False,  # not used
            autograd_compatible=False,  # not used
        )

        # prepare outputs per wavelength
        mssd.fill_outputs(outputs_wl, wl_idx)

    # format final outputs
    outputs = mssd.get_outputs()

    return outputs


def solve_multi_stream_rt_disort_wrapper(
    land, atmosphere, irradiance, config, NT_cor=True
):
    """

    Compute upward and downward radiances for a column of homogeneous layers
    using upper-level function of PythonicDISORT.

    Parameters
    ----------
    land : LandColumn
        Instance of the LandColumn class, storing the physical
        and optical properties of the ice/snow column.
    atmosphere : AtmosphereColumn
        Instance of the AtmosphereColumn class, storing the physical
        and optical properties of the atmosphere column.
    irradiance : SolarIrradiance
        Instance of the SolarIrradiance class, storing the properties of the
        incoming solar irradiance.
    config : dictionary
        Solver parameters set in the input Yaml file.

    Returns
    -------
    outputs : dictionary
        Results of the solvers (radiance/reflectance/albedo at TOA/BOA).
    """

    mssd = _MultiStreamSolverDISORT(land, atmosphere, irradiance, config)

    for wl_idx in range(mssd.nbr_wvl):

        outputs_wl = pydisort(
            tau_arr=mssd.unscaled_tau[:, wl_idx],
            omega_arr=mssd.unscaled_w[:, wl_idx],
            NQuad=mssd.n_streams,
            Leg_coeffs_all=mssd.unscaled_legendre_moments[:, :, wl_idx].T,
            mu0=mssd.mu0,
            I0=mssd.direct_irradiance[wl_idx],
            b_neg=mssd.diffuse_irradiance[wl_idx],
            phi0=mssd.phi0,
            NFourier=mssd.n_fourier,
            only_flux=mssd.only_fourier_m0,
            f_arr=mssd.unscaled_legendre_moments[mssd.n_expansion, :, wl_idx],
            use_banded_solver_NLayers=10, # default is 10
            NT_cor=NT_cor,
            NLeg=mssd.n_expansion,
            # b_pos = 0, # dirichlet condition
            # BRDF_Fourier_modes=[],
            # s_poly_coeffs=array([], shape=(1, 0), dtype=float64),
            # autograd_compatible=False
        )

        # prepare outputs per wavelength
        mssd.fill_outputs(outputs_wl, wl_idx)

    # format final outputs
    outputs = mssd.get_outputs()

    return outputs
