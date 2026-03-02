"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from PythonicDISORT import pydisort
from PythonicDISORT.subroutines import interpolate as interp_u
from PythonicDISORT._assemble_intensity_and_fluxes import _assemble_intensity_and_fluxes

import numpy as np


class _MultiStreamSolverDISORT:
    """
    Compute and store the variables necessary to solve the
    unpolarized radiative transfer equation using the PythonicDISORT python
    package, an implementation of the DISORT solver.
    
    Attributes
    ----------
    solar_irradiance : ndarray
        Total spectral solar irradiance.
    """
    
    def __init__(
        self,
        land,
        atmosphere,
        irradiance,
        SOLVER,
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
        SOLVER : dictionary
            Solver parameters set in the input Yaml file.
        """
        
        self.irradiance = irradiance.flx_slr
        self.mu0 = np.cos(np.deg2rad(irradiance.sza))
        self.nbr_wvl = len(irradiance.flx_slr)
        self.n_streams = SOLVER.N_STREAMS
        self.n_fourier = SOLVER.N_FOURIER_MODES
        self.n_expansion = SOLVER.N_LEGENDRE_MOMENTS
        self.only_fourier_m0 = (self.n_fourier==1)
        self.output_levels = SOLVER.OUTPUT_LEVELS
        
        self.set_gaussian_quadrature()
        
        ## INPUTS TO DECIDE ON: 
            # phi0 and phi are needed if IMS correction is turned on, otherwise
            # only the relative angle is important.
            # do we keep rel. azimuth in yaml or do we use phi0/phi ?
            # do we have IMS as optional?
            # how can we prescribe a user-input polar angle array?
        self.relative_azimuths = np.arange(*SOLVER.RELATIVE_AZIMUTH)
        self.relative_azimuths_rad = np.deg2rad(self.relative_azimuths)
        self.IMS_TMS_correction = True
        self.phi0 = 0
        self.output_polar_angles = np.cos(np.deg2rad(np.arange(20, 80, 10)))
        
        if SOLVER.DELTA_SCALING == "M" or SOLVER.DELTA_SCALING == "M+":
            (
                tau_land,
                ss_alb_land,
                legendre_moments_land,
                tau_atm,
                ss_alb_atm,
                legendre_moments_atm,
                scale_factor
            ) = self.apply_delta_scaling(atmosphere, land, SOLVER)
            
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
            # in DISORT the optical depth of each layer is calculated as
            # tau(n) - tau(n-1) so we need to cumsum
            self.t_od = np.cumsum(tau_land, axis=0) 
            self.w = ss_alb_land
            self.legendre_moments = legendre_moments_land
            self.tau_surface = self.t_od[0, :]
            self.unscaled_tau = np.cumsum(land.tau, axis=0) 

        else:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.cumsum(np.vstack([tau_atm, tau_land]), axis=0)
            self.w = np.vstack([ss_alb_atm, ss_alb_land])
            self.legendre_moments = np.hstack(
                [legendre_moments_atm, legendre_moments_land]
            )
            self.unscaled_tau = np.cumsum(np.vstack([atmosphere.tau, 
                                                     land.tau]), axis=0)
            self.tau_surface = self.unscaled_tau[-land.nbr_lyr - 1, :]

    def set_gaussian_quadrature(self):
        '''
        Set nodes and weights of gaussian quadrature in [0-1] (cos polar angle).

        '''

        # generate nodes / weights in [-1:1] and then remap to [0-1]
        nodes, weights = np.polynomial.legendre.leggauss(int(self.n_streams // 2))
        self.cos_angle = 0.5 * (nodes + 1.0)
        self.cos_weight = 0.5 * weights
            
    def apply_delta_scaling(self, atmosphere, land, SOLVER):
        '''
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

        '''

        # initialize and set in case atmosphere is not used
        tau_atm = None
        ss_alb_atm = None
        legendre_moments_atm = None


        if SOLVER.DELTA_SCALING == "M":
            # apply delta scaling to land column only -> HG function (!)
            # Delta truncation: get highest Legendre term following
            # Wicombe 1977 Eq. (15) - 2M = N_MOMENTS
            f = np.array(land.legendre_moments[land.n_expansion])
            scale_factor = (1.0 - land.ss_alb * f)
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

        elif SOLVER.DELTA_SCALING == "M+":

            # sigma_sq cannot get negative with HG function so as long as we
            # use HG we don't need to check that the scaling is applicable
            sigma_sq = ((land.n_expansion + 1) ** 2 - land.n_expansion**2) / (
                np.log((land.legendre_moments[land.n_expansion]) ** 2)
                - np.log((land.legendre_moments[land.n_expansion + 1]) ** 2)
            )
            f = np.array(
                land.legendre_moments[land.n_expansion]
                * np.exp(land.n_expansion**2 / (2 * sigma_sq))
            )
            scale_factor = (1.0 - land.ss_alb * f)
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

                # boundary aerosol layer, where not only rayleigh ie moment 1 is not 0

                # if rayleigh scattering only, no need to scale
                if atmosphere.AOD == 0:
                    ss_alb_atm = np.array(atmosphere.ss_alb)
                    tau_atm = np.array(atmosphere.tau)
                    legendre_moments_atm = np.array(
                        atmosphere.legendre_moments[: atmosphere.n_expansion, :, :]
                    )
                    
                    scale_factor = np.vstack([np.zeros((atmosphere.nbr_lyr, 
                                                        atmosphere.nbr_wvl)), 
                                              scale_factor])

                if atmosphere.AOD > 0:

                    boundary_layer_aerosols = np.where(
                        atmosphere.legendre_moments[1, :, 0] != 0.0
                    )[0][0]

                    # ! DISORT reverts to Delta-M+ under conditions below
                    if (
                        atmosphere.legendre_moments[
                            atmosphere.n_expansion, boundary_layer_aerosols:, :
                        ]
                        < 1e-4
                    ).any() or (
                        atmosphere.legendre_moments[
                            atmosphere.n_expansion + 1, boundary_layer_aerosols:, :
                        ]
                        < 0.7
                        * atmosphere.legendre_moments[
                            atmosphere.n_expansion, boundary_layer_aerosols:, :
                        ]
                    ).any():
                        print("WARNING with Delta-M+ scaling.")

                    sigma_sq = (
                        (atmosphere.n_expansion + 1) ** 2 - atmosphere.n_expansion**2
                    ) / (
                        np.log(
                            (
                                atmosphere.legendre_moments[
                                    atmosphere.n_expansion, boundary_layer_aerosols:, :
                                ]
                            )
                            ** 2
                        )
                        - np.log(
                            (
                                atmosphere.legendre_moments[
                                    atmosphere.n_expansion + 1,
                                    boundary_layer_aerosols:,
                                    :,
                                ]
                            )
                            ** 2
                        )
                    )
                    f = atmosphere.legendre_moments[
                        atmosphere.n_expansion,
                        boundary_layer_aerosols:,
                    ] * np.exp(atmosphere.n_expansion**2 / (2 * sigma_sq))

                    legendre_moments_scaled = np.array(
                        (
                            atmosphere.legendre_moments[
                                : atmosphere.n_expansion, boundary_layer_aerosols:, :
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
                        (1.0 - atmosphere.ss_alb[boundary_layer_aerosols:, :] * f)
                        * atmosphere.tau[boundary_layer_aerosols:, :]
                    )
                    ss_alb_scaled = np.array(
                        (1.0 - f)
                        * atmosphere.ss_alb[boundary_layer_aerosols:, :]
                        / (1 - atmosphere.ss_alb[boundary_layer_aerosols:, :] * f)
                    )

                    legendre_moments_atm = np.hstack(
                        [
                            atmosphere.legendre_moments[
                                : atmosphere.n_expansion, :boundary_layer_aerosols, :
                            ],
                            legendre_moments_scaled,
                        ]
                    )

                    tau_atm = np.vstack(
                        [atmosphere.tau[:boundary_layer_aerosols, :], tau_scaled]
                    )
                    ss_alb_atm = np.vstack(
                        [atmosphere.ss_alb[:boundary_layer_aerosols, :], ss_alb_scaled]
                    )
                    
                    scale_factor = np.vstack(
                        [np.zeros((boundary_layer_aerosols,
                                   atmosphere.nbr_wvl)), 
                         (1.0 - atmosphere.ss_alb[boundary_layer_aerosols:, :] * f),
                         scale_factor])

        return (
            tau_land,
            ss_alb_land,
            legendre_moments_land,
            tau_atm,
            ss_alb_atm,
            legendre_moments_atm,
            scale_factor
        )
    
    def solve_multi_stream_rt_v2(self):
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
        SOLVER : dictionary
            Solver parameters set in the input Yaml file.
            
        Returns 
        -------
        outputs : dictionary
            Results of the solvers (radiance/reflectance/albedo at TOA/BOA).
        """ 
        
        self.albedo_toa = np.zeros(self.nbr_wvl)  
        self.directional_reflectance_toa_m0 = np.zeros((len(self.output_polar_angles), 
                                                   self.nbr_wvl))
        self.directional_radiance_toa_m0 = np.zeros_like(
        self.directional_reflectance_toa_m0)
        self.directional_radiance_toa = np.zeros((len(self.output_polar_angles),
                                             len(self.relative_azimuths),
                                             self.nbr_wvl))
        self.directional_reflectance_toa = np.zeros_like(
        self.directional_radiance_toa)
        
        self.albedo_boa = np.zeros(self.nbr_wvl)  
        self.directional_reflectance_boa_m0 = np.zeros((len(self.output_polar_angles), 
                                                   self.nbr_wvl))
        self.directional_radiance_boa_m0 = np.zeros_like(
        self.directional_reflectance_boa_m0)
        
        for wl_idx in range(self.nbr_wvl): 
            
            if self.n_fourier == 1 : 

                flux_up, flux_down, u0 = _assemble_intensity_and_fluxes(
                    scaled_omega_arr = self.w[:, wl_idx],
                    scale_tau = self.scale_tau[:, wl_idx], # factor scaling tau
                    tau_arr = self.unscaled_tau[:, wl_idx], # unscaled tau
                    scaled_tau_arr_with_0 = np.insert(self.t_od, 0, 0, axis=0)[:, wl_idx], # scaled cumsum tau with 0 at TOA
                    use_banded_solver_NLayers = 10, # default is 10, we will have to test it
                    mu_arr_pos = self.cos_angle, 
                    M_inv = 1 / self.cos_angle, 
                    W = self.cos_weight,
                    N = int(self.n_streams // 2), 
                    NQuad = self.n_streams, 
                    NLeg = self.n_expansion,
                    NFourier = self.n_fourier, 
                    NLayers = self.nbr_lyr, 
                    is_atmos_multilayered = (self.nbr_lyr > 1),
                    weighted_scaled_Leg_coeffs = (
                        self.legendre_moments 
                        * (2 * np.arange(self.n_expansion) + 1)[:, None, None]
                        )[:, :, wl_idx].T,
                    mu0 = self.mu0, 
                    I0 = self.irradiance[wl_idx], 
                    I0_div_4pi = self.irradiance[wl_idx] / (4 * np.pi), 
                    rescale_factor = self.irradiance[wl_idx], 
                    phi0 = self.phi0,
                    there_is_beam_source = True,
                    only_flux = self.only_fourier_m0,
                    
                    # unused arguments
                    BDRF_Fourier_modes = [], # not used
                    NBDRF = 0, # not used = len(BDRF_Fourier_modes)
                    b_pos = 0, # not used
                    b_neg = 0, # not used
                    b_pos_is_scalar = True, # not used
                    b_neg_is_scalar = True, # not used
                    b_pos_is_vector = False, # not used
                    b_neg_is_vector = False, # not used
                    Nscoeffs = 0, # not used
                    scaled_s_poly_coeffs = np.atleast_2d([]), # not used 
                    there_is_iso_source = False, # not used
                    autograd_compatible = False, # not used
                )
                
                if "TOA" in self.output_levels:
                    # calculate all fluxes at TOA ie tau = 0
                    # sum downward flux (diff + dir)
                    self.albedo_toa[wl_idx] = flux_up(0) / np.sum(flux_down(0))
                    # intensity function only in upward angles
                    self.directional_radiance_toa_m0[:, wl_idx] = (
                        interp_u(u0)(self.output_polar_angles, # interpolate at user angles
                                      0, # TOA
                                      )
                        )
                    self.directional_reflectance_toa_m0[:, wl_idx] = (
                        self.directional_radiance_toa_m0[:, wl_idx] * np.pi / np.sum(flux_down(0))
                        )
                    
                if "BOA" in self.output_levels:
                    # calculate all fluxes at BOA
                    # sum downward flux (diff + dir)
                    self.albedo_boa[wl_idx] = (flux_up(self.tau_surface[wl_idx]) 
                                               / np.sum(flux_down(self.tau_surface[wl_idx])
                                                        )
                                               )
                    # intensity function only in upward angles
                    self.directional_radiance_boa_m0[:, wl_idx] = (
                        interp_u(u0)(self.output_polar_angles, # interpolate at user angles
                                     self.tau_surface[wl_idx], # TOA
                                      )
                        )
                    self.directional_reflectance_boa_m0[:, wl_idx] = (
                        self.directional_radiance_boa_m0[:, wl_idx] * np.pi / np.sum(flux_down(0))
                        )
            
                                
                
            else: 
                
                flux_up, flux_down, u0, u = _assemble_intensity_and_fluxes(
                    scaled_omega_arr = self.w[:, wl_idx],
                    scale_tau = self.scale_tau[:, wl_idx], # factor scaling tau
                    tau_arr = self.unscaled_tau[:, wl_idx], # unscaled tau
                    scaled_tau_arr_with_0 = np.insert(self.t_od, 0, 0, axis=0)[:, wl_idx], # scaled cumsum tau with 0 at TOA
                    use_banded_solver_NLayers = 10, # default is 10, we will have to test it
                    mu_arr_pos = self.cos_angle, 
                    M_inv = 1 / self.cos_angle, 
                    W = self.cos_weight,
                    N = int(self.n_streams // 2), 
                    NQuad = self.n_streams, 
                    NLeg = self.n_expansion,
                    NFourier = self.n_fourier, 
                    NLayers = self.nbr_lyr, 
                    is_atmos_multilayered = (self.nbr_lyr > 1),
                    weighted_scaled_Leg_coeffs = (
                        self.legendre_moments 
                        * (2 * np.arange(self.n_expansion) + 1)[:, None, None]
                        )[:, :, wl_idx].T,
                    mu0 = self.mu0, 
                    I0 = self.irradiance[wl_idx], 
                    I0_div_4pi = self.irradiance[wl_idx] / (4 * np.pi), 
                    rescale_factor = self.irradiance[wl_idx], 
                    phi0 = self.phi0,
                    there_is_beam_source = True,
                    only_flux = self.only_fourier_m0,
                    
                    # unused arguments
                    BDRF_Fourier_modes = [], # not used
                    NBDRF = 0, # not used = len(BDRF_Fourier_modes)
                    b_pos = 0, # not used
                    b_neg = 0, # not used
                    b_pos_is_scalar = True, # not used
                    b_neg_is_scalar = True, # not used
                    b_pos_is_vector = False, # not used
                    b_neg_is_vector = False, # not used
                    Nscoeffs = 0, # not used
                    scaled_s_poly_coeffs = np.atleast_2d([]), # not used 
                    there_is_iso_source = False, # not used
                    autograd_compatible = False, # not used
                )
                
                if "TOA" in self.output_levels:
                    # calculate all fluxes at TOA ie tau = 0
                    # sum downward flux (diff + dir)
                    self.albedo_toa[wl_idx] = flux_up(0) / np.sum(flux_down(0))
                    # intensity function only in upward angles
                    self.directional_radiance_toa_m0[:, wl_idx] = (
                        interp_u(u0)(self.output_polar_angles, # interpolate at user angles
                                      0, # TOA
                                      )
                        )
                    
                    self.directional_reflectance_toa_m0[:, wl_idx] = (
                        self.directional_radiance_toa_m0[:, wl_idx] * np.pi / np.sum(flux_down(0))
                        )
                    self.directional_radiance_toa[:, :, wl_idx] = (
                        interp_u(u)(self.output_polar_angles, # interpolate at user angles
                                      0, # TOA 
                                      self.phi0 + self.relative_azimuths_rad) # phi = phi0 + delta_phi
                        )
                    self.directional_reflectance_toa[:, :,  wl_idx] = (
                        self.directional_radiance_toa[:, :,  wl_idx] * np.pi / np.sum(flux_down(0))
                        )
                    
                
    

# #%% test 
# import matplotlib.pyplot as plt
# from snicarfx.core import Session
# input_file = "/Users/au660413/Documents/openosmia/snicar-fx/src/snicarfx/inputs.yaml"
# session = Session(input_file)
# results = session.run()

# #%%
# plt.plot(results.wavelength, results.albedo_boa)

# # plt.plot(results.wavelength, solver.albedo_toa, '--')
# plt.plot(results.wavelength, solver.albedo_boa, '--')




        