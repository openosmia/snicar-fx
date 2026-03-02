"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from PythonicDISORT import pydisort
from PythonicDISORT.subroutines import interpolate as interp_u
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
        

        if not atmosphere.use_atmosphere:
            self.nbr_lyr = land.nbr_lyr
            # in DISORT the optical depth of each layer is calculated as
            # tau(n) - tau(n-1) so we need to cumsum
            self.t_od = np.cumsum(land.tau, axis=0) 
            self.w = land.ss_alb
            self.legendre_moments = land.legendre_moments
            self.tau_surface = self.t_od[0, :]

        else:
            self.nbr_lyr = land.nbr_lyr + atmosphere.nbr_lyr
            self.t_od = np.cumsum(np.vstack([atmosphere.tau, land.tau]), axis=0)
            self.w = np.vstack([atmosphere.ss_alb, land.ss_alb])
            self.legendre_moments = np.hstack(
                [atmosphere.legendre_moments, land.legendre_moments]
            )
            self.tau_surface = self.t_od[-land.nbr_lyr - 1, :]
            
            
        if SOLVER.DELTA_SCALING == "M":
            self.truncation_factor = self.legendre_moments[self.n_streams, :, :] 
        else: 
            self.truncation_factor = None
            

    def solve_multi_stream_rt(self):
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
        
        
        for wl_idx in range(self.nbr_wvl): 
            
            if self.n_fourier == 1 : 
                mu, flux_up, flux_down, u0 = pydisort(
                    tau_arr=self.t_od[:, wl_idx], 
                    omega_arr=self.w[:, wl_idx], 
                    NQuad=self.n_streams,
                    Leg_coeffs_all=self.legendre_moments[:, :, wl_idx].T, 
                    mu0=self.mu0, 
                    I0=self.irradiance[wl_idx], 
                    phi0=self.phi0, 
                    NLeg=self.n_expansion,
                    NFourier=self.n_fourier,
                    # b_pos = 0,
                    # b_neg = 0,
                    only_flux = self.only_fourier_m0,
                    f_arr=self.truncation_factor[:, wl_idx],
                    NT_cor=self.IMS_TMS_correction,
                    # BRDF_Fourier_modes=[],
                    # s_poly_coeffs=array([], shape=(1, 0), dtype=float64),
                    # use_banded_solver_NLayers=10, 
                    # autograd_compatible=False
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
            else:
                mu, flux_up, flux_down, u0, uu = pydisort(
                    tau_arr=self.t_od[:, wl_idx], 
                    omega_arr=self.w[:, wl_idx], 
                    NQuad=self.n_streams,
                    Leg_coeffs_all=self.legendre_moments[:, :, wl_idx].T, 
                    mu0=self.mu0, 
                    I0=self.irradiance[wl_idx], 
                    phi0=self.phi0, 
                    NLeg=self.n_expansion,
                    NFourier=self.n_fourier,
                    # b_pos = 0, # dirichlet condition
                    # b_neg = 0, # dirichlet condition
                    only_flux = self.only_fourier_m0,
                    f_arr=self.truncation_factor[:, wl_idx],
                    NT_cor=self.IMS_TMS_correction,
                    # BRDF_Fourier_modes=[], 
                    # s_poly_coeffs=array([], shape=(1, 0), dtype=float64),
                    # use_banded_solver_NLayers=10, 
                    # autograd_compatible=False
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
                        interp_u(uu)(self.output_polar_angles, # interpolate at user angles
                                     0, # TOA 
                                     self.phi0 + self.relative_azimuths_rad) # phi = phi0 + delta_phi
                        )
                    self.directional_reflectance_toa[:, :,  wl_idx] = (
                        self.directional_radiance_toa[:, :,  wl_idx] * np.pi / np.sum(flux_down(0))
                        )
                        
        self.cos_angle = mu
        