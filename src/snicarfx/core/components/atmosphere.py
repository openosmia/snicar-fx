"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import pandas as pd
import xarray as xr


class AtmosphereColumn:
    """
    Properties of the atmosphere column.

    This class computes and stores the properties of an atmosphere column for each
    layer based on the YAML input file.

    Attributes
    ----------
    nbr_lyr : int
        Number of layers in the column.
    nbr_wvl : int
        Number of wavelengths in the spectral grid.
    ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each layer [unitless].
    asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each layer [unitless].
    tau : ndarray
        Wavelength-dependent optical thickness of each layer [unitless].
    n_expansion : int
        Order of the Legendre expansion of the phase function.
    rayleigh_legendre_moments: ndarray
        Moments of the Legendre expansion of the Rayleigh phase function.

    """

    def __init__(self, config, PACKAGE_ROOT):

        self.PACKAGE_ROOT = PACKAGE_ROOT
        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.surface_elevation = config.LAND.ALTITUDE
        self.wavelengths = np.arange(
            config.SOLVER.WVL_START, config.SOLVER.WVL_END, config.SOLVER.RESOLUTION
        )
        self.nbr_wvl = len(self.wavelengths)
        self.use_atmosphere = config.SOLVER.ATMOSPHERE_COUPLING
        

        if self.use_atmosphere: 
            
            self.atmosphere_profile_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE
            
            # load atm profile with gas conc., P/T/density etc
            self.atmosphere_profile = self.set_atmospheric_profile()
    
            # set nb of atm layers (dep on altitude)
            self.nbr_lyr = self.atmosphere_profile.shape[0]
    
            # init the ssps
            self.ss_alb = np.zeros((self.nbr_lyr, self.nbr_wvl))
            self.tau = np.zeros((self.nbr_lyr, self.nbr_wvl))
            self.rayleigh_legendre_moments = np.zeros(
                (self.n_expansion, self.nbr_lyr, self.nbr_wvl)
            )
            self.tau_molecular_scatter = np.zeros((self.nbr_lyr, self.nbr_wvl))
        
            # compute rayleigh scattering (tau + legendre moments)
            self.compute_rayleigh_scattering()
    
            self.compute_gas_optical_thickness()
    
            self.set_atmospheric_properties_wout_aerosols()

    def set_atmospheric_profile(self):
        profile = pd.read_csv(
            f"{self.PACKAGE_ROOT}/data/atmospheric_profiles/"
            + self.atmosphere_profile_type
            + ".dat",
            skiprows=1,
            sep=r"\s+",
            comment="#",
            header=None,
        )

        profile.columns = [
            "z(km)",
            "p(mb)",
            "T(K)",
            "air(cm-3)",
            "o3(cm-3)",
            "o2(cm-3)",
            "h2o(cm-3)",
            "co2(cm-3)",
            "no2(cm-3)",
        ]

        # calculate layer thicknesses
        profile["dz(km)"] = [
            profile["z(km)"].iloc[-1 + i] - profile["z(km)"].iloc[i]
            for i in range(profile.shape[0])
        ]
        # remove upper level (no layer)
        profile = profile.iloc[1:, :]
        profile.index = np.arange(0, profile.shape[0])

        # truncate dep. on altitude
        profile = profile[profile["z(km)"] >= self.surface_elevation]

        return profile

    def compute_rayleigh_cross_section_bodhaine(self, co2_ppm):
        """
        Compute Rayleigh scattering cross-section as in Bodhaine et al. 1999,
        (default in LibRadtran).

        Parameters:
        - co2_ppm: float, CO2 concentration in ppm

        Returns:
        - crs: numpy array of Rayleigh scattering cross-sections (cm^2)
        """

        # Convert wavelength array
        lambda_cm = self.wavelengths * 1e-7
        lambda_um = self.wavelengths * 1e-3

        # Number density of air at standard conditions (mol/cm3)
        N_s = 2.546899e19

        ray_const = 24 * np.pi**3 / N_s**2

        # Convert CO2 mixing ratio from ppm to parts per volume by percent
        co2_vp = co2_ppm * 1.0e-4

        # (n_air - 1) at 300 ppm CO2 (Eq. 18 in Bodhaine et al. 1999)
        n_300 = (
            8060.51
            + 2480990 / (132.274 - lambda_um**-2)
            + 17455.7 / (39.32957 - lambda_um**-2)
        ) * 1e-8

        # n_air at given CO2 concentration (Eq. 19 in Bodhaine et al. 1999)
        n_air = (1 + 0.54 * (co2_ppm * 1e-6 - 0.0003)) * n_300 + 1

        ref_ratio = ((n_air**2 - 1) ** 2) / ((n_air**2 + 2) ** 2)

        # Depolarization factor of N2 (Eq. 5 in Bodhaine et al. 1999)
        F_N2 = 1.034 + 3.17e-4 / lambda_um**2
        # Depolarization factor of O2 (Eq. 6 in Bodhaine et al. 1999)
        F_O2 = 1.096 + 1.385e-3 / lambda_um**2 + 1.448e-4 / lambda_um**4
        # Depolarization factor of dry air (Eq. 23 in Bodhaine et al. 1999)
        F_air = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 + co2_vp * 1.15) / (
            78.084 + 20.946 + 0.934 + co2_vp
        )

        # Rayleigh scatt. cross-section (cm2, Eq. 22 in Bodhaine et al. 1999)
        # note that F_air can be calculated as (6+3*rho)/(6-7*rho) if the
        # depol. ratio (rho) is known/prescribed
        crs = (ray_const / lambda_cm**4) * ref_ratio * F_air

        return crs

    def compute_rayleigh_scattering(self):
        """
        Compute wavelength-dependent optical thickness and rayleigh scattering
        phase function of air molecules for each layer.
        """

        for lyr in range(self.nbr_lyr):

            # convert co2 number density to ppm

            co2_ppm = (
                self.atmosphere_profile["co2(cm-3)"][lyr]
                / self.atmosphere_profile["air(cm-3)"][lyr]
            ) * 1e6

            rayleigh_cross_section = self.compute_rayleigh_cross_section_bodhaine(
                co2_ppm
            )

            # molecular cross section is in cm2 / mol
            # molecular nb density from mol/cm3 to mol/m3
            # layer depth from km to m)

            self.tau_molecular_scatter[lyr, :] = (
                rayleigh_cross_section
                * 1e-4
                * self.atmosphere_profile["air(cm-3)"][lyr]
                * 1e6
                * self.atmosphere_profile["dz(km)"][lyr]
                * 1e3
            )

        # phase coeffs of order > 3 are null
        self.rayleigh_legendre_moments[:3, :, :] = np.array([1, 0, 0.5])[:, None, None]

        return None

    def load_gas_absorptions(self):
        # get absorption in (c)m2 / molecule for each gas
        return None

    def compute_gas_optical_thickness(self):
        """
        Compute wavelength-dependent optical thickness of atmospheric gases
        for each layer based on their concentrations.
        """

        # for lyr in range(self.nbr_lyr):

        #     # sum absorption * conc for all gas for given layer
        #     absorption = np.sum(gas_absorptions * gas_concentrations, axis=1)

        #     # get gaseous optical thickness
        #     self.tau_gases[lyr, :] = absorption * self.atmosphere_profile.thickness

        hapi2libis_data = xr.open_dataset(
            f"{self.PACKAGE_ROOT}/data/atmospheric_profiles/uvspec_afglss_test_file.nc"
        )
        hapi2libis_data['nwvl'] = hapi2libis_data.wvl
        
        self.tau_gases = hapi2libis_data["tau"].interp(nwvl=self.wavelengths).values
        
        # if z = 1, remove layer 0 ie index at 1
        # if z = 2, remove layer 0+1 ie index at 2, etc
        
        self.tau_gases = self.tau_gases[int(self.surface_elevation):, :]
        
        # self.tau_gases[self.tau_gases < 2e-4] = 2e-4

        return None

    def set_atmospheric_properties_wout_aerosols(self):

        self.tau = self.tau_molecular_scatter + self.tau_gases
        self.ss_alb = self.tau_molecular_scatter / (
            self.tau_gases + self.tau_molecular_scatter
        )
        self.legendre_moments = self.rayleigh_legendre_moments

        return None
