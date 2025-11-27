"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import math
import numpy as np
import pandas as pd


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

        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.surface_elevation = config.LAND.ALTITUDE
        self.wavelengths = np.arange(
            config.SOLVER.WVL_START, config.SOLVER.WVL_END, config.SOLVER.RESOLUTION
        )
        self.nbr_wvl = len(self.wavelengths)
        self.use_atmosphere = config.SOLVER.ATMOSPHERE_COUPLING
        self.atmosphere_profile_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

        # 1 - load atm profile with gas conc., P/T/density first
        # self.atmosphere_profile = self.set_atmospheric_profile()

        # # 2 - set nb of atm layers second depending on altitude
        # self.nbr_lyr = self.set_nb_atmospheric_layers()

        # # 3 - init the ssps third
        # self.ss_alb = np.zeros((self.nbr_lyr, self.nbr_wvl))
        # self.tau = np.zeros((self.nbr_lyr, self.nbr_wvl))
        # self.rayleigh_legendre_moments = np.zeros(
        #     (self.n_expansion, self.nbr_lyr, self.nbr_wvl)
        # )

    def compute_rayleigh_cross_section_bodhaine(self, mixing_ratio_co2):
        """
        Compute Rayleigh scattering cross-section using the wavelength array
        stored in the class, following Bodhaine et al. (1999).

        Parameters:
        - mixing_ratio_co2: float, CO2 mixing ratio in ppmv

        Returns:
        - crs: numpy array of Rayleigh scattering cross-sections (cm^2)
        """

        # Conversion constants
        from_nm_to_cm = 1.0e-7  # wavelength from nm to cm
        from_nm_to_um = 1.0e-3  # wavelength from nm to µm

        # Number density of air at standard conditions
        N_s = 2.546899e19  # molecules per cm^3

        # Rayleigh scattering constant (Bodhaine et al.)
        ray_const = 24 * np.pi**3 / N_s**2

        # Convert CO2 mixing ratio from ppmv to volume fraction
        co2 = mixing_ratio_co2 * 1.0e-4

        # Convert wavelength array
        lambda_cm = self.wavelengths * from_nm_to_cm
        lambda_um = self.wavelengths * from_nm_to_um

        # Refractive index of air at 300 ppm CO2 (Bodhaine et al., Eq. 18)
        n_300 = (
            8060.51
            + 2480990 / (132.274 - lambda_um**-2)
            + 17455.7 / (39.32957 - lambda_um**-2)
        ) * 1e-8

        # Adjust refractive index for actual CO2 concentration (Eq. 19)
        n = (1 + 0.54 * (mixing_ratio_co2 * 1e-6 - 0.0003)) * n_300 + 1

        # Clausius-Mossotti factor (ref_ratio)
        ref_ratio = ((n**2 - 1) ** 2) / ((n**2 + 2) ** 2)

        # King factors for N2 and O2 (Eq. 5 & 6)
        F_N2 = 1.034 + 3.17e-4 / lambda_um**2
        F_O2 = 1.096 + 1.385e-3 / lambda_um**2 + 1.448e-4 / lambda_um**4

        # Effective King factor for dry air (Eq. 23)
        F_air = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 + co2 * 1.15) / (
            78.084 + 20.946 + 0.934 + co2
        )

        # Depolarization ratio (used in Rayleigh cross-section)
        depol = 6 * (F_air - 1) / (3 + 7 * F_air)

        # Rayleigh scattering cross-section (cm^2)
        crs = (ray_const / lambda_cm**4) * ref_ratio * F_air

        return crs

    def set_atmospheric_profile(self):
        profile = pd.read_csv(self.atmosphere_profile_type)
        # !!!!!! truncates depending on altitude !!!!!!
        return profile

    def set_nb_atmospheric_layers(self):
        return len(self.atmosphere_profile.P)  # select one column of the profile

    def load_gas_absorptions(self):
        # get absorption in (c)m2 / molecule for each gas
        return None

    def compute_gas_optical_thickness(self):
        """
        Compute wavelength-dependent optical thickness of atmospheric gases
        for each layer based on their concentrations.
        """

        for lyr in range(self.nbr_lyr):

            # sum absorption * conc for all gas for given layer
            absorption = np.sum(gas_absorptions * gas_concentrations, axis=1)

            # get gaseous optical thickness
            self.tau_gases[lyr, :] = absorption * self.atmosphere_profile.thickness

        return None

    def compute_rayleigh_scattering(self):
        """
        Compute wavelength-dependent optical thickness and rayleigh scattering
        phase function of air molecules for each layer.
        """

        for lyr in range(self.nbr_lyr):

            # equation from Bodhaine et al. (1999) as in Libradtran
            # (cm2 / mol * mol / cm3 * cm) = no units
            # Nair is moleculair nb density (mol / cm3)
            # atmosphere_profile.thickness is layer depth (cm)
            # molecular cross section is in cm / mol

            kb = 1.380649 * 1e-23  # boltzman constant J/K
            N = (
                self.atmosphere_profile.pressure
                * kb
                / self.atmosphere_profile.temperature
            )  # mol cm-3
            n_air = None
            king_factor = None

            rayleigh_cross_section = (
                (24 * (np.pi**3))
                / ((N**2) * (self.wavelengths**4))
                * (n_air**2 - 1) ** 2
                / (n_air**2 + 2) ** 2
                * king_factor
            )

            self.tau_molecular_scatter[lyr, :] = (
                rayleigh_cross_section
                * self.atmosphere_profile.Nair
                * self.atmosphere_profile.thickness
            )

        # phase coeffs of order > 3 are null
        self.rayleigh_legendre_moments[:3, lyr, :] = np.array([1, 0, 1 / 10])[
            :, None, None
        ]

        return None

    def set_atmospheric_properties_wout_aerosols(self):

        self.tau = self.tau_molecular_scatter + self.tau_gases
        self.ss_alb = self.tau_molecular_scatter / (
            self.tau_gases + self.tau_molecular_scatter
        )
        self.legendre_moments = self.rayleigh_legendre_moments

        return None
