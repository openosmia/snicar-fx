"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

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
