"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np

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

    def __init__(self, config):

        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.surface_elevation = config.LAND.ALTITUDE
        self.wavelengths = np.arange(config.SOLVER.WVL_START, 
                                     config.SOLVER.WVL_END, 
                                     config.SOLVER.RESOLUTION)
        self.nbr_wvl = len(self.wavelengths)
        
        # 1 - load atm profile first
        
        # 2 - set nb of atm layers second depending on altitude
        self.nbr_lyr = self.set_nb_atmospheric_layers()
        
        # 3 - init the ssps third
        self.ss_alb = np.zeros((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.zeros((self.nbr_lyr, self.nbr_wvl))
        self.rayleigh_legendre_moments = np.zeros((self.n_expansion, self.nbr_lyr, self.nbr_wvl))


    def load_atmospheric_profile(self):
        # get concentration of each gas + air pressure/density/temperature for
        # each layer
        return None
    
    def set_nb_atmospheric_layers(self):
        # get concentration of each gas + air pressure/density/temperature for
        # each layer
        return 50

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
            self.tau_gases[lyr, :] = absorption * dZ 

        return None

    def compute_rayleigh_scattering(self):
        """
        Compute wavelength-dependent optical thickness and rayleigh scattering
        phase function of air molecules for each layer.
        """

        for lyr in range(self.nbr_lyr):

            # (cm2 / mol * mol / cm3 * cm)

            self.tau_molecules[lyr, :] = f(_lambda) * N_air * dZ

            # phase coeffs of order > 3 are null
            self.rayleigh_legendre_moments[:3, lyr, :] = np.array([1, 0, 1/10])[:, None, None]

        return None

    def set_atmospheric_properties_wout_aerosols(self):

        self.tau_atm = self.tau_molecules + self.tau_gases
        self.ss_alb_atm = self.tau_molecules / (self.tau_gases + self.tau_molecules)
        self.legendre_moments_atm = self.rayleigh_legendre_moments
        
        return None
