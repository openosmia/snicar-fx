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
    model_inputs : ModelInputs
        An instance of the ModelInputs class containing model input data parsed
        from the YAML configuration file.
    layer_type : list
        Type of layer (0 for snow grains in air, 1 for air bubbles in ice).
    nbr_lyr : int
        Number of layers in the column.
    thickness_profile : list
        Thicknesses [m] of each layer of the snow or ice column.
    density : list
        Density of snow/ice for each layer [kg/m3].
    rf_type : str
        Source of refractive index data.
    grain_shape : list
        Identifier for grain shape model for each layer.
    lwc : list
        Liquid water content fraction for each layer [0-1].
    ssa : list
        Specific surface area of snow or ice in each layer [m2/kg].
    sfc : float
        Reflectance of underlying surface (wavelength independent).
    wavelengths : list
        Spectral grid used for all optical property calculations [m].
    nbr_wvl : int
        Number of wavelengths in the spectral grid.
    lap_ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each LAP [unitless].
    lap_asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each LAP [unitless].
    lap_ext_cff : ndarray
        Wavelength-dependent mass extinction coefficient of each LAP [m2/kg].
    lap_concentrations : ndarray
        Mass concentrations of LAPs per layer [kg/kg].
    ext_cff : ndarray
        Wavelength-dependent mass extinction coefficient of each layer [m2/kg].
    ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each layer [unitless].
    asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each layer [unitless].
    tau : ndarray
        Wavelength-dependent optical thickness of each layer [unitless].
    layer_mass : ndarray
        Mass per unit area of each layer [kg/m2].

    """

    def __init__(self, model_inputs):
        self.model_inputs = model_inputs


        # init the ssps
        self.phase_coefficients = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr, self.nbr_wvl))


    def load_atmospheric_profile(self):
        # get concentration of each gas + air pressure/density/temperature for
        # each layer
        return None

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

            self.rayleigh_phase_coefficients = None

        return None

    def set_atmospheric_properties_wout_aerosols(self):

        self.tau = self.tau_molecules + self.tau_gases
        self.ss_alb = self.tau_molecules / (self.tau_gases + self.tau_molecules)
        self.phase_coefficients = self.rayleigh_phase_coefficients
        
        return None
