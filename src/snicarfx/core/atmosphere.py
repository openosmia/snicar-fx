"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


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
        self.layer_type = model_inputs.inputs["ICE"]["LAYER_TYPE"]
        self.nbr_lyr = len(self.layer_type)
        self.thickness_profile = model_inputs.inputs["ICE"]["THICKNESS"]
        self.density = model_inputs.inputs["ICE"]["DENSITY"]
        self.rf_type = model_inputs.inputs["ICE"]["RF_TYPE"]
        self.grain_shape = model_inputs.inputs["ICE"]["GRAIN_SHAPE"]
        self.lwc = model_inputs.inputs["ICE"]["LWC"]
        self.ssa = model_inputs.inputs["ICE"]["SPECIFIC_SURFACE_AREA"]

        self.wavelengths = (
            np.arange(
                self.model_inputs.inputs["RTM"]["WVL_START"],
                self.model_inputs.inputs["RTM"]["WVL_END"],
                self.model_inputs.inputs["RTM"]["RESOLUTION"],
            )
            * 1e-9
        )

        self.nbr_wvl = len(self.wavelengths)
        self.sfc = np.ones(self.nbr_wvl) * model_inputs.inputs["ICE"]["SFC"]

        # init the ssps
        self.ext_cff = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.ss_alb = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.asm_prm = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.tau = np.ones((self.nbr_lyr, self.nbr_wvl))
        self.layer_mass = np.zeros(self.nbr_lyr)

        self.set_refractive_index_and_diffuse_fresnel_coeffs()
        self.set_column_ops_without_laps()

        if self.model_inputs.inputs["LIGHT_ABSORBING_PARTICLES"] is not None:
            self.set_lap_properties()
            self.update_column_ops_with_laps()

    def load_atmospheric_profile(self):
        return None

    def load_gas_absorptions(self):
        return None

    def compute_gas_absorptions(self):
        """
        Compute optical properties of a clean snow/ice column (no aerosols).

        This method calculates wavelength-dependent extinction coefficients,
        single scattering albedo, asymmetry parameters, and optical thickness
        for each layer based on the input physical parameters and refractive
        indices. Different models are used depending on whether layers are made
        of snow grains in air or ice with air inclusions, but all use geometric
        optics approximation (grain/bubble larger than the wavelength).
        """
        self.layer_mass = np.array(self.density) * np.array(self.thickness_profile)

        for lyr in range(self.nbr_lyr):

            # sum sigma * Ngas * dZ
            self.ext_cff_gases[lyr, :] = abs_cff

            self.tau_gases[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]

        return None

    def compute_rayleigh_scattering(self):
        """
        Compute optical properties of a clean snow/ice column (no aerosols).

        This method calculates wavelength-dependent extinction coefficients,
        single scattering albedo, asymmetry parameters, and optical thickness
        for each layer based on the input physical parameters and refractive
        indices. Different models are used depending on whether layers are made
        of snow grains in air or ice with air inclusions, but all use geometric
        optics approximation (grain/bubble larger than the wavelength).
        """
        self.layer_mass = np.array(self.density) * np.array(self.thickness_profile)

        for lyr in range(self.nbr_lyr):

            # sigma * Nr * dZ
            self.ext_cff_molecules[lyr, :] = scattering_cff

            self.tau_molecules[lyr, :] = self.layer_mass[lyr] * self.ext_cff[lyr, :]

            self.phase_function = None

        return None

    def set_atmospheric_properties(self):

        self.tau = self.tau_molecules + self.tau_gases
        self.ss_alb = self.tau_molecules / (self.tau_gases + self.tau_molecules)

        return None
