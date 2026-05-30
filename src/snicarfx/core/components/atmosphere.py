"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import pandas as pd
import xarray as xr


class AtmosphereColumn:
    """
    Compute and store the physical and optical properties of an atmospheric
    column based on the YAML input file.

    Attributes
    ----------
    ROOT_PATH : str
        Path to the snicarfx module.
    use_atmosphere : bool
        Boolean encoding for land-atmosphere coupling.
    _wavelengths : ndarray
        Wavelength array (m).
    nbr_wvl : int
        Number of wavelengths in the spectral array.
    surface_elevation : int
        Altitude of the surface (km).
    n_expansion : int
        Order of expansion of the phase function.
    atmosphere_profile_type : str
        AFGL tag for type of atmospheric profile.
    atmosphere_profile : pandas DataFrame
        Atmospheric profile (altitudes, layer thicknesses, gas concentrations,
        pressure, temperature).
    nbr_lyr : int
        Number of layers in the column.
    aerosol_boundary_height : int
        Maximum altitude with aerosols (fixed to 30km).
    AOD : float
        Column-integrated aerosol optical thickness.
    integrated_gas_concentrations : dict
        Column-integrated concentrations of atmospheric gases.
    tau_molecular_scatter : ndarray
        Rayleigh optical thickness.
    rayleigh_legendre_moments: ndarray
        Moments of the Legendre expansion of the Rayleigh phase function.
    gas_cross_sections : ndarray
        Spectral absorption cross section of atmospheric gases.
    tau_gases : ndarray
        Spectral optical thickness of atmospheric gases.
    aerosol_ss_alb : ndarray
        Spectral single scattering albedo of atmospheric aerosols.
    aerosol_ext_cff : ndarray
        Spectral extinction coefficient of atmospheric aerosols.
    aerosol_ext_cff_550 : float
        Extinction coefficient of atmospheric aerosols at 550nm.
    aerosol_legendre_moments : ndarray
        Moments of the Legendre expansion of the aerosol phase function.
    tau_aerosols : ndarray
        Spectral optical thickness of atmospheric aerosols.
    ss_alb : ndarray
        Wavelength-dependent single scattering albedo of each layer [unitless].
    asm_prm : ndarray
        Wavelength-dependent asymmetry parameter of each layer [unitless].
    tau : ndarray
        Wavelength-dependent optical thickness of each layer [unitless].
    legendre_moments: ndarray
        Moments of the Legendre expansion of the atmosphere phase function.
    """

    def __init__(self, config):

        self.ROOT_PATH = config._ROOT_PATH
        self.use_atmosphere = config.SOLVER.ATMOSPHERE_COUPLING

        if self.use_atmosphere:

            self._wavelengths = config._wavelengths_atmosphere
            self.nbr_wvl = len(self._wavelengths)
            self.surface_elevation = config.LAND.ALTITUDE
            self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
            self.atmosphere_profile_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE
            self.initial_atmosphere_profile = self.set_atmospheric_profile()
            self.set_profile_integrated_gas_concentrations()
            # as no scaling is applied for now, the atmospheric
            # profile is just the initial profile
            self.atmosphere_profile = self.initial_atmosphere_profile.copy(deep=True)

            self.nbr_lyr = self.atmosphere_profile.shape[0]
            self.aerosol_boundary_height = 30
            self.AOD = config.ATMOSPHERE.INTEGRATED_AOD_550

            if config.ATMOSPHERE.INTEGRATED_GAS_CONCENTRATIONS is not None:
                self.integrated_gas_concentrations = (
                    config.ATMOSPHERE.INTEGRATED_GAS_CONCENTRATIONS.model_dump()
                )
                self.scale_atmospheric_profile()

            self.compute_rayleigh_scattering()
            self.set_rayleigh_legendre_moments()

            self.load_gas_absorption_cross_sections()
            self.compute_gas_optical_thickness()

            if self.AOD == 0.0 or self.surface_elevation > self.aerosol_boundary_height:
                self.set_atmospheric_properties_without_aerosols()

            else:
                self.aerosol_file = config.ATMOSPHERE.AEROSOL_PROPERTIES
                self.set_aerosol_properties()
                self.scale_tau_aerosols()
                self.set_atmospheric_properties_with_aerosols()

            self.set_legendre_moments()

    def set_atmospheric_profile(self):
        profile = pd.read_csv(
            f"{self.ROOT_PATH}/data/atmospheric_profiles/"
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

        # truncate dep. on altitude
        profile = profile[profile["z(km)"] >= self.surface_elevation]

        # calculate layer thicknesses

        dz = np.abs(np.diff(profile["z(km)"].values))

        # # TODO: ransform profile into layer variables (mid-point)
        profile = profile.rolling(2).mean().iloc[1:, :].reset_index()

        # add layer thicknesses
        profile["dz(km)"] = dz

        return profile

    def set_profile_integrated_gas_concentrations(self):
        """
        Compute integrated gas concentrations based on the
        atmospheric profile.
        """

        AVOGADRO_NUMBER = 6.02214076e23

        # molecular masses of gases (kg/mol)
        MOLECULAR_MASSES = {
            "O3": 0.048,
            "O2": 0.032,
            "H2O": 0.018015,
            "CO2": 0.04401,
            "NO2": 0.04601,
        }

        # create a key matching dict between AFGL keys (gas in lower
        # case + "(cm-3)") and input file keys (gas in upper case)
        profile_key_matching = {
            g.split("(")[0].upper(): g for g in self.initial_atmosphere_profile.columns
        }
        self.gas_key_matching = {
            k: v
            for k, v in profile_key_matching.items()
            if k in MOLECULAR_MASSES.keys()
        }

        # Compute integrated gas concentrations (kg/m²) for all gases
        # of the atmospheric profile
        self.profile_integrated_gas_concentrations = {
            input_key: np.sum(
                self.initial_atmosphere_profile[profile_key].values
                * 1e6  # convert to molecules/m3
                * self.initial_atmosphere_profile["dz(km)"].values
                * 1e3  # convert to m
            )
            * (MOLECULAR_MASSES[input_key] / AVOGADRO_NUMBER)
            for input_key, profile_key in self.gas_key_matching.items()
        }

    def scale_atmospheric_profile(self):
        """
        Scale atmospheric profile by given integrated gas
        concentrations.

        """

        for input_key, gas_igc in self.integrated_gas_concentrations.items():

            # Filter out gases with None concentration
            if gas_igc is not None:
                profile_key = self.gas_key_matching[input_key]

                # compute scale factor based on input
                scale_factor = (
                    gas_igc / self.profile_integrated_gas_concentrations[input_key]
                )

                # Apply scaling (back to cm⁻³)
                self.atmosphere_profile[profile_key] = (
                    self.initial_atmosphere_profile[profile_key] * scale_factor
                )

    def compute_rayleigh_cross_section_bodhaine(self, co2_ppm):
        """
        Compute Rayleigh scattering cross-section as in Bodhaine et al. 1999,
        (default in LibRadtran).

        Parameters:
        - co2_ppm: float, CO2 concentration in ppm

        Returns:
        - crs: numpy array of Rayleigh scattering cross-sections (cm2)
        """

        # Convert wavelength array
        lambda_cm = self._wavelengths * 1e-7
        lambda_um = self._wavelengths * 1e-3

        co2_ppm = co2_ppm[:, None]

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
        Compute rayleigh spectral optical thickness.
        """

        # convert co2 number density to ppm
        co2_ppm = (
            self.atmosphere_profile["co2(cm-3)"].values
            / self.atmosphere_profile["air(cm-3)"].values
        ) * 1e6

        rayleigh_cross_section = self.compute_rayleigh_cross_section_bodhaine(co2_ppm)

        # molecular cross section is in cm2 / mol
        # molecular nb density from mol/cm3 to mol/m3
        # layer depth from km to m)
        self.tau_molecular_scatter = (
            rayleigh_cross_section
            * 1e-4
            * self.atmosphere_profile["air(cm-3)"].values[:, None]
            * 1e6
            * self.atmosphere_profile["dz(km)"].values[:, None]
            * 1e3
        )

        return None

    def set_rayleigh_legendre_moments(self):
        """
        Set coefficients for Legendre expansion of Rayleigh phase function.
        """

        self.rayleigh_legendre_moments = np.zeros(
            (self.n_expansion + 2, self.nbr_lyr, self.nbr_wvl)
        )

        # phase coeffs of order > 3 are null (already init at 0)
        self.rayleigh_legendre_moments[:3, :, :] = np.array([1, 0, 0.1])[:, None, None]

    def load_gas_absorption_cross_sections(self):
        """
        Load spectral absorption cross sections of atmospheric gases.
        """

        # get absorption in (c)m2 / molecule for each gas
        file_name = f"{self.ROOT_PATH}/data/gases/uvspec_{self.atmosphere_profile_type}_cross_sections.nc"
        try:
            self.gas_cross_sections = xr.open_dataset(file_name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Gas absorption cross section file not found ({file_name}). Please run `snicarfx-download-data` in a terminal (or manually download file from https://zenodo.org/records/20457918)."
            )

        # truncate depending on altitude
        self.gas_cross_sections = self.gas_cross_sections.sel(
            nlyr=self.atmosphere_profile.index
        )

        # interpolate on wvl
        self.gas_cross_sections["nwvl"] = self.gas_cross_sections.wvl

        self.gas_cross_sections = self.gas_cross_sections.interp(
            nwvl=self._wavelengths, kwargs={"fill_value": "extrapolate"}
        )

        return None

    def compute_gas_optical_thickness(self):
        """
        Compute spectral optical thickness of atmospheric gases.
        """

        sigma_vars = [
            var
            for var in self.gas_cross_sections.data_vars
            if var.startswith("sigma_") and "O2-O2" not in var
        ]

        total_absorption = np.zeros_like(self.gas_cross_sections[sigma_vars[0]].values)

        for sigma_var in sigma_vars:

            # match profile gas tag
            sigma_var_lower = sigma_var.split("_")[1].lower()
            profile_tag = f"{sigma_var_lower}(cm-3)"

            total_absorption += (
                self.gas_cross_sections[sigma_var].values
                * self.atmosphere_profile[profile_tag].values[:, None]
            )

        # as per hapi2libis
        total_absorption += (
            1e-46 * self.atmosphere_profile["o2(cm-3)"].values[:, None] ** 2
        ) * self.gas_cross_sections["sigma_O2-O2"].values

        self.tau_gases = (
            total_absorption * self.atmosphere_profile["dz(km)"].values[:, None] * 1e5
        )

        return None

    def load_aerosol_properties(self):
        """
        Load optical properties of aerosols.
        """

        aerosol_properties = xr.open_dataset(
            f"{self.ROOT_PATH}/data/aerosols/{self.aerosol_file}"
        ).interp(wavelength=self._wavelengths, kwargs={"fill_value": "extrapolate"})

        return aerosol_properties

    def set_aerosol_properties(self):
        """
        Set optical properties of aerosols.
        """

        aerosol_properties = self.load_aerosol_properties()

        self.aerosol_ss_alb = aerosol_properties["single_scattering_albedo"].values
        self.aerosol_ext_cff = aerosol_properties["extinction_coefficient"].values
        self.aerosol_ext_cff_550 = (
            aerosol_properties["extinction_coefficient"].interp(wavelength=550).values
        )
        self.aerosol_legendre_moments = aerosol_properties["legendre_moments"].values.T[
            : self.n_expansion + 2, :
        ]

    def scale_tau_aerosols(self):
        """
        Scale aerosol optical thickness by the user-input column-integrated
        aerosol thickness at 550nm.
        """

        profile_aerosol_z = self.atmosphere_profile["z(km)"].values.copy()
        profile_aerosol_dz = self.atmosphere_profile["dz(km)"].values.copy()

        # set dz to 0 outside of the aerosol layer (propagating to tau=0)
        profile_aerosol_dz[profile_aerosol_z > self.aerosol_boundary_height] = 0.0

        self.tau_aerosols = (
            self.aerosol_ext_cff[None, :]
            * profile_aerosol_dz[:, None]
            * self.AOD
            / self.aerosol_ext_cff_550
            / np.nansum(profile_aerosol_dz)
        )

    def prevent_pure_scattering(self):
        """
        Prevent single scattering albedo to be exactly 1 which creates
        numerical instabilities.
        """

        self.ss_alb[self.ss_alb == 1] = 1.0 - 1e-7

    def set_atmospheric_properties_without_aerosols(self):
        """
        Set atmospheric optical properties in the absence of aerosols.
        """

        self.tau = self.tau_molecular_scatter + self.tau_gases
        self.ss_alb = self.tau_molecular_scatter / (self.tau)
        self.prevent_pure_scattering()

        return None

    def set_atmospheric_properties_with_aerosols(self):
        """
        Set atmospheric optical properties in the presence of aerosols.
        """

        self.tau = self.tau_molecular_scatter + self.tau_gases + self.tau_aerosols

        self.ss_alb = (
            self.tau_molecular_scatter + self.aerosol_ss_alb * self.tau_aerosols
        ) / (self.tau)

        self.prevent_pure_scattering()

        return None

    def set_legendre_moments(self):
        """
        Set legendre expansion coefficients of the atmospheric phase function.
        """

        if self.AOD > 0:
            aerosol_tau_ss_alb = (
                self.tau_aerosols[None, :, :] * self.aerosol_ss_alb[None, None, :]
            )
            self.legendre_moments = (
                (self.aerosol_legendre_moments[:, None, :] * aerosol_tau_ss_alb)
                + (
                    self.tau_molecular_scatter[None, :, :]
                    * self.rayleigh_legendre_moments
                )
            ) / (self.tau_molecular_scatter[None, :, :] + aerosol_tau_ss_alb)

        else:
            self.legendre_moments = self.rayleigh_legendre_moments
