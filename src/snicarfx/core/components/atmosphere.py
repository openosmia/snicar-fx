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

    def __init__(self, config):

        self.ROOT_PATH = config._ROOT_PATH
        self.wavelengths = config._wavelengths_atmosphere

        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.surface_elevation = config.LAND.ALTITUDE

        self.nbr_wvl = len(self.wavelengths)
        self.use_atmosphere = config.SOLVER.ATMOSPHERE_COUPLING

        self.integrated_gas_concentrations = (
            config.ATMOSPHERE.INTEGRATED_GAS_CONCENTRATIONS
        )
        self.AOD = config.ATMOSPHERE.INTEGRATED_AOD_550
        self.aerosol_file = config.ATMOSPHERE.AEROSOL_PROPERTIES

        if self.use_atmosphere:

            self.atmosphere_profile_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

            # load atm profile with gas conc., P/T/density etc
            self.atmosphere_profile = self.set_atmospheric_profile()

            # scale atmospheric profile by integrated gas concentrations if passed
            if self.integrated_gas_concentrations is not None:
                self.scale_atmospheric_profile()

            # set nb of atm layers (dependent on altitude)
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

            # load gas cross sections
            self.load_gas_absorption_cross_sections(config)
            self.compute_gas_optical_thickness()

            if self.AOD == 0.0:
                self.set_atmospheric_properties_without_aerosols()

            else:
                # set aerosol properties
                self.set_aerosol_properties()

                # scale aerosol with AOD
                self.scale_tau_aerosols()

                self.set_atmospheric_properties_with_aerosols()

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

    def scale_atmospheric_profile(self):
        """
        Scale atmospheric profile by given integrated gas concentrations.
        """

        # Avogadro number
        AVOGADRO_NUMBER = 6.02214076e23

        # molecular masses of gases of interest (kg/mol)
        MOLECULAR_MASSES = {
            "O3": 0.048,  # Ozone
            "O2": 0.032,  # Oxygen
            "H2O": 0.018015,  # Water vapor
            "CO2": 0.04401,  # Carbon dioxide
            "NO2": 0.04601,  # Nitrogen dioxide
        }

        # keep only passed gases
        gases_to_scale = self.integrated_gas_concentrations.model_dump(
            exclude_none=True
        )

        for gas_name, integrated_gas_column in gases_to_scale.items():

            # match profile column name
            profile_key = f"{gas_name.lower()}(cm-3)"

            # convert profile to molecules/m3
            n = self.atmosphere_profile[profile_key].values * 1e6
            dz = self.atmosphere_profile["dz(km)"].values * 1e3

            # initial column in kg/m²
            current_column = (
                np.sum(n * dz) * MOLECULAR_MASSES[gas_name] / AVOGADRO_NUMBER
            )

            # scale factor
            scale_factor = integrated_gas_column / current_column

            # apply scaling (back to cm⁻³)
            self.atmosphere_profile[profile_key] *= scale_factor

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

        # phase coeffs of order > 3 are null (already init at 0)
        # self.rayleigh_legendre_moments[:3, :, :] = np.array([1, 0, 0.5])[:, None, None]
        self.rayleigh_legendre_moments[:3, :, :] = np.array([1, 0, 0.1])[:, None, None]

        return None

    def load_gas_absorption_cross_sections(self, config):

        # get absorption in (c)m2 / molecule for each gas
        self.gas_cross_sections = xr.open_dataset(
            f"{self.ROOT_PATH}/data/atmospheric_profiles/uvspec_afglss_test_file_cross_sections.nc"
        )
        self.gas_cross_sections["nwvl"] = self.gas_cross_sections.wvl

        # if (
        #     config.SPECTRAL.MODE == "monochromatic"
        #     or config.SPECTRAL.BAND_METHOD == "srf-integration"
        # ):
        self.gas_cross_sections = self.gas_cross_sections.interp(nwvl=self.wavelengths)

        # elif config.SPECTRAL.BAND_METHOD == "snicar-default":
        #     self.gas_cross_sections = compute_band_average(
        #         self.gas_cross_sections, config._band_ranges, wavelength_dim="nwvl"
        #     )

        #     print(self.gas_cross_sections)

        return None

    def compute_gas_optical_thickness(self):
        """
        Compute wavelength-dependent optical thickness of atmospheric gases
        for each layer based on their concentrations.
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

        # if z = 1, remove layer 0 ie index at 1
        # if z = 2, remove layer 0+1 ie index at 2, etc
        self.tau_gases = self.tau_gases[int(self.surface_elevation) :, :]

        return None

    def load_aerosol_properties(self):
        """
        Load optical properties of aerosols
        """

        aerosol_properties = xr.open_dataset(
            f"{self.ROOT_PATH}/data/aerosols/{self.aerosol_file}"
        ).interp(wavelength=self.wavelengths, kwargs={"fill_value": "extrapolate"})

        return aerosol_properties

    def set_aerosol_properties(self):
        """
        Set optical properties of light-absorbing particles (AEROSOLs).

        This method sets the properties of each AEROSOL defined in the input
        configuration, converting their concentrations to consistent units,
        and interpolating their properties to the model's spectral grid.
        """

        aerosol_properties = self.load_aerosol_properties()

        self.aerosol_ss_alb = aerosol_properties["single_scattering_albedo"].values
        self.aerosol_ext_cff = aerosol_properties["extinction_coefficient"].values
        self.aerosol_ext_cff_550 = (
            aerosol_properties["extinction_coefficient"].interp(wavelength=550).values
        )
        self.aerosol_legendre_moments = aerosol_properties["legendre_moments"].values.T[
            : self.n_expansion, :
        ]

    def scale_tau_aerosols(self):
        """
        Scale aerosols by given .
        """

        self.tau_aerosols = np.zeros_like(self.tau_molecular_scatter)

        profile_aerosol_z = self.atmosphere_profile["z(km)"].values
        profile_aerosol_dz = self.atmosphere_profile["dz(km)"].values
        # set dz to 0 outside of the aerosol layer (propagating to tau=0)
        profile_aerosol_dz[profile_aerosol_z > 3] = 0.0

        self.tau_aerosols = (
            self.aerosol_ext_cff[None, :]
            * profile_aerosol_dz[:, None]
            * self.AOD
            / self.aerosol_ext_cff_550
            / np.nansum(profile_aerosol_dz)
        )

    def set_atmospheric_properties_without_aerosols(self):

        self.tau = self.tau_molecular_scatter + self.tau_gases
        self.ss_alb = self.tau_molecular_scatter / (
            self.tau_gases + self.tau_molecular_scatter
        )
        self.legendre_moments = self.rayleigh_legendre_moments

        return None

    def set_atmospheric_properties_with_aerosols(self):

        self.tau = self.tau_molecular_scatter + self.tau_gases + self.tau_aerosols

        self.ss_alb = (
            self.tau_molecular_scatter + self.aerosol_ss_alb * self.tau_aerosols
        ) / (self.tau_gases + self.tau_molecular_scatter + self.tau_aerosols)

        self.legendre_moments = (
            (
                self.aerosol_legendre_moments[:, None, :]
                * self.tau_aerosols[None, :, :]
                * self.aerosol_ss_alb[None, None, :]
            )
            + (self.tau_molecular_scatter[None, :, :] * self.rayleigh_legendre_moments)
        ) / (
            self.tau_molecular_scatter[None, :, :]
            + self.tau_aerosols[None, :, :] * self.aerosol_ss_alb[None, None, :]
        )

        return None
