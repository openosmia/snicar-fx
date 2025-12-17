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
        self.wavelengths = config._wavelengths

        self.n_expansion = config.SOLVER.N_LEGENDRE_MOMENTS
        self.surface_elevation = config.LAND.ALTITUDE

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

            # load gas cross sections
            self.load_gas_absorption_cross_sections()

            self.compute_gas_optical_thickness()

            self.set_atmospheric_properties_wout_aerosols()

            self.apply_spectral_response_function(config)

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

    def load_gas_absorption_cross_sections(self):

        # get absorption in (c)m2 / molecule for each gas
        self.gas_cross_sections = xr.open_dataset(
            f"{self.ROOT_PATH}/data/atmospheric_profiles/uvspec_afglss_test_file_cross_sections.nc"
        )
        self.gas_cross_sections["nwvl"] = self.gas_cross_sections.wvl

        self.gas_cross_sections = self.gas_cross_sections.interp(nwvl=self.wavelengths)

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

        # hapi2libis_data = xr.open_dataset(
        #     f"{self.ROOT_PATH}/data/atmospheric_profiles/uvspec_afglss_test_file.nc"
        # )
        # hapi2libis_data["nwvl"] = hapi2libis_data.wvl
        # self.tau_gases = hapi2libis_data["tau"].interp(nwvl=self.wavelengths).values

        # if z = 1, remove layer 0 ie index at 1
        # if z = 2, remove layer 0+1 ie index at 2, etc

        self.tau_gases = self.tau_gases[int(self.surface_elevation) :, :]

        return None

    def set_atmospheric_properties_wout_aerosols(self):

        self.tau = self.tau_molecular_scatter + self.tau_gases
        self.ss_alb = self.tau_molecular_scatter / (
            self.tau_gases + self.tau_molecular_scatter
        )
        self.legendre_moments = self.rayleigh_legendre_moments

        return None

    def apply_spectral_response_function(self, config):

        if config.SOLVER.SPECTRAL_RANGE == "SENTINEL-3-OLCI":

            # interpolate SENTINEL-3-OLCI response on cross-section wavelengths
            srf_on_crs_grid = np.vstack(
                [
                    np.interp(
                        self.gas_cross_sections.wvl,
                        config._wavelengths_srf[band_number, :],
                        config._spectral_response_function[band_number, :],
                    )
                    for band_number in range(21)
                ]
            )
