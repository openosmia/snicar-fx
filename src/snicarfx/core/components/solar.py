"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


class SolarIrradiance:
    """
    Compute and store the properties of the incoming solar irradiance to be
    used as boundary for the solver (top-of-atmosphere or bottom-of-atmosphere)
    based on the YAML input file.

    Attributes
    ----------
    ROOT_PATH : str
        Path to the snicarfx module.
    sky_conditions : str
        Type of sky conditions ('cloudy' or 'clear').
    atmosphere_type : str
        Type of atmospheric profile (AFGL tag).
    sza : int
        Solar zenith angle in degrees.
    saa : int
        Solar azimuth angle in degrees.
    _wavelengths : ndarray
        Wavelength grid (nm).
    total_irradiance : ndarray
        Total spectral solar irradiance (normal incidence).
    direct_beam : ndarray
        Direct solar spectral irradiance (normal incidence).
    diffuse : ndarray
        Diffuse solar spectral irradiance (isotropic).

    """

    def __init__(self, config):
        """
        Initialize the SolarIrradiance class using model configuration inputs.

        This constructor extracts parameters from the given
        `config` object and then triggers irradiance processing via the
        `set_irradiance()` method.

        Parameters
        ----------
        config : Config
            An instance of the Config class containing model input data parsed
            from the YAML configuration file.
        """

        self.ROOT_PATH = config._ROOT_PATH

        self.sky_conditions = config.ATMOSPHERE.SKY_CONDITIONS

        self.sza = config.SOLAR.SZA

        self.saa = config.SOLAR.SAA

        self.atmosphere_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

        self._wavelengths = config._wavelengths_solar

        if config.SOLVER.ATMOSPHERE_COUPLING:
            self.irradiance_dataset = self.load_toa_irradiance()
            self.set_toa_irradiance()

        else:
            self.irradiance_dataset = self.load_surface_irradiance()
            self.set_surface_irradiance()

    def load_surface_irradiance(self):
        """
        Load surface irradiance file.

        The surface irradiance file stores pre-computed irradiances simulated
        with LibRadTran for a given atmospheric profile and range of SZAs.
        """

        tag = self.sky_conditions.split("_")[0]

        ds = (
            xr.open_dataset(
                str(
                    f"{self.ROOT_PATH}/data/solar_fluxes/"
                    + "libradtranv206_surface_irradiance_clean_ice"
                    + f"_{self.atmosphere_type}_{tag}.nc"
                )
            )
            / 1e3
        )

        return ds

    def load_toa_irradiance(self):
        """
        Load top-of-atmosphere (TOA) irradiance file.

        The TOA irradiance file corresponds Version 2 of the TSIS-1 Hybrid
        Solar Reference Spectrum from Coddington et al. 2022.
        """

        ds = xr.open_dataset(
            str(
                f"{self.ROOT_PATH}/data/solar_fluxes/"
                + "hybrid_reference_spectrum_p025nm_resolution_c2022-11-30_with_unc.nc"
            )
        )

        return ds

    def set_toa_irradiance(self):
        """
        Set monochromatic top-of-atmosphere (TOA) solar spectral irradiance
        array used as boundary for the solver.

        This method interpolates the TOA spectral irradiance to the solar
        wavelength array.
        """

        self.irradiance_dataset["wavelength"] = self.irradiance_dataset[
            "Vacuum Wavelength"
        ]

        self.direct_beam = self.irradiance_dataset.interp(
            wavelength=self._wavelengths
        ).SSI.values

        self.total_irradiance = self.direct_beam

        self.diffuse = self.direct_beam * 0

        return None

    def set_surface_irradiance(self):
        """
        Set surface solar spectral irradiance arrays used as
        boundary for the solver.

        This method selects the irradiance corresponding to the user-input
        solar zenith angle (SZA) and interpolates the spectral irradiance to
        the solar wavelength array.
        """

        ds_sza = self.irradiance_dataset.sel(SZA=self.sza)

        ds_sza = ds_sza.interp(
            wavelength=self._wavelengths, kwargs={"fill_value": "extrapolate"}
        )

        irradiance_direct = (
            ds_sza.sel(irradiance_type="direct")["irradiance"].clip(min=1e-30).values
        )
        irradiance_diffuse = (
            ds_sza.sel(irradiance_type="diffuse")["irradiance"].clip(min=1e-30).values
        )

        if self.sky_conditions == "clear":
            # convert direct horizontal to direct normal incident irradiance
            self.direct_beam = irradiance_direct / np.cos(np.deg2rad(self.sza))
            # convert direct horizontal to isotropic
            self.diffuse = irradiance_diffuse / np.pi

        if self.sky_conditions == "clear_fully_direct":
            # convert direct horizontal to direct normal incident irradiance
            self.direct_beam = (irradiance_direct + irradiance_diffuse) / np.cos(
                np.deg2rad(self.sza)
            )
            self.diffuse = irradiance_diffuse * 0

        # solar flux is direct + diffuse
        self.total_irradiance = self.direct_beam + self.diffuse * np.pi
