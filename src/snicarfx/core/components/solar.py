"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr


class SolarIrradiance:
    """
    Compute and and store the properties of the incoming solar irradiance.

    Attributes
    ----------
    model_inputs : ModelInputs
        An instance of the ModelInputs class containing model input data parsed
        from the YAML configuration file.
    direct : boolean
        If True, use direct radiation; if False, use diffuse radiation.
    sza : int
        Solar zenith angle (SZA).
    atmosphere_type : str
        The atmopshere profile to use.
    """

    def __init__(self, config):
        """
        Initialize the SolarIrradiance class using model configuration inputs.

        This constructor extracts solar irradiance parameters from the given
        `model_inputs` object, including whether to use direct/diffuse radiation,
        the solar zenith angle (SZA), and the irradiance type. It then triggers
        irradiance processing via the `set_irradiance()` method.

        Parameters
        ----------
        config : Config
            An instance of the Config class containing model input data parsed
            from the YAML configuration file.
        """

        # set module root path for data loading
        self.ROOT_PATH = config._ROOT_PATH

        self.wavelengths = config._wavelengths

        self.sky_conditions = config.ATMOSPHERE.SKY_CONDITIONS

        self.sza = config.SOLAR.SZA

        self.atmosphere_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

        # set irradiance based on user inputs
        if config.SOLVER.ATMOSPHERE_COUPLING:
            self.irradiance_dataset = self.load_toa_irradiance()
            self.set_toa_irradiance(config)

        elif not config.SOLVER.ATMOSPHERE_COUPLING:
            self.irradiance_dataset = self.load_surface_irradiance()
            self.set_surface_irradiance()

    def load_surface_irradiance(self):
        """
        Load irradiance file: surface irradiance profiles modelled with
        LibRadTran for a range of SZAs.
        """

        # read libradtran surface irradiance
        ds = xr.open_dataset(
            str(
                f"{self.ROOT_PATH}/data/solar_fluxes/"
                + "libradtranv206_surface_irradiance"
                + f"_{self.atmosphere_type}_{self.sky_conditions}.nc"
            )
        )

        return ds

    def load_toa_irradiance(self):
        """
        Load TOA irradiance profile: Version 2 of the TSIS-1 Hybrid Solar
        Reference Spectrum from Coddington et al. 2022.
        """

        ds = xr.open_dataset(
            str(
                f"{self.ROOT_PATH}/data/solar_fluxes/"
                + "hybrid_reference_spectrum_p025nm_resolution_c2022-11-30_with_unc.nc"
            )
        )

        return ds

    def set_toa_irradiance(self, config):
        """
        Compute solar spectral irradiance at top-of-atmosphere (TOA) on the
        user-defined wavelength grid / satellite bands.
        """

        self.irradiance_dataset["wavelength"] = self.irradiance_dataset[
            "Vacuum Wavelength"
        ]

        # 1: if satellite bands
        if config.SOLVER.SPECTRAL_RANGE == "SENTINEL-3-OLCI":

            # interpolate SENTINEL-3-OLCI SRF on TOA irradiance wavelengths
            # srf_on_toa_grid = np.vstack(
            #     [
            #         np.interp(
            #             self.irradiance_dataset["Vacuum Wavelength"].values,
            #             config._wavelengths_srf[band_number, :],
            #             config._spectral_response_function[band_number, :],
            #         )
            #         for band_number in range(21)
            #     ]
            # )

            # toa_irradiance = self.irradiance_dataset.SSI.values

            # # collapse TOA irradiance on S3 bands
            # irradiance_on_bands = np.nansum(
            #     (srf_on_toa_grid * toa_irradiance[None, :]), axis=1
            # ) / (np.nansum(srf_on_toa_grid, axis=1))

            self.flx_slr = self.irradiance_dataset.interp(
                wavelength=config._wavelengths
            ).SSI.values  # irradiance_on_bands  # W m-2 nm-1

        else:

            # # group into wavelength bins as per user-defined wvl grid
            # grouped = self.irradiance_dataset.SSI.groupby_bins(
            #     "wavelength", config._wavelengths * 1e9, right=False
            # )

            # # trapezoidal integration to get the flux in the bands
            # self.flx_slr = grouped.apply(lambda x: x.integrate("wavelength"))

            self.flx_slr = self.irradiance_dataset.interp(
                wavelength=config._wavelengths * 1e-9
            ).SSI.values

        return None

    def set_surface_irradiance(self):
        """
        Compute the solar spectral irradiance at the surface.

        This method interpolates the solar flux to the spectral resolution
        defined in the input file and loads the direct (`fs`) and
        diffuse (`fd`) fluxes.

        The following instance attributes are set:
        - `flx_slr` : ndarray
            Normalized total spectral solar flux at the surface.
        - `fs` : ndarray
            Normalized spectral direct irradiance at the surface.
        - `fd` : ndarray
            Normalized spectral diffuse irradiance at the surface.
        """

        # index irradiance dataset on given SZA
        ds_sza = self.irradiance_dataset.sel(SZA=self.sza)

        # wvl in these files are in um --> convert wvl from m to um
        ds_sza_interpolated = ds_sza.interp(wavelength=self.wavelengths)

        irradiance_direct = ds_sza_interpolated.sel(irradiance_type="direct")[
            "irradiance"
        ]
        irradiance_diffuse = ds_sza_interpolated.sel(irradiance_type="diffuse")[
            "irradiance"
        ]

        # sum over wavelengths to get total
        irradiance_total_sum = (irradiance_direct + irradiance_diffuse).sum(
            dim="wavelength"
        )

        irradiance_direct_normalized = irradiance_direct / irradiance_total_sum
        irradiance_diffuse_normalized = irradiance_diffuse / irradiance_total_sum

        # replace 0s by 1e-30 to avoid invalid operations
        self.fs = irradiance_direct_normalized.clip(min=1e-30).values
        self.fd = irradiance_diffuse_normalized.clip(min=1e-30).values

        # solar flux is direct + diffuse
        self.flx_slr = self.fs + self.fd
