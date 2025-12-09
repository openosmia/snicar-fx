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

        # load irradiance file with SZA range
        self.irradiance_dataset = self.load_irradiance()

        # set irradiance based on user inputs
        self.set_irradiance()

    def load_irradiance(self):
        """
        Load irradiance file containing arrays for a range of SZAs
        """
        # read libradtran surface irradiance
        ds = xr.open_dataset(
            str(
                f"{self.ROOT_PATH}/data/solar_fluxes/"
                + f"libradtranv206_surface_irradiance"
                + f"_{self.atmosphere_type}_{self.sky_conditions}.nc"
            )
        )

        return ds

    def set_irradiance(self):
        """
        Compute the solar spectral irradiance.

        Based on the `direct` flag, this method loads either a clear-sky or
        cloudy-sky flux file. It interpolates the solar flux to the spectral
        resolution defined in the input file and computes direct (`fs`) and
        diffuse (`fd`) fluxes accordingly.

        The following instance attributes are set:
        - `flx_slr` : ndarray
            Normalized solar flux over the defined wavelength range.
        - `fs` : ndarray
            Spectral direct irradiance.
        - `fd` : ndarray
            Spectral diffuse irradiance.
        """

        # index irradiance dataset on given SZA
        ds_sza = self.irradiance_dataset.sel(SZA=self.sza)

        # wvl in these files are in um --> convert wvl from m to um
        ds_sza_interpolated = ds_sza.interp(wavelength=self.wavelengths * 1e9)

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
