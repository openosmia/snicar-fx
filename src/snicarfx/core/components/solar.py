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

    def __init__(self, config, PACKAGE_ROOT):
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
        self.PACKAGE_ROOT = PACKAGE_ROOT

        self.sky_conditions = config.ATMOSPHERE.SKY_CONDITIONS
        self.sza = config.SOLAR.SZA
        self.wavelengths = (
            np.arange(
                config.SOLVER.WVL_START,
                config.SOLVER.WVL_END,
                config.SOLVER.RESOLUTION,
            )
            * 1e-9
        )
        self.atmosphere_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

        # set irradiance based on user inputs
        self.set_irradiance()

    def set_irradiance(self):
        """
        Load and compute the solar spectral irradiance.

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

        # read libradtran surface irradiance and index on given SZA
        ds = xr.open_dataset(
            str(
                f"{self.PACKAGE_ROOT}/data/solar_fluxes/"
                + f"libradtranv206_surface_irradiance"
                + f"_{self.atmosphere_type}_{self.sky_conditions}.nc"
            )
        ).sel(SZA=self.sza)

        # wvl in these files are in um --> convert wvl from m to um
        ds_interpolated = ds.interp(wavelength=self.wavelengths * 1e9)

        # normalize each irradiance for the spectral sum to be equal to 1
        irradiance_sum = ds_interpolated["irradiance"].sum(dim="wavelength")
        irradiance_normalized = ds_interpolated["irradiance"] / irradiance_sum

        # replace 0 by 1e-30 to avoid invalid operations
        irradiance_normalized = irradiance_normalized.clip(min=1e-30)

        self.fs = irradiance_normalized.sel(irradiance_type="direct").values
        self.fd = irradiance_normalized.sel(irradiance_type="diffuse").values

        # solar flux is direct + diffuse
        self.flx_slr = self.fs + self.fd
