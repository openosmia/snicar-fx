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
    irradiance_type : str
        The irradiance profile to use.
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

        self.direct = config.ATMOSPHERE.SKY_CONDITIONS == "clear"
        self.sza = config.SOLAR.SZA
        self.wavelengths = (
            np.arange(
                config.SOLVER.WVL_START,
                config.SOLVER.WVL_END,
                config.SOLVER.RESOLUTION,
            )
            * 1e-9
        )

        # hardcoded for tests for now
        self.irradiance_type = "mls"
        self.set_irradiance()
        
        # to do: 
        # self.irradiance_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE
        # read file from selected profile
            # flux_file = xr.open_dataset(libradtran_file)
        # select wavelength range: 
            # solar_irradiance = flux_file.interp(wvl_ctr=self.wavelengths * 1e6) 
        # calculate fs, fd and flx_slr
            # fs = solar_irradiance.direct / (cos_sza *pi)
            # fd = solar_irradiance.diffuse
            # flx_slr = solar_irradiance.diffuse + solar_irradiance.direct
            
        # the indexing in SZA will be in the solvers directly


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

        if self.direct:

            flux_file = xr.open_dataset(
                str(
                    f"{self.PACKAGE_ROOT}/data/solar_fluxes/"
                    + "swnb_480bnd_"
                    + self.irradiance_type
                    + "_clr_"
                    + str("SZA" + str(self.sza).rjust(2, "0"))
                    + ".nc"
                )
            )
        else:

            flux_file = xr.open_dataset(
                str(
                    f"{self.PACKAGE_ROOT}/data/solar_fluxes/"
                    + "swnb_480bnd_"
                    + self.irradiance_type
                    + "_cld.nc"
                )
            )

        # wvl in these files are in um --> convert wvl from m to um
        self.flx_slr = flux_file.interp(wvl_ctr=self.wavelengths * 1e6)[
            "flx_frc_sfc"
        ].values

        # normalize

        self.flx_slr = self.flx_slr / np.sum(self.flx_slr)

        self.flx_slr[self.flx_slr == 0] = 1e-30
        
        cos_sza = np.cos(np.deg2rad(np.rint(self.sza)))

        if self.direct:
            self.fs = self.flx_slr / (cos_sza * np.pi)
            self.fd = np.zeros_like(self.fs)
        else:
            self.fd = self.flx_slr
            self.fs = np.zeros_like(self.fd)
