"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import xarray as xr


class SolarIrradiance:
    """
    Compute and and store the properties of the incoming solar irradiance to be
    used as boundary for the solver (top-of-atmosphere or bottom-of-atmosphere).

    Attributes
    ----------
    config : Config
        An instance of the Config class containing model input data parsed
        from the YAML configuration file.
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

        self.atmosphere_type = config.ATMOSPHERE.ATMOSPHERIC_PROFILE_TYPE

        if config.SOLVER.ATMOSPHERE_COUPLING:
            self.irradiance_dataset = self.load_toa_irradiance()
            self.set_toa_irradiance(config)

        elif not config.SOLVER.ATMOSPHERE_COUPLING:
            self.irradiance_dataset = self.load_surface_irradiance()
            self.set_surface_irradiance(config)

    def load_surface_irradiance(self):
        """
        Load surface irradiance file.
        
        The surface irradiance file stores pre-computed irradiances simulated 
        with LibRadTran for a given atmospheric profile and range of SZAs.
        """

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

    def set_toa_irradiance(self, config):
        """
        Set monochromatic top-of-atmosphere (TOA) solar spectral irradiance 
        array used as boundary for the solver.
        
        This method interpolates the TOA spectral irradiance to the solar
        wavelength array. 
        
        The following instance attributes are set:
        - `flx_slr` : ndarray
            Total spectral solar irradiance at TOA.
        """

        self.irradiance_dataset["wavelength"] = self.irradiance_dataset[
            "Vacuum Wavelength"
        ]

        self.flx_slr = self.irradiance_dataset.interp(
            wavelength=config._wavelengths_solar
        ).SSI.values  

        return None

    def set_surface_irradiance(self, config):
        """
        Set monochromatic surface solar spectral irradiance array used as 
        boundary for the solver.

        This method selects the irradiance corresponding to the user-input 
        solar zenith angle (SZA) and interpolates the spectral irradiance to 
        the solar wavelength array. 

        The following instance attributes are set:
        - `flx_slr` : ndarray
            Total (diffuse + direct) spectral solar irradiance at the surface.
        - `fs` : ndarray
            Spectral direct irradiance at the surface.
        - `fd` : ndarray
            Spectral diffuse irradiance at the surface.
        """


        ds_sza = self.irradiance_dataset.sel(SZA=self.sza)

        ds_sza = ds_sza.interp(wavelength=config._wavelengths_solar,
                               kwargs={"fill_value": "extrapolate"})

        irradiance_direct = ds_sza.sel(irradiance_type="direct")["irradiance"]
        irradiance_diffuse = ds_sza.sel(irradiance_type="diffuse")["irradiance"]

        # replace 0s by 1e-30 to avoid invalid operations
        self.fs = irradiance_direct.clip(min=1e-30).values
        self.fd = irradiance_diffuse.clip(min=1e-30).values

        # solar flux is direct + diffuse
        self.flx_slr = self.fs + self.fd
