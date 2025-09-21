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

    def __init__(self, model_inputs):
        """
        Initialize the SolarIrradiance class using model configuration inputs.
        
        This constructor extracts solar irradiance parameters from the given
        `model_inputs` object, including whether to use direct/diffuse radiation,
        the solar zenith angle (SZA), and the irradiance type. It then triggers
        irradiance processing via the `set_irradiance()` method.
    
        Parameters
        ----------
        model_inputs : ModelInputs
            An instance of the ModelInputs class containing model input data parsed
            from the YAML configuration file.
        """
        self.model_inputs = model_inputs
        self.direct = model_inputs.inputs["RTM"]["DIRECT"]
        self.sza = model_inputs.inputs["RTM"]["SZA"]
        self.irradiance_type = model_inputs.inputs["RTM"]["IRRADIANCE_TYPE"]

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
        - `cos_sza` : float
            Cosine of the solar zenith angle.
        """

        self.cos_sza = np.cos(np.deg2rad(np.rint(self.sza)))

        if self.direct:

            flux_file = xr.open_dataset(
                str(
                    self.model_inputs.solar_fluxes_path
                    + 'swnb_480bnd_'
                    + self.irradiance_type
                    + "_clr_"
                    + str("SZA" + str(self.sza).rjust(2, "0"))
                    + ".nc"
                )
            )
        else:

            flux_file = xr.open_dataset(
                str(
                    self.model_inputs.solar_fluxes_path
                    + 'swnb_480bnd_'
                    + self.irradiance_type
                    + "_cld.nc"
                )
            )

        # interp at the correct spectral resolution
        self.flx_slr = np.interp(
            np.arange(self.model_inputs.inputs["RTM"]["WVL_START"], 
                      self.model_inputs.inputs["RTM"]["WVL_END"], 
                      self.model_inputs.inputs["RTM"]["RESOLUTION"]),
            flux_file.wvl_ctr.values * 1e3,
            flux_file["flx_frc_sfc"].values,
        )
        
        # normalize
        
        self.flx_slr = self.flx_slr / np.sum(self.flx_slr)

        self.flx_slr[self.flx_slr == 0] = 1e-30

        if self.direct:
            self.fs = self.flx_slr / (self.cos_sza * np.pi)
            self.fd = np.zeros_like(self.fs)
        else:
            self.fd = self.flx_slr 
            self.fs = np.zeros_like(self.fd)
