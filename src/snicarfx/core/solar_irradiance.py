import math

import numpy as np
import xarray as xr


class SolarIrradiance:
    """Properties of incoming irradiance.

    Instances of Illumination contain all data relating to the incoming irradiance.

    Attributes:
        direct: Boolean toggling between direct and diffuse irradiance
        sza: solar zenith angle in degrees from the vertical
        incoming: choice of type of irradiance type
        flx_dir: directory containing irradiance files
    """

    def __init__(self, model_inputs):
        self.model_inputs = model_inputs
        self.direct = model_inputs.inputs["RTM"]["DIRECT"]
        self.sza = model_inputs.inputs["RTM"]["SZA"]
        self.irradiance_type = model_inputs.inputs["RTM"]["IRRADIANCE_TYPE"]

        self.set_irradiance()

    def set_irradiance(self):
        """Calculates irradiance from initialized attributes.

        Args:
            self

        Returns:
            flx_slr: incoming flux from file
            fd: diffuse irradiance
            fs: direct irradiance

        """

        self.cos_sza = np.cos(math.radians(np.rint(self.sza)))

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
            flux_file.wvl_ctr.values * 1e3,  # from um to nm
            flux_file["flx_frc_sfc"].values,
        )

        self.flx_slr[self.flx_slr == 0] = 1e-30

        if self.direct:
            self.fs = self.flx_slr / (self.cos_sza * np.pi)
            self.fd = np.zeros_like(self.fs)
        else:
            self.fd = self.flx_slr 
            self.fs = np.zeros_like(self.fd)
