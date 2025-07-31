import math

import numpy as np
import xarray as xr


class SolarIrradiance:
    """Properties of incoming irradiance.

    Instances of Illumination contain all data relating to the incoming irradiance.

    Attributes:
        direct: Boolean toggling between direct and diffuse irradiance
        solzen: solar zenith angle in degrees from the vertical
        incoming: choice of spectral distribution from file 0-6
        flx_dir: directory containing irradiance files
        stubs: array of stub strings for selecting irradiance files
    """

    def __init__(self, modelconfig):
        self.modelconfig = modelconfig
        self.direct = modelconfig.inputs["RTM"]["DIRECT"]
        self.solzen = modelconfig.inputs["RTM"]["SOLZEN"]
        self.incoming = modelconfig.inputs["RTM"]["INCOMING"]

        self.stubs = [
            f"swnb_480bnd_{i}"
            for i in ["mlw", "mls", "saw", "sas", "smm", "hmn", "trp"]
        ]

        self.calculate_irradiance()

    def calculate_irradiance(self):
        """Calculates irradiance from initialized attributes.

        Takes mu_not, incoming and file stubs from self and calculates irradiance.

        Args:
            self

        Returns:
            flx_slr: incoming flux from file
            fd: diffuse irradiance
            fs: direct irradiance


        Raises:
            ValueError is incoming is out of range
        """

        if self.incoming < 0 or self.incoming > 6:
            raise ValueError("Irradiance type out of range - between 0 and 6 only")

        # update mu_not from solzen
        self.mu_not = np.cos(math.radians(np.rint(self.solzen)))

        if self.direct:

            incoming_file = xr.open_dataset(
                str(
                    self.modelconfig.solar_fluxes_path
                    + self.stubs[self.incoming]
                    + "_clr_"
                    + str("SZA" + str(self.solzen).rjust(2, "0"))
                    + ".nc"
                )
            )
        else:

            incoming_file = xr.open_dataset(
                str(
                    self.modelconfig.solar_fluxes_path
                    + self.stubs[self.incoming]
                    + "_cld"
                    + ".nc"
                )
            )

        # set spectral resolution
        resolution = self.modelconfig.inputs["RTM"]["RESOLUTION"]
        wvl_start = self.modelconfig.inputs["RTM"]["WVL_START"]
        wvl_end = self.modelconfig.inputs["RTM"]["WVL_END"]

        # interp at the correct spectral resolution
        self.flx_slr = np.interp(
            np.arange(wvl_start, wvl_end, resolution),
            incoming_file.wvl_ctr.values * 1e3,  # from um to nm
            incoming_file["flx_frc_sfc"].values,
        )

        self.flx_slr[self.flx_slr <= 0] = 1e-30

        out = self.flx_slr / (self.mu_not * np.pi)

        if self.direct:
            self.fs = out
            self.fd = np.zeros_like(out)
        else:
            self.fd = out
            self.fs = np.zeros_like(out)
