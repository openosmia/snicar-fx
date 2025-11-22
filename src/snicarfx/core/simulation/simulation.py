"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from snicarfx.core import LandColumn, AtmosphereColumn, SolarIrradiance
from snicarfx.core import Config
from snicarfx.core.two_stream_solver import solve_two_stream_rt


class Simulation:
    def __init__(self, input_file: str):

        # parse configuration file
        self.config = Config.from_yaml(input_file)

        # build components
        self.land_column = LandColumn(self.config)
        self.solar_irradiance = SolarIrradiance(self.config)
        self.atmosphere_column = AtmosphereColumn(self.config)

    def run(self):
        """Run the radiative transfer solver."""
        self.outputs = solve_two_stream_rt(self.column, self.irradiance)
        return self.outputs

    def to_dataset(self):
        """Optional: return results and column properties as an xarray Dataset."""
        ds = self.column.to_dataset()
        ds["irradiance"] = (("wavelength"), self.irradiance.spectral_flux)
        if self.outputs is not None:
            ds["rtm_output"] = (("wavelength", "layer"), self.outputs)
        return ds
