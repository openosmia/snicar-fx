"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from ..components.land import LandColumn
from ..components.atmosphere import AtmosphereColumn
from ..components.solar import SolarIrradiance
from .config import Config
from ..solvers.two_stream_solver import solve_two_stream_rt
from ..solvers.multi_stream_solver import solve_multi_stream_rt
import pathlib
import sys
from importlib.metadata import version
import xarray as xr
from datetime import datetime
import numpy as np


class Session:
    """
    Session driver orchestrating the input configuation, the
    different snicarfx objects, the solvers, run execution and output
    storage.

    It also provides a simple API to update certain input fields in
    the different components at run time.
    """

    def __init__(self, input_file: str):

        # parse configuration file
        self.config = Config.from_yaml(input_file)

        # set module root path for data loading
        self.config._ROOT_PATH = self.get_package_root()

        # set spectral range arrays based on user inputs
        if isinstance(self.config.SOLVER.SPECTRAL_RANGE, tuple):
            self.config._wavelengths = np.arange(
                self.config.SOLVER.SPECTRAL_RANGE[0],
                self.config.SOLVER.SPECTRAL_RANGE[1],
                self.config.SOLVER.SPECTRAL_RANGE[2],
            )

        elif self.config.SOLVER.SPECTRAL_RANGE == "SENTINEL-3-OLCI":

            ds = xr.open_dataset(
                f"{self.config._ROOT_PATH}/data/satellite_spectral_responses/S3A_OL_SRF_20160713_mean_rsr.nc4"
            )
            self.config._wavelengths_srf = (
                ds.mean_spectral_response_function_wavelength.values
            )
            self.config._spectral_response_function = (
                ds.mean_spectral_response_function.values
            )

            if self.config.SOLVER.SPECTRAL_MODE == "monochromatic":

                mins_per_band_wavelength = np.nanmin(
                    ds.mean_spectral_response_function_wavelength.values, axis=1
                )
                maxs_per_band_wavelength = np.nanmax(
                    ds.mean_spectral_response_function_wavelength.values, axis=1
                )

                min_global_wavelength = (
                    np.nanmin(ds.mean_spectral_response_function_wavelength.values)
                    * 1e-7
                )
                max_global_wavelength = (
                    np.nanmax(ds.mean_spectral_response_function_wavelength.values)
                    * 1e-7
                )

                wavelength_array = (
                    1e7
                    / np.arange(
                        1 / max_global_wavelength, 1 / min_global_wavelength, 1
                    )[::-1]
                )

                mask = (
                    (wavelength_array[:, None] >= mins_per_band_wavelength)
                    & (wavelength_array[:, None] <= maxs_per_band_wavelength)
                ).any(axis=1)

                self.config._wavelengths = wavelength_array[mask]

        # build components
        self.land_column = LandColumn(self.config)
        self.solar_irradiance = SolarIrradiance(self.config)
        self.atmosphere_column = AtmosphereColumn(self.config)

        # store history of updates
        self._latest_updates = {
            "SOLVER": {},
            "SOLAR": {},
            "ATMOSPHERE": {},
            "LAND": {},
        }

        # save outputs
        self.outputs = None

    def _prepare_updates(self, kwargs, allowed_fields):
        forbidden = set(kwargs) - allowed_fields
        if forbidden:
            raise ValueError(
                f"Cannot update: {', '.join(forbidden)}. "
                "These fields cannot be changed at runtime. "
                "Please modify them in the input YAML file."
            )

        return {k: v for k, v in kwargs.items() if v is not None}

    def _write_current_state(self):
        """Write current session state to dictionnary, to be addded to
        the input file"""

        current_state = self.config.dict()

        # merge applied updates on top of the static config
        for section, section_updates in self._latest_updates.items():
            current_state[section].update(section_updates)

        return current_state

    def update_solver(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"TYPE"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._latest_updates["SOLVER"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"SOLVER": updates})

        if "TYPE" in updates:
            pass

    def update_solar(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"SZA"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._latest_updates["SOLAR"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"SOLAR": updates})

        # update SZA and recompute irradiance
        if "SZA" in updates:
            self.solar_irradiance.sza = updates["SZA"]
            self.solar_irradiance.set_irradiance()

    def update_atmosphere(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"SKY_CONDITIONS"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._latest_updates["ATMOSPHERE"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"ATMOSPHERE": updates})

        # update SZA and sky conditions, reload irradiance file and recompute
        if "SKY_CONDITIONS" in updates:
            self.solar_irradiance.sky_conditions = updates["SKY_CONDITIONS"]
            self.solar_irradiance.load_irradiance()
            self.solar_irradiance.set_irradiance()

    def update_land(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {
            "THICKNESS",
            "LAYER_TYPE",
            "DENSITY",
            "SPECIFIC_SURFACE_AREA",
            "LWC",
            "GRAIN_SHAPE",
            "RF_TYPE",
            "SFC",
        }
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._latest_updates["LAND"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"LAND": updates})

        if "SKY_CONDITIONS" in updates:
            pass

    def run(self, to_xarray=True):
        """
        Run the radiative transfer solver and return outputs
        """

        if self.config.SOLVER.TYPE == "two-stream":
            self.outputs = solve_two_stream_rt(self.land_column, self.solar_irradiance)
            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.two_stream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream":
            self.outputs = solve_multi_stream_rt(
                self.land_column, self.atmosphere_column, self.solar_irradiance
            )

            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.multi_stream_results_to_xarray()

        return self.outputs

    @staticmethod
    def get_package_root() -> pathlib.Path:
        """
        Return the root path of the snicarfx package.
        """
        snicarfx_module = sys.modules["snicarfx"]
        snicarfx_root_path = pathlib.Path(snicarfx_module.__file__).resolve().parents[2]

        return snicarfx_root_path

    def two_stream_results_to_xarray(self) -> xr.Dataset:
        """
        Save results to an xarray with rather extensive model and
        session state metadata.
        """

        attrs = {
            "model_name": "snicar-fx",
            "model_version": version("snicarfx"),
            "model_url": "https://github.com/openosmia/snicar-fx",
            "creation_date": datetime.utcnow().isoformat(),
            "session_state": self._write_current_state(),
        }

        ds = xr.Dataset(
            data_vars={
                "albedo": ("wavelength", self.outputs.albedo),
                "BBA": self.outputs.BBA,
                "absorbed_flux_fraction": (
                    "layer",
                    self.outputs.absorbed_flux_fraction_per_layer,
                ),
                "absorbed_flux_fraction_bottom": self.outputs.absorbed_flux_fraction_bottom,
            },
            coords={
                "wavelength": self.outputs.wavelengths,
                "layer": np.arange(len(self.outputs.absorbed_flux_fraction_per_layer)),
            },
        )

        # add attributes
        ds.attrs.update(attrs)

        return ds

    def multi_stream_results_to_xarray(self) -> xr.Dataset:
        """
        Save results to an xarray with rather extensive model and
        session state metadata.
        """

        # build wavelength array here (as it is not needed in the solver)
        wavelengths = np.arange(
            self.config.SOLVER.SPECTRAL_RANGE[0],
            self.config.SOLVER.SPECTRAL_RANGE[1],
            self.config.SOLVER.SPECTRAL_RANGE[2],
        )

        attrs = {
            "model_name": "snicar-fx",
            "model_version": version("snicarfx"),
            "model_url": "https://github.com/openosmia/snicar-fx",
            "creation_date": datetime.utcnow().isoformat(),
            "session_state": self._write_current_state(),
        }

        ds = xr.Dataset(
            data_vars={
                "albedo": ("wavelength", self.outputs.albedo),
                "gaussian_integration_angle": (
                    "angle",
                    self.outputs.cos_angle,
                ),
                "gaussian_integration_weight": (
                    "angle",
                    self.outputs.cos_weight,
                ),
                "directional_reflectance_top": (
                    ("angle", "wavelength"),
                    self.outputs.directional_reflectance_top,
                ),
            },
            coords={
                "wavelength": wavelengths,
                "angle": np.arange(len(self.outputs.cos_weight)),
            },
        )

        # add attributes
        ds.attrs.update(attrs)

        return ds
