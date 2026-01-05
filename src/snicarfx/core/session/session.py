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

        # set spectral parameters based on user input
        self._set_spectral_array()

        # build components
        self.land_column = LandColumn(self.config)
        self.solar_irradiance = SolarIrradiance(self.config)
        self.atmosphere_column = AtmosphereColumn(self.config)

        if self.config.SPECTRAL.MODE == "band":
            self.compute_band_average()

        # elif self.config.SPECTRAL.BAND_METHOD == "snicar-default":
        # store history of updates
        self._latest_updates = {
            "SOLVER": {},
            "SOLAR": {},
            "ATMOSPHERE": {},
            "LAND": {},
        }

        # save outputs
        self.outputs = None

    def _set_spectral_array(self):

        # if the wavelength array is passed by user
        if isinstance(self.config.SPECTRAL.RESOLUTION, tuple):

            # create a homogeneous array based on user input
            wavelength_homogeneous = np.arange(*self.config.SPECTRAL.RESOLUTION)
            center_wavelength_homogeneous = (
                wavelength_homogeneous[:-1] + np.diff(wavelength_homogeneous) / 2
            )
            band_ranges_homogeneous = np.column_stack(
                (
                    wavelength_homogeneous[:-1],
                    wavelength_homogeneous[1:],
                    center_wavelength_homogeneous,
                )
            )

            # create an array at 1cm-1 resolution
            min_wavelength = self.config.SPECTRAL.RESOLUTION[0]
            max_wavelength = self.config.SPECTRAL.RESOLUTION[1]
            wavelength_1cm_m1 = (
                1e7
                / np.arange(
                    1 / (max_wavelength * 1e-7),
                    1 / (min_wavelength * 1e-7),
                    1,
                )[::-1]
            )

            if self.config.SPECTRAL.MODE == "monochromatic":
                self.config._wavelengths_solar = wavelength_homogeneous
                self.config._wavelengths_land = wavelength_homogeneous
                self.config._wavelengths_atmosphere = wavelength_homogeneous

            elif self.config.SPECTRAL.MODE == "band":
                self.config._wavelengths_solar = wavelength_1cm_m1
                self.config._wavelengths_land = wavelength_1cm_m1
                self.config._wavelengths_atmosphere = wavelength_1cm_m1

                self._band_ranges = band_ranges_homogeneous

                if self.config.SPECTRAL.BAND_METHOD == "snicar-default":
                    self.config._wavelengths_land = center_wavelength_homogeneous

        if self.config.SPECTRAL.RESOLUTION == "SENTINEL-3-OLCI":

            ds = xr.open_dataset(
                f"{self.config._ROOT_PATH}/data/satellite_spectral_responses/S3A_OL_SRF_20160713_mean_rsr.nc4"
            )
            self._wavelengths_srf = ds.mean_spectral_response_function_wavelength.values
            self._spectral_response_function = ds.mean_spectral_response_function.values

            # create a global wavelength array
            mins_per_band_wavelength = np.nanmin(
                ds.mean_spectral_response_function_wavelength.values, axis=1
            )
            maxs_per_band_wavelength = np.nanmax(
                ds.mean_spectral_response_function_wavelength.values, axis=1
            )

            min_wavelength = np.nanmin(
                ds.mean_spectral_response_function_wavelength.values
            )
            max_wavelength = np.nanmax(
                ds.mean_spectral_response_function_wavelength.values
            )

            restricted_wavelength_1cm_m1 = (
                1e7
                / np.arange(
                    1 / (max_wavelength * 1e-7),
                    1 / (min_wavelength * 1e-7),
                    1,
                )[::-1]
            )

            mask = (
                (restricted_wavelength_1cm_m1[:, None] >= mins_per_band_wavelength)
                & (restricted_wavelength_1cm_m1[:, None] <= maxs_per_band_wavelength)
            ).any(axis=1)

            self.config._wavelengths_solar = restricted_wavelength_1cm_m1[mask]
            self.config._wavelengths_land = restricted_wavelength_1cm_m1[mask]
            self.config._wavelengths_atmosphere = restricted_wavelength_1cm_m1[mask]

            self._band_ranges = np.column_stack(
                (
                    mins_per_band_wavelength,
                    maxs_per_band_wavelength,
                    ds.nominal_centre_wavelength.values,
                )
            )

            if self.config.SPECTRAL.BAND_METHOD == "snicar-default":
                self.config._wavelengths_land = ds.nominal_centre_wavelength.values

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

            if self.config.SPECTRAL.BAND_METHOD == "srf-integration":
                self.apply_spectral_response_function()

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
                "wavelength": (
                    self._band_ranges[:, -1]
                    if self.config.SPECTRAL.MODE == "band"
                    else self.config._wavelengths_land
                ),
                "angle": np.arange(len(self.outputs.cos_weight)),
            },
        )

        # add attributes
        ds.attrs.update(attrs)

        return ds

    def apply_spectral_response_function(self) -> None:

        # interpolate SENTINEL-3-OLCI SRF on homogeneous grid
        # (self.config._wavelengths_land could be any self.config._wavelengths_*
        # since they are the same in band_method srf-integration)
        srf_on_grid = np.vstack(
            [
                np.interp(
                    self.config._wavelengths_land,
                    self._wavelengths_srf[band_number, :],
                    self._spectral_response_function[band_number, :],
                )
                for band_number in range(21)
            ]
        )

        self.outputs.directional_reflectance_top = (
            np.nansum(
                self.outputs.directional_reflectance_top[:, None, :]
                * srf_on_grid[None, :, :],
                axis=-1,
            )
            / np.nansum(srf_on_grid, axis=1)[None, :]
        )

        self.outputs.albedo = np.nansum(
            self.outputs.albedo[None, :] * srf_on_grid,
            axis=-1,
        ) / np.nansum(srf_on_grid, axis=1)

        # overwrite high-resolution wavelength array (used for
        # computation) with center wavelengths
        self.config._wavelengths = self._band_ranges[:, -1]

    def compute_band_average(self) -> None:

        def compute_flat_band_mean(column, wavelengths, band_ranges, var_names):
            """
            Flat (unweighted) band mean using trapezoidal integration.
            Works for variables of shape (..., N_wavelengths).
            """

            band_means = {}

            for name in var_names:
                arr = getattr(column, name)  # (..., N)
                leading_shape = arr.shape[:-1]
                N = arr.shape[-1]

                arr2d = arr.reshape(-1, N)  # (K, N)
                K = arr2d.shape[0]

                out = np.empty((K, band_ranges.shape[0]))

                for b, (lam_min, lam_max, _) in enumerate(band_ranges):
                    # indices inside band
                    i0 = np.searchsorted(wavelengths, lam_min, side="left")
                    i1 = np.searchsorted(wavelengths, lam_max, side="right")

                    # safety (important)
                    i0 = max(i0, 0)
                    i1 = min(i1, N)

                    if i1 - i0 < 2:
                        # not enough points to integrate
                        out[:, b] = arr2d[:, i0]
                        continue

                    wl_slice = wavelengths[i0:i1]
                    arr_slice = arr2d[:, i0:i1]  # (K, nb)

                    # trapezoidal integration
                    integral = np.trapz(arr_slice, wl_slice, axis=1)
                    width = wl_slice[-1] - wl_slice[0]

                    out[:, b] = integral / width

                band_means[name] = out.reshape(*leading_shape, -1)

            return band_means

        if self.config.SPECTRAL.BAND_METHOD == "snicar-default":

            # average solar variables
            solar_flat_means = compute_flat_band_mean(
                self.solar_irradiance,
                self.config._wavelengths_solar,
                self._band_ranges,
                var_names=["flx_slr"],
            )
            self.solar_irradiance.flx_slr = solar_flat_means["flx_slr"]

            # average atmosphere variables
            atmosphere_flat_means = compute_flat_band_mean(
                self.atmosphere_column,
                self.config._wavelengths_atmosphere,
                self._band_ranges,
                var_names=["tau", "ss_alb", "legendre_moments"],
            )
            self.atmosphere_column.tau = atmosphere_flat_means["tau"]
            self.atmosphere_column.ss_alb = atmosphere_flat_means["ss_alb"]
            self.atmosphere_column.legendre_moments = atmosphere_flat_means[
                "legendre_moments"
            ]

            # average land variables
            land_flat_means = compute_flat_band_mean(
                self.land_column,
                self.config._wavelengths_land,
                self._band_ranges,
                var_names=["tau", "ss_alb", "legendre_moments"],
            )
            self.land_column.tau = land_flat_means["tau"]
            self.land_column.ss_alb = land_flat_means["ss_alb"]
            self.land_column.legendre_moments = land_flat_means["legendre_moments"]

        elif self.config.SPECTRAL.BAND_METHOD == "chandrasekhar-mean":
            return
