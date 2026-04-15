"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import xarray as xr

from ..components.atmosphere import AtmosphereColumn
from ..components.land import LandColumn
from ..components.solar import SolarIrradiance
from ..solvers.multi_stream_solver_ada import solve_multi_stream_rt_ada
from ..solvers.multi_stream_solver_disort import (
    solve_multi_stream_rt_disort,
    solve_multi_stream_rt_disort_wrapper,
)
from ..solvers.two_stream_solver_ad import solve_two_stream_rt_ad
from .config import Config


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

        if "band-" in self.config.SPECTRAL.MODE:
            self.compute_band_average()

        # save outputs
        self.outputs = None

    def _set_spectral_array(self):

        # if the wavelength array is passed by user
        if isinstance(self.config.SPECTRAL.RESOLUTION, tuple):

            # create a homogeneous array based on user input
            # wavelength_homogeneous = np.arange(
            #     *self.config.SPECTRAL.RESOLUTION)
            wavelength_homogeneous = np.concatenate(
                [
                    np.arange(*self.config.SPECTRAL.RESOLUTION),
                    [self.config.SPECTRAL.RESOLUTION[-2]],
                ]
            )
            center_wavelength_homogeneous = (
                wavelength_homogeneous[:-1] + self.config.SPECTRAL.RESOLUTION[-1] / 2
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

            elif "band-" in self.config.SPECTRAL.MODE:
                self.config._wavelengths_solar = wavelength_1cm_m1
                self.config._wavelengths_land = wavelength_1cm_m1
                self.config._wavelengths_atmosphere = wavelength_1cm_m1

                self._band_ranges = band_ranges_homogeneous

                if self.config.SPECTRAL.MODE == "band-snicar-default":
                    self.config._wavelengths_land = center_wavelength_homogeneous

                elif self.config.SPECTRAL.MODE == "band-solar-weighted-mean":
                    # just compute a flat SRF of a given length (which
                    # is arbitrary since it's anyway flat)
                    srf_length = 10
                    self._spectral_response_function = np.vstack(
                        [
                            np.interp(
                                self.config._wavelengths_land,
                                np.linspace(
                                    self._band_ranges[band_number, 0],
                                    self._band_ranges[band_number, 1],
                                    srf_length,
                                ),
                                np.ones(srf_length),
                                left=0.0,
                                right=0.0,
                            )
                            for band_number in range(self._band_ranges.shape[0])
                        ]
                    )

        elif isinstance(self.config.SPECTRAL.RESOLUTION, str):

            srf_base_path = (
                f"{self.config._ROOT_PATH}/data/satellite_spectral_responses"
            )
            if self.config.SPECTRAL.RESOLUTION == "SENTINEL-3-OLCI":
                srf_file_path = f"{srf_base_path}/S3A_OL_SRF_20160713_mean_rsr.nc4"

            elif self.config.SPECTRAL.RESOLUTION == "PRISMA-HYC":
                srf_file_path = f"{srf_base_path}/PRISMA_HYC_SRF.nc4"

            elif self.config.SPECTRAL.RESOLUTION == "ENVISAT-MERIS":
                srf_file_path = f"{srf_base_path}/ENVISAT_MERIS_SRF.nc4"

            ds = xr.open_dataset(srf_file_path)

            self._wavelengths_srf = ds.mean_spectral_response_function_wavelength.values

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

            if self.config.SPECTRAL.MODE == "band-snicar-default":
                self.config._wavelengths_land = ds.nominal_centre_wavelength.values

            # interpolate satellite SRF on homogeneous grid
            # (self.config._wavelengths_land could be any
            # self.config._wavelengths_ since they are the same in
            # band_method srf-integration)
            self._spectral_response_function = np.vstack(
                [
                    np.interp(
                        self.config._wavelengths_land,
                        self._wavelengths_srf[band_number, :],
                        ds.mean_spectral_response_function.values[band_number, :],
                        left=0.0,
                        right=0.0,
                    )
                    for band_number in range(self._band_ranges.shape[0])
                ]
            )

    def _prepare_updates(self, kwargs, allowed_fields):
        forbidden = set(kwargs) - allowed_fields
        if forbidden:
            raise ValueError(
                f"Cannot update: {', '.join(forbidden)}. "
                "These fields cannot be changed at runtime. "
                "Please modify them in the input YAML file."
            )

        # specific situation for LAPs because of nested structure
        if "LIGHT_ABSORBING_PARTICLES" in set(kwargs):
            # check that no additional LAP or change of file occurred
            current_laps = self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.keys()
            current_laps_files = [
                p.FILE for p in self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.values()
            ]
            if any(
                key not in current_laps
                for key in kwargs["LIGHT_ABSORBING_PARTICLES"].keys()
            ) or any(
                p["FILE"] not in current_laps_files
                for p in kwargs["LIGHT_ABSORBING_PARTICLES"].values()
            ):
                raise ValueError(
                    "LAP number and files cannot be changed at runtime. "
                    "Please modify them in the input file. "
                )

    def _write_current_state(self):
        """Write current session state to dictionnary, to be addded to
        the output file"""

        current_state = self.config.model_dump()

        return current_state

    def update_solver(self, updates, validate=True):
        """
        Update allowed solver fields from user-defined dictionary.

        All fields allowed except legendre moments, atmosphere
        coupling and the number of streams as this would require to
        recalculate all optical properties at the moment, especially
        for the atmosphere.

        """

        # validate by creating a new instance of Solver
        if validate:
            allowed_fields = {
                "TYPE",
                "OUTPUT_LEVELS",
                "N_FOURIER_MODES",
                "AZIMUTH_ANGLES",
                "POLAR_ANGLES",
            }
            self._prepare_updates(updates, allowed_fields)
            updated_config = Config.model_validate(
                {
                    **self.config.model_dump(),
                    "SOLVER": {
                        **self.config.SOLVER.model_dump(),
                        **updates,
                    },
                }
            )
            self.config.SOLVER = updated_config.SOLVER

        # update solver parameters only if updates not empty
        # explicit conditions for all keys in case they require
        # further processing
        if updates:

            if "TYPE" in updates:
                self.config.SOLVER.TYPE = updates["TYPE"]

            if "OUTPUT_LEVELS" in updates:
                self.config.SOLVER.OUTPUT_LEVELS = updates["OUTPUT_LEVELS"]

            if "N_STREAMS" in updates:
                self.config.SOLVER.N_STREAMS = updates["N_STREAMS"]

            if "N_FOURIER_MODES" in updates:
                self.config.SOLVER.N_FOURIER_MODES = updates["N_FOURIER_MODES"]

            if "AZIMUTH_ANGLES" in updates:
                self.config.SOLVER.AZIMUTH_ANGLES = updates["AZIMUTH_ANGLES"]

            if "POLAR_ANGLES" in updates:
                self.config.SOLVER.POLAR_ANGLES = updates["POLAR_ANGLES"]

    def update_solar(self, updates, validate=True):
        """
        Update allowed solar fields from user-defined dictionary.

        All fields, hence SZA + SAA.
        """

        # validate by creating a new instance of Solver
        if validate:
            allowed_fields = {"SZA", "SAA"}

            if (
                not self.config.SOLVER.ATMOSPHERE_COUPLING
                and self.config.SPECTRAL.MODE == "band-solar-weighted-mean"
            ):
                raise ValueError(
                    "Updating SZA without atmosphere coupling using band-solar-weighted-mean spectral mode requires a new input file."
                )

            self._prepare_updates(updates, allowed_fields)
            updated_config = Config.model_validate(
                {
                    **self.config.model_dump(),
                    "SOLAR": {
                        **self.config.SOLAR.model_dump(),
                        **updates,
                    },
                }
            )
            self.config.SOLAR = updated_config.SOLAR

        # update SZA and recompute irradiance only if updates not empty
        if updates:
            if "SZA" in updates:
                self.solar_irradiance.sza = updates["SZA"]

            if "SAA" in updates:
                self.solar_irradiance.saa = updates["SAA"]

            if not self.config.SOLVER.ATMOSPHERE_COUPLING:

                # surface irradiance needs to be re-calculated if SZA updated
                self.solar_irradiance.set_surface_irradiance(self.config)

                self.compute_band_average(components=["solar"])

    def update_atmosphere(self, updates, validate=True):
        """
        Update allowed atmospheric fields from user-defined dictionary.
        """

        # for now we do not change sky conditions, atmospheric profile type
        # & aerosol properties

        # validate a copy of the config if requested
        if validate:

            allowed_fields = {
                "INTEGRATED_AOD_550",
                "INTEGRATED_GAS_CONCENTRATIONS",
            }

            self._prepare_updates(updates, allowed_fields)
            updated_config = Config.model_validate(
                {
                    **self.config.model_dump(),
                    "ATMOSPHERE": {
                        **self.config.ATMOSPHERE.model_dump(),
                        **updates,
                    },
                }
            )
            self.config.ATMOSPHERE = updated_config.ATMOSPHERE

        # update only if not empty
        if updates:

            if "INTEGRATED_AOD_550" in updates:
                # load aerosols properties if not in session already
                if self.atmosphere_column.AOD == 0:
                    self.atmosphere_column.set_aerosol_properties()

                # update AOD (final OP calculations after gas update)
                self.atmosphere_column.AOD = updates["INTEGRATED_AOD_550"]

                # only scale if there are aerosols
                if self.atmosphere_column.AOD > 0:
                    self.atmosphere_column.scale_tau_aerosols()

            # if any gas to update, re-compute gas optical thickness
            if "INTEGRATED_GAS_CONCENTRATIONS" in updates:
                self.atmosphere_column.integrated_gas_concentrations = updates[
                    "INTEGRATED_GAS_CONCENTRATIONS"
                ]
                self.atmosphere_column.scale_atmospheric_profile()
                self.atmosphere_column.compute_gas_optical_thickness()

            # if aerosols, re-compute aerosol AND atmosphere optics
            if self.atmosphere_column.AOD > 0:
                self.atmosphere_column.set_atmospheric_properties_with_aerosols()

            # if no aerosols, re-compute atmosphere optics w/out aerosols
            else:
                self.atmosphere_column.set_atmospheric_properties_without_aerosols()

            # recalculate legendre moments
            self.atmosphere_column.set_legendre_moments()
            self.compute_band_average(components=["atmosphere"])

    def update_land(self, updates, validate=True):
        """
        Update allowed solar fields.
        """

        # validate a copy of the config if requested
        if validate:

            allowed_fields = {
                "LAYER_TYPE",
                "GRAIN_SHAPE",
                "RF_TYPE",
                "LWC",
                "THICKNESS",
                "SPECIFIC_SURFACE_AREA",
                "DENSITY",
                "LIGHT_ABSORBING_PARTICLES",
            }

            self._prepare_updates(updates, allowed_fields)
            updated_config = Config.model_validate(
                {
                    **self.config.model_dump(),
                    "LAND": {
                        **self.config.LAND.model_dump(),
                        **updates,
                    },
                }
            )
            self.config.LAND = updated_config.LAND

        # update only if not empty
        if updates:

            # update arguments of the land column class
            if "THICKNESS" in updates:
                self.land_column.thickness_profile = updates["THICKNESS"]
            if "LAYER_TYPE" in updates:
                self.land_column.layer_type = updates["LAYER_TYPE"]
            if "DENSITY" in updates:
                self.land_column.density = updates["DENSITY"]
            if "SPECIFIC_SURFACE_AREA" in updates:
                self.land_column.ssa = updates["SPECIFIC_SURFACE_AREA"]
            if "LWC" in updates:
                self.land_column.lwc = updates["LWC"]
            if "GRAIN_SHAPE" in updates:
                self.land_column.grain_shape = updates["GRAIN_SHAPE"]
            if "RF_TYPE" in updates:
                self.land_column.rf_type = updates["RF_TYPE"]
                self.land_column.set_refractive_index()
                if self.config.SOLVER.TYPE == "two-stream-ad":
                    self.land_column.set_diffuse_fresnel_coeffs()

            if "LIGHT_ABSORBING_PARTICLES" in updates:
                self.land_column.lap_concentrations = (
                    np.array(
                        [
                            obj["CONC"]
                            for obj in updates["LIGHT_ABSORBING_PARTICLES"].values()
                        ]
                    )
                    * 1e-9
                ).T

            # all allowed parameters require to re-calculate clean column ops
            self.land_column.set_column_ops_without_laps()

            # if there are particles we need to update the properties even if
            # we do not change the particle concentrations as the clean snow/ice
            # has changed
            if self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.keys():
                self.land_column.update_column_ops_with_laps()

            # finally update legendre moments
            self.land_column.set_legendre_moments()

            # and recompute band-averaged properties
            self.compute_band_average(components=["land"])

    def run(self, to_xarray=True):
        """
        Run the radiative transfer solver and return outputs
        """

        if self.config.SOLVER.TYPE == "two-stream-ad":
            self.outputs = solve_two_stream_rt_ad(
                self.land_column, self.solar_irradiance
            )
            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_twostream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream-ada":

            self.outputs = solve_multi_stream_rt_ada(
                self.land_column,
                self.atmosphere_column,
                self.solar_irradiance,
                self.config.SOLVER,
            )

            if self.config.SPECTRAL.MODE == "band-srf-integration":
                self.apply_spectral_response_function()

            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_multistream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream-disort":

            if self.config.SOLVER.DELTA_SCALING == "M+":
                self.outputs = solve_multi_stream_rt_disort(
                    self.land_column,
                    self.atmosphere_column,
                    self.solar_irradiance,
                    self.config.SOLVER,
                )
            elif self.config.SOLVER.DELTA_SCALING == "M":
                self.outputs = solve_multi_stream_rt_disort_wrapper(
                    self.land_column,
                    self.atmosphere_column,
                    self.solar_irradiance,
                    self.config.SOLVER,
                )

            if self.config.SPECTRAL.MODE == "band-srf-integration":
                self.apply_spectral_response_function()

            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_multistream_results_to_xarray()

        return self.outputs

    @staticmethod
    def get_package_root() -> pathlib.Path:
        """
        Return the root path of the snicarfx package.
        """
        snicarfx_module = sys.modules["snicarfx"]
        snicarfx_root_path = pathlib.Path(snicarfx_module.__file__).resolve().parents[2]

        return snicarfx_root_path

    def format_twostream_results_to_xarray(self) -> xr.Dataset:
        """
        Save results to an xarray with rather extensive model and
        session state metadata.
        """

        attrs = {
            "model_name": "snicar-fx",
            "model_version": version("snicarfx"),
            "model_url": "https://github.com/openosmia/snicar-fx",
            "creation_date": datetime.now(timezone.utc).isoformat(),
            "session_state": self._write_current_state(),
        }

        ds = xr.Dataset(
            data_vars={
                "albedo_boa": ("wavelength", self.outputs["albedo_boa"]),
                "broadband_albedo_boa": self.outputs["broadband_albedo_boa"],
                "absorbed_flux_fraction": (
                    "layer",
                    self.outputs["absorbed_flux_fraction"],
                ),
                "absorbed_flux_fraction_bottom": self.outputs[
                    "absorbed_flux_fraction_bottom"
                ],
            },
            coords={
                "wavelength": (
                    self._band_ranges[:, -1]
                    if "band-" in self.config.SPECTRAL.MODE
                    else self.config._wavelengths_land
                ),
                "layer": np.arange(len(self.outputs["absorbed_flux_fraction"])),
            },
        )

        # add attributes
        ds.attrs.update(attrs)

        # Add coordinate metadata
        ds["wavelength"].attrs.update(
            {
                "description": "Wavelength",
                "units": "m",
            }
        )
        ds["broadband_albedo_boa"].attrs.update(
            {
                "description": "Broadband albedo (spectrally-integrated albedo) at the surface (Bottom of Atmosphere, BOA)",
                "units": None,
            }
        )
        ds["albedo_boa"].attrs.update(
            {
                "description": "Spectrally resolved surface albedo at the surface (Bottom of Atmosphere, BOA)",
                "units": None,
            }
        )
        ds["absorbed_flux_fraction"].attrs.update(
            {
                "description": "Layer-wise spectrally-resolved absorbed solar flux",
                "units": "W/m2",
            }
        )
        ds["absorbed_flux_fraction_bottom"].attrs.update(
            {
                "description": "Spectrally-resolved absorbed solar energy at the bottom layer",
                "units": "W/m2",
            }
        )

        return ds

    def format_multistream_results_to_xarray(self) -> xr.Dataset:
        """
        Save results to an xarray with rather extensive model and
        session state metadata.
        """

        attrs = {
            "model_name": "snicar-fx",
            "model_version": version("snicarfx"),
            "model_url": "https://github.com/openosmia/snicar-fx",
            "creation_date": datetime.now(timezone.utc).isoformat(),
            "session_state": self._write_current_state(),
        }

        # store albedo, reflectance and radiance variables along with
        # their dimensions
        albedo_variables = {}
        directional_variables = {}
        for var_name, data in self.outputs.items():
            if "albedo_" in var_name:
                albedo_variables[var_name] = ("wavelength", data)

            elif "bba" in var_name:
                albedo_variables[var_name] = data

            elif "directional_" in var_name:
                if "m0" in var_name:
                    directional_variables[var_name] = (
                        ("polar_angle", "wavelength"),
                        data,
                    )
                else:
                    directional_variables[var_name] = (
                        ("polar_angle", "wavelength", "azimuth_angle"),
                        data,
                    )

        # create xarray dataset
        if self.config.SOLVER.N_FOURIER_MODES == 1:
            ds = xr.Dataset(
                data_vars={**albedo_variables, **directional_variables},
                coords={
                    "wavelength": (
                        self._band_ranges[:, -1]
                        if "band-" in self.config.SPECTRAL.MODE
                        else self.config._wavelengths_land
                    ),
                    "polar_angle": self.outputs["polar_angle"],
                },
            )
        else:
            ds = xr.Dataset(
                data_vars={**albedo_variables, **directional_variables},
                coords={
                    "wavelength": (
                        self._band_ranges[:, -1]
                        if "band-" in self.config.SPECTRAL.MODE
                        else self.config._wavelengths_land
                    ),
                    "polar_angle": self.outputs["polar_angle"],
                    "azimuth_angle": self.outputs["azimuth_angle"],
                },
            )

        # add attributes
        ds.attrs.update(attrs)

        # Add coordinate metadata
        ds["wavelength"].attrs.update(
            {
                "description": "Wavelength",
                "units": "nm",
            }
        )
        ds["polar_angle"].attrs.update(
            {
                "description": "Viewing polar angle.",
                "units": "degrees",
            }
        )

        if self.config.SOLVER.N_FOURIER_MODES > 1:
            ds["azimuth_angle"].attrs.update(
                {
                    "description": "Viewing azimuth angle.",
                    "units": "degrees",
                }
            )

        return ds

    def apply_spectral_response_function(self) -> None:

        for key, var in self.outputs.items():

            if any(tag in key for tag in ["albedo_"]):
                self.outputs[key] = np.nansum(
                    self.outputs[key][None, :] * self._spectral_response_function,
                    axis=-1,
                ) / np.nansum(self._spectral_response_function, axis=1)
            elif any(tag in key for tag in ["m0"]):
                self.outputs[key] = (
                    np.nansum(
                        self.outputs[key][:, None, :]
                        * self._spectral_response_function[None, :, :],
                        axis=-1,
                    )
                    / np.nansum(self._spectral_response_function, axis=1)[None, :]
                )
            elif any(tag in key for tag in ["directional_"]):
                self.outputs[key] = (
                    np.nansum(
                        self.outputs[key][:, None, :, :]  # polar, wvl1, wvl2, azim
                        * self._spectral_response_function[
                            None, :, :, None
                        ],  # polar, wvl, wvl, azim
                        axis=-2,
                    )
                    / np.nansum(self._spectral_response_function, axis=1)[
                        None, :, :
                    ]  # polar, wvl, azim
                )

        # overwrite high-resolution wavelength array (used for
        # computation) with center wavelengths
        self.config._wavelengths = self._band_ranges[:, -1]

    def compute_flat_band_average(self, component, wavelengths, band_ranges, var_names):
        """
        Flat (unweighted) band average on spectral variables on
        given variables of a component object.
        """

        band_means = {}

        for name in var_names:

            arr = getattr(component, name)
            original_shape = arr.shape[:-1]
            arr_flat = arr.reshape(-1, arr.shape[-1])

            n_bands = band_ranges.shape[0]
            averaged_rows = np.empty((arr_flat.shape[0], n_bands))

            for b, (lam_min, lam_max, _) in enumerate(band_ranges):
                i_start = np.searchsorted(wavelengths, lam_min, side="left")
                i_end = np.searchsorted(wavelengths, lam_max, side="right")
                i_start = max(i_start, 0)
                i_end = min(i_end, arr_flat.shape[1])

                if i_end - i_start < 2:
                    averaged_rows[:, b] = arr_flat[:, i_start]
                else:
                    wl_slice = wavelengths[i_start:i_end]
                    values_slice = arr_flat[:, i_start:i_end]

                    # trapzal integration along last axis (wavelength)
                    integral = np.trapz(values_slice, wl_slice, axis=1)
                    width = wl_slice[-1] - wl_slice[0]
                    averaged_rows[:, b] = integral / width

            band_means[name] = averaged_rows.reshape(*original_shape, n_bands)

        return band_means

    def compute_solar_weighted_average(
        self, column, wavelengths, band_ranges, var_names
    ):
        """
        Solar weighted average on spectral variables on given
        variables of a component object.
        """
        band_means = {}

        # Precompute denominator integral
        denominator_integral_total = np.trapz(
            self._spectral_response_function_sw_total, x=wavelengths, axis=-1
        )
        
        denominator_integral_diff = np.trapz(
            self._spectral_response_function_sw_diff, x=wavelengths, axis=-1
        )
        
        denominator_integral_dir = np.trapz(
            self._spectral_response_function_sw_dir, x=wavelengths, axis=-1
        )
        
        
        for name in var_names:
            arr = getattr(column, name)

            n_bands = len(band_ranges)
            original_shape = arr.shape[:-1] if arr.ndim > 1 else ()

            # Reshape to (n_flat, n_wl) for broadcasting
            if arr.ndim > 1:
                arr_flat = arr.reshape(-1, arr.shape[-1])
            else:
                arr_flat = arr[None, :]

            # Compute numerator and denominator
            # if name in ["total_irradiance", "direct_beam", "diffuse"]:
            #     numerator = self._spectral_response_function_sw
            #     denominator = self._spectral_response_function
            #     # Recompute denominator integral for these
            #     denominator_integral_local = np.trapz(
            #         denominator, x=wavelengths, axis=-1
            #     )
            if name == "total_irradiance":
                numerator = self._spectral_response_function_sw_total
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapz(
                    denominator, x=wavelengths, axis=-1
                )
            if name == "direct_beam":
                numerator = self._spectral_response_function_sw_dir
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapz(
                    denominator, x=wavelengths, axis=-1
                )
            if name == "diffuse":
                numerator = self._spectral_response_function_sw_diff
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapz(
                    denominator, x=wavelengths, axis=-1
                )
            else:
                # weigh all variables with srf * total flux
                numerator = self._spectral_response_function_sw_total * arr_flat[:, None, :]
                denominator = self._spectral_response_function_sw_total
                denominator_integral_local = denominator_integral_total

            # Integrate along wavelength axis
            numerator_integral = np.trapz(numerator, x=wavelengths, axis=-1)
            averaged_rows = numerator_integral / denominator_integral_local[None, :]

            # Reshape back
            if arr.ndim > 1:
                averaged_rows = averaged_rows.reshape(*original_shape, n_bands)
            else:
                averaged_rows = averaged_rows.flatten()

            band_means[name] = averaged_rows

        return band_means

    def compute_band_average(self, components=["solar", "atmosphere", "land"]) -> None:
        """
        Compute band averages for given properties of given components.
        """

        if self.config.SPECTRAL.MODE == "band-snicar-default":

            # average solar variables
            if "solar" in components:
                solar_flat_means = self.compute_flat_band_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["total_irradiance"],
                )
                self.solar_irradiance.total_irradiance = solar_flat_means["total_irradiance"]

                solar_flat_means = self.compute_flat_band_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["direct_beam", "diffuse"],
                )
                self.solar_irradiance.direct_beam = solar_flat_means["direct_beam"]
                self.solar_irradiance.diffuse = solar_flat_means["diffuse"]

            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING == True:
                if "atmosphere" in components:
                    atmosphere_flat_means = self.compute_flat_band_average(
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

        elif self.config.SPECTRAL.MODE == "band-solar-weighted-mean":

            # only compute if it hasn't been yet
            if not hasattr(self, "_spectral_response_function_sw_total"):
                # cache solar weighted SRF
                self._spectral_response_function_sw_total = (
                    self.solar_irradiance.total_irradiance[None, :]
                    * self._spectral_response_function
                )
                self._spectral_response_function_sw_diff = (
                    self.solar_irradiance.diffuse[None, :]
                    * self._spectral_response_function
                )
                self._spectral_response_function_sw_dir = (
                    self.solar_irradiance.direct_beam[None, :]
                    * self._spectral_response_function
                )

            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING == True:
                if "atmosphere" in components:
                    atmosphere_weighted_means = self.compute_solar_weighted_average(
                        self.atmosphere_column,
                        self.config._wavelengths_atmosphere,
                        self._band_ranges,
                        var_names=["tau", "ss_alb", "legendre_moments"],
                    )
                    self.atmosphere_column.tau = atmosphere_weighted_means["tau"]
                    self.atmosphere_column.ss_alb = atmosphere_weighted_means["ss_alb"]
                    self.atmosphere_column.legendre_moments = atmosphere_weighted_means[
                        "legendre_moments"
                    ]

            if "land" in components:
                if self.config.SOLVER.TYPE == "two-stream-ad":
                    land_weighted_means = self.compute_solar_weighted_average(
                        self.land_column,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=["ref_idx_re", "ref_idx_im", "sfc",
                                   "tau", "ss_alb", 
                                   "asm_prm"],
                    )
                    self.land_column.ref_idx_re = land_weighted_means["ref_idx_re"]
                    self.land_column.ref_idx_im = land_weighted_means["ref_idx_im"]
                    self.land_column.tau = land_weighted_means["tau"]
                    self.land_column.ss_alb = land_weighted_means["ss_alb"]
                    self.land_column.asm_prm = land_weighted_means["asm_prm"]

                    self.land_column.sfc = land_weighted_means["sfc"].flatten()
                    
                else: 
                    # average land variables
                    land_weighted_means = self.compute_solar_weighted_average(
                        self.land_column,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=["tau", "ss_alb", "legendre_moments", "asm_prm"],
                    )
                    self.land_column.tau = land_weighted_means["tau"]
                    self.land_column.ss_alb = land_weighted_means["ss_alb"]
                    self.land_column.legendre_moments = land_weighted_means[
                        "legendre_moments"
                    ]
                    self.land_column.asm_prm = land_weighted_means["asm_prm"]

            if "solar" in components:
                solar_weighted_means = self.compute_solar_weighted_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["direct_beam", "diffuse"],
                )
                self.solar_irradiance.direct_beam = solar_weighted_means["direct_beam"].flatten()
                self.solar_irradiance.diffuse = solar_weighted_means["diffuse"].flatten()
                
                # average solar variables
                solar_weighted_means = self.compute_solar_weighted_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["total_irradiance"],
                )
                self.solar_irradiance.total_irradiance = solar_weighted_means[
                    "total_irradiance"
                ].flatten()
