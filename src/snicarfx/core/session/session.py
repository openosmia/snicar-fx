"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import scipy
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

    Attributes
    ----------
    config : Config
        Direct spectral solar irradiance.
    land : LandColumn
        Instance of the LandColumn class.
    atmosphere : AtmosphereColumn
        Instance of the AtmosphereColumn class.
    solar : SolarIrradiance
        Instance of the SolarIrradiance class.
    outputs : ndarray
        Outputs of the RTE solver.

    """

    def __init__(self, input_file: str):
        # parse configuration file
        self.config = Config.from_yaml(input_file)

        # set module root path for data loading
        self.config._ROOT_PATH = self.get_package_root()

        # set spectral parameters based on user input
        self._set_spectral_array()

        # build components
        self.land = LandColumn(self.config)
        self.solar = SolarIrradiance(self.config)
        self.atmosphere = AtmosphereColumn(self.config)

        if "band-" in self.config.SPECTRAL.MODE:
            self.compute_band_average()

        # save outputs
        self.outputs = None

    def _set_spectral_array(self):
        """
        Set spectral arrays before solving the RTE, depending on the spectral
        mode and resolution.
        """

        # if the wavelength array is passed by user
        if isinstance(self.config.SPECTRAL.RESOLUTION, tuple):
            if self.config.SPECTRAL.RESOLUTION[2] == "1cm-1":
                # create an array at 1cm-1 resolution
                wavelength_homogeneous = (
                    1e7
                    / np.arange(
                        1 / (self.config.SPECTRAL.RESOLUTION[1] * 1e-7),
                        1 / (self.config.SPECTRAL.RESOLUTION[0] * 1e-7),
                        1,
                    )[::-1]
                )

            else:
                # create an array with nm step
                wavelength_homogeneous = np.arange(
                    self.config.SPECTRAL.RESOLUTION[0],
                    self.config.SPECTRAL.RESOLUTION[1]
                    + self.config.SPECTRAL.RESOLUTION[2],
                    self.config.SPECTRAL.RESOLUTION[2],
                )

            if self.config.SPECTRAL.MODE == "monochromatic":
                self.config._wavelengths_solar = wavelength_homogeneous
                self.config._wavelengths_land = wavelength_homogeneous
                self.config._wavelengths_atmosphere = wavelength_homogeneous
                self.config._wavelengths = wavelength_homogeneous

            elif "band-" in self.config.SPECTRAL.MODE:
                # set at 1cm-1 resolution by default to compute OPs
                wavelength_1cm_m1 = (
                    1e7
                    / np.arange(
                        1 / (self.config.SPECTRAL.RESOLUTION[1] * 1e-7),
                        1 / (self.config.SPECTRAL.RESOLUTION[0] * 1e-7),
                        1,
                    )[::-1]
                )

                self.config._wavelengths_solar = wavelength_1cm_m1
                self.config._wavelengths_land = wavelength_1cm_m1
                self.config._wavelengths_atmosphere = wavelength_1cm_m1

                center_wavelength_homogeneous = (
                    wavelength_homogeneous[:-1]
                    + self.config.SPECTRAL.RESOLUTION[-1] / 2
                )

                band_ranges_homogeneous = np.column_stack(
                    (
                        wavelength_homogeneous[:-1],
                        wavelength_homogeneous[1:],
                        center_wavelength_homogeneous,
                    )
                )

                self._band_ranges = band_ranges_homogeneous
                self.config._wavelengths = center_wavelength_homogeneous
                
                if self.config.SPECTRAL.MODE == "band-snicar-default":
                    self.config._wavelengths_land = center_wavelength_homogeneous

        elif isinstance(self.config.SPECTRAL.RESOLUTION, str):
            srf_base_path = f"{self.config._ROOT_PATH}/data/satellite_SRFs"
            if self.config.SPECTRAL.RESOLUTION == "SENTINEL-3-OLCI":
                srf_file_path = f"{srf_base_path}/S3A_OL_SRF_20160713_mean_rsr.nc4"

            elif self.config.SPECTRAL.RESOLUTION == "PRISMA-HYC":
                srf_file_path = f"{srf_base_path}/PRISMA_HYC_SRF.nc4"

            elif self.config.SPECTRAL.RESOLUTION == "ENVISAT-MERIS":
                srf_file_path = f"{srf_base_path}/ENVISAT_MERIS_SRF.nc4"

            ds = xr.open_dataset(srf_file_path)

            self._wavelengths_srf = ds.mean_spectral_response_function_wavelength.values

            # create a global wavelength array
            mins_per_band_wavelength = np.nanmin(self._wavelengths_srf, axis=1)
            maxs_per_band_wavelength = np.nanmax(self._wavelengths_srf, axis=1)

            min_wavelength = np.nanmin(self._wavelengths_srf)
            max_wavelength = np.nanmax(self._wavelengths_srf)

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

            if self.config.SPECTRAL.MODE in ["monochromatic", "band-srf-integration"]:
                self.config._wavelengths = restricted_wavelength_1cm_m1[mask]
            else:
                self.config._wavelengths = ds.srf_centre_wavelength.values
             
            self.config._wavelengths_solar = restricted_wavelength_1cm_m1[mask]
            self.config._wavelengths_land = restricted_wavelength_1cm_m1[mask]
            self.config._wavelengths_atmosphere = restricted_wavelength_1cm_m1[mask]

            self._band_ranges = np.column_stack(
                (
                    mins_per_band_wavelength,
                    maxs_per_band_wavelength,
                    ds.srf_centre_wavelength.values,
                )
            )

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
        """
        Verify that fields to update are allowed to be updated.

        Parameters
        ----------
        kwargs : dict
            Fields to update.
        allowed_fields : dict
            Fields allowed to be updated.
        """

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
                key not in current_laps for key in kwargs["LIGHT_ABSORBING_PARTICLES"]
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

        Parameters
        ----------
        updates : dict
            Fields to update with corresponding new values.
        validate : bool
            Validate update with Pydantic or not.
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

        Parameters
        ----------
        updates : dict
            Fields to update with corresponding new values.
        validate : bool
            Validate update with Pydantic or not.
        """

        # validate by creating a new instance of Solver
        if validate:
            allowed_fields = {"SZA", "SAA"}

            if (
                not self.config.SOLVER.ATMOSPHERE_COUPLING
                and self.config.SPECTRAL.MODE == "band-srf-solar-weighted-mean"
            ):
                raise ValueError(
                    "Updating SZA without atmosphere coupling using "
                    "band-srf-solar-weighted-mean spectral mode requires "
                    "a new input file."
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
                self.solar.sza = updates["SZA"]

            if "SAA" in updates:
                self.solar.saa = updates["SAA"]

            if not self.config.SOLVER.ATMOSPHERE_COUPLING:
                # surface irradiance needs to be re-calculated if SZA updated
                self.solar.set_surface_irradiance()

                self.compute_band_average(components=["solar"])

    def update_atmosphere(self, updates, validate=True):
        """
        Update allowed atmospheric fields from user-defined dictionary.

        Parameters
        ----------
        updates : dict
            Fields to update with corresponding new values.
        validate : bool
            Validate update with Pydantic or not.
        """

        # raise error if update atmopshere is called but not
        # atmosphere is set
        if not self.config.SOLVER.ATMOSPHERE_COUPLING:
            raise ValueError(
                "Cannot update atmosphere properties when SOLVER.ATMOSPHERE_COUPLING"
                " is false. Please set SOLVER.ATMOSPHERE_COUPLING to true in input "
                "file before attempting to update atmosphere."
            )

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
                if self.atmosphere.AOD == 0:
                    self.atmosphere.set_aerosol_properties()

                # update AOD (final OP calculations after gas update)
                self.atmosphere.AOD = updates["INTEGRATED_AOD_550"]

                # only scale if there are aerosols
                if self.atmosphere.AOD > 0:
                    self.atmosphere.scale_tau_aerosols()

            # if any gas to update, re-compute gas optical thickness
            if "INTEGRATED_GAS_CONCENTRATIONS" in updates:
                self.atmosphere.integrated_gas_concentrations = updates[
                    "INTEGRATED_GAS_CONCENTRATIONS"
                ]
                self.atmosphere.scale_atmospheric_profile()
                self.atmosphere.compute_gas_optical_thickness()

            # if aerosols, re-compute aerosol AND atmosphere optics
            if self.atmosphere.AOD > 0:
                self.atmosphere.set_atmospheric_properties_with_aerosols()

            # if no aerosols, re-compute atmosphere optics w/out aerosols
            else:
                self.atmosphere.set_atmospheric_properties_without_aerosols()

            # recalculate legendre moments
            self.atmosphere.set_legendre_moments()
            self.compute_band_average(components=["atmosphere"])

    def update_land(self, updates, validate=True):
        """
        Update allowed land fields from user-defined dictionary.

        Parameters
        ----------
        updates : dict
            Fields to update with corresponding new values.
        validate : bool
            Validate update with Pydantic or not.
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
                self.land.thickness_profile = updates["THICKNESS"]
            if "LAYER_TYPE" in updates:
                self.land.layer_type = updates["LAYER_TYPE"]
            if "DENSITY" in updates:
                self.land.density = updates["DENSITY"]
            if "SPECIFIC_SURFACE_AREA" in updates:
                self.land.ssa = updates["SPECIFIC_SURFACE_AREA"]
            if "LWC" in updates:
                self.land.lwc = updates["LWC"]
            if "GRAIN_SHAPE" in updates:
                self.land.grain_shape = updates["GRAIN_SHAPE"]
            if "RF_TYPE" in updates:
                self.land.rf_type = updates["RF_TYPE"]
                self.land.set_refractive_index()
                if self.config.SOLVER.TYPE == "two-stream-ad":
                    self.land.set_diffuse_fresnel_coeffs()

            if "LIGHT_ABSORBING_PARTICLES" in updates:
                self.land.lap_concentrations = (
                    np.array(
                        [
                            obj["CONC"]
                            for obj in updates["LIGHT_ABSORBING_PARTICLES"].values()
                        ]
                    )
                    * 1e-9
                ).T

            # all allowed parameters require to re-calculate clean column ops
            self.land.set_column_ops_without_laps()

            # if there are particles we need to update the properties even if
            # we do not change the particle concentrations as the clean snow/ice
            # has changed
            if self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.keys():
                self.land.update_column_ops_with_laps()

            # finally update legendre moments if necessary
            if self.land.n_expansion is not None:
                self.land.set_legendre_moments()

            # and recompute band-averaged properties
            self.compute_band_average(components=["land"])

    def run(self, to_xarray=True):
        """
        Run the radiative transfer solver and return outputs.

        Parameters
        ----------
        to_xarray : bool
            Return outputs as xarray or not (else dictionary).
        """

        if self.config.SOLVER.TYPE == "two-stream-ad":
            self.outputs = solve_two_stream_rt_ad(self.land, self.solar)
            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_twostream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream-ada":
            self.outputs = solve_multi_stream_rt_ada(
                self.land,
                self.atmosphere,
                self.solar,
                self.config,
            )

            if self.config.SPECTRAL.MODE == "band-srf-integration":
                self.apply_spectral_response_function()

            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_multistream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream-disort":
            if self.config.SOLVER.DELTA_SCALING == "M+":
                self.outputs = solve_multi_stream_rt_disort(
                    self.land,
                    self.atmosphere,
                    self.solar,
                    self.config,
                )
            elif self.config.SOLVER.DELTA_SCALING == "M":
                self.outputs = solve_multi_stream_rt_disort_wrapper(
                    self.land,
                    self.atmosphere,
                    self.solar,
                    self.config,
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
        Save results of two-stream simulation to an xarray with model
        and session state metadata.
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
                "bba_boa": self.outputs["bba_boa"],
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
        ds["bba_boa"].attrs.update(
            {
                "description": (
                    "Broadband albedo (spectrally-integrated albedo) "
                    "at the surface (Bottom of Atmosphere, BOA)"
                ),
                "units": None,
            }
        )
        ds["albedo_boa"].attrs.update(
            {
                "description": (
                    "Spectrally resolved surface albedo at the surface "
                    "(Bottom of Atmosphere, BOA)"
                ),
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
                "description": (
                    "Spectrally-resolved absorbed solar energy at the bottom layer"
                ),
                "units": "W/m2",
            }
        )

        return ds

    def format_multistream_results_to_xarray(self) -> xr.Dataset:
        """
        Save results of multi-stream simulation to an xarray with model
        and session state metadata.
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
                        self.config._wavelengths
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
        """
        Apply spectral response function of user-defined satellite instrument
        to outputs of radiative transfer solver in order to return satellite
        bands.
        """

        for key, var in self.outputs.items():
            if any(tag in key for tag in ["albedo_"]):
                self.outputs[key] = np.trapezoid(
                    var[None, :] * self._spectral_response_function,
                    x=self.config._wavelengths_solar,
                    axis=-1,
                ) / np.trapezoid(
                    self._spectral_response_function,
                    x=self.config._wavelengths_solar,
                    axis=-1,
                )

            elif any(tag in key for tag in ["m0"]):
                # (polar, None, wl) * (None, bands, wl)
                self.outputs[key] = np.trapezoid(
                    var[:, None, :] * self._spectral_response_function[None, :, :],
                    x=self.config._wavelengths_solar,
                    axis=-1,
                ) / np.trapezoid(
                    self._spectral_response_function,
                    x=self.config._wavelengths_solar,
                    axis=-1,
                )

            elif any(tag in key for tag in ["directional_"]):
                self.outputs[key] = (
                    np.trapezoid(
                        # (polar, 1, wl, azimuth)
                        var[:, None, :, :]
                        # (1, 21, wl, 1)
                        * self._spectral_response_function[None, :, :, None],
                        x=self.config._wavelengths_solar,
                        axis=-2,
                    )
                    / np.trapezoid(
                        self._spectral_response_function,
                        x=self.config._wavelengths_solar,
                        axis=-1,
                    )[None, :, None]
                )

    def compute_flat_band_average(self, component, wavelengths, band_ranges, var_names):
        """
        Compute band averages of each spectral variable along the wavelength axis.
        Since the wavelength grid is not necessarily homogeneous, the average
        translates into an integration divided by the spectral range.

        Parameters
        ----------
        component : LandColumn, AtmosphereColumn or SolarIrradiance
            Class of the component to update.
        wavelengths : ndarray
            Wavelength array used in the simulation.
        band_ranges : ndarray
            Bounds and center wavelengths of bands.
        var_names : list
            Name of variables to update to bands.
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

                    # trapezoidal integration along last axis (wavelength)
                    integral = np.trapezoid(values_slice, wl_slice, axis=1)
                    width = wl_slice[-1] - wl_slice[0]
                    averaged_rows[:, b] = integral / width

            band_means[name] = averaged_rows.reshape(*original_shape, n_bands)

        return band_means

    def compute_srf_weighted_average(self, column, wavelengths, band_ranges, var_names):
        """
        Compute bands for each spectral variable by integrating over the
        satellite spectral response function.

        Parameters
        ----------
        column : LandColumn, AtmosphereColumn or SolarIrradiance
            Class of the component to update.
        wavelengths : ndarray
            Wavelength array used in the simulation.
        band_ranges : ndarray
            Bounds and center wavelengths of bands.
        var_names : list
            Name of variables to update to bands.
        """

        band_means = {}

        # Precompute denominator integral
        denominator_integral_srf = np.trapezoid(
            self._spectral_response_function, x=wavelengths, axis=-1
        )

        for name in var_names:
            arr = getattr(column, name)

            n_bands = len(band_ranges)
            original_shape = arr.shape[:-1] if arr.ndim > 1 else ()

            # Reshape to (n_flat, n_wl) for broadcasting
            arr_flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None, :]

            if name == "total_irradiance":
                # shape (1, wl) * (21, wl) integ on wl
                numerator = (
                    self.solar.total_irradiance[None, :]
                    * self._spectral_response_function
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = denominator_integral_srf

            elif name == "direct_beam":
                numerator = (
                    self.solar.direct_beam[None, :] * self._spectral_response_function
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = denominator_integral_srf

            elif name == "diffuse":
                numerator = (
                    self.solar.diffuse[None, :] * self._spectral_response_function
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = denominator_integral_srf

            elif name == "tau":
                numerator = self._spectral_response_function * arr_flat[:, None, :]
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = denominator_integral_srf[None, :]

            elif name == "ss_alb":
                # srf * flux * w * tau
                # shape  (21, wl) * (lyr, 1, wl) * (lyr, 1, wl) integ on wl
                numerator = (
                    self._spectral_response_function
                    * arr_flat[:, None, :]
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = np.trapezoid(
                    self._spectral_response_function
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :],
                    x=wavelengths,
                    axis=-1,
                )

            elif name == "asm_prm":
                # srf * flux * g * w * tau
                # shape (21, wl) * (lyr, 1, wl) * (lyr, 1, wl) * (lyr,
                # 1, wl) integ on wl
                numerator = (
                    self._spectral_response_function
                    * arr_flat[:, None, :]
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                    * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :]
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = np.trapezoid(
                    self._spectral_response_function
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                    * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :],
                    x=wavelengths,
                    axis=-1,
                )

            # leg moments are taken at central wl to reduce comp. burden
            elif name == "legendre_moments":
                f = scipy.interpolate.interp1d(
                    self.config._wavelengths_atmosphere,
                    arr_flat,
                    axis=1,
                )
                arr_flat_interp = f(self.config._wavelengths)

                numerator_integral = arr_flat_interp
                denominator_integral_local = np.ones_like(numerator_integral)

                # proper formula to mix leg coeffs
                # shape  (21, wl) * (1, lyr, 1, wl) * (1, lyr, 1, wl) integ on wl
                # numerator = (
                #     self._spectral_response_function
                #     * column.tau.reshape(-1, arr.shape[-1])[None, :, None, :]
                #     * column.ss_alb.reshape(-1, arr.shape[-1])[None, :, None, :]
                #     * arr[:, :, None, :]
                #     )
                # numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                # denominator_integral_local = np.trapezoid(
                #     self._spectral_response_function
                #     * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                #     * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :],
                #     x=wavelengths,
                #     axis=-1,
                # )

            elif name in ["ref_idx_re", "ref_idx_im", "sfc"]:
                # weigh with srf * total flux
                numerator = self._spectral_response_function * arr_flat[:, None, :]
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = denominator_integral_srf[None, :]

            averaged_rows = numerator_integral / denominator_integral_local

            # Reshape back
            if arr.ndim > 1:
                averaged_rows = averaged_rows.reshape(*original_shape, n_bands)
            else:
                averaged_rows = averaged_rows.flatten()

            band_means[name] = averaged_rows

        return band_means

    def compute_srf_solar_weighted_average(
        self, column, wavelengths, band_ranges, var_names
    ):
        """
        Compute bands for each spectral variable by integrating over the
        satellite spectral response function multiplied by the solar irradiance.

        Parameters
        ----------
        column : LandColumn, AtmosphereColumn or SolarIrradiance
            Class of the component to update.
        wavelengths : ndarray
            Wavelength array used in the simulation.
        band_ranges : ndarray
            Bounds and center wavelengths of bands.
        var_names : list
            Name of variables to update to bands.
        """

        band_means = {}

        # Precompute denominator integral
        denominator_integral_total = np.trapezoid(
            self._spectral_response_function_sw_total, x=wavelengths, axis=-1
        )

        for name in var_names:

            arr = getattr(column, name)

            n_bands = len(band_ranges)
            original_shape = arr.shape[:-1] if arr.ndim > 1 else ()

            # Reshape to (n_flat, n_wl) for broadcasting
            arr_flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None, :]

            if name == "total_irradiance":
                numerator = self._spectral_response_function_sw_total
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapezoid(
                    denominator, x=wavelengths, axis=-1
                )

            elif name == "direct_beam":
                numerator = self._spectral_response_function_sw_dir
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapezoid(
                    denominator, x=wavelengths, axis=-1
                )

            elif name == "diffuse":
                numerator = self._spectral_response_function_sw_diff
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator = self._spectral_response_function
                # Recompute denominator integral for these
                denominator_integral_local = np.trapezoid(
                    denominator, x=wavelengths, axis=-1
                )

            elif name == "ss_alb":
                # srf * flux * w * tau
                numerator = (
                    self._spectral_response_function_sw_total
                    * arr_flat[:, None, :]
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = np.trapezoid(
                    self._spectral_response_function_sw_total
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :],
                    x=wavelengths,
                    axis=-1,
                )

            elif name == "asm_prm":
                # srf * flux * g * w * tau
                # shape (21, wl) * (lyr, 1, wl) * (lyr, 1, wl) * (lyr,
                # 1, wl) integ on wl
                numerator = (
                    self._spectral_response_function_sw_total
                    * arr_flat[:, None, :]
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                    * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :]
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator_integral_local = np.trapezoid(
                    self._spectral_response_function_sw_total
                    * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                    * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :],
                    x=wavelengths,
                    axis=-1,
                )[None, :]

            # leg moments are taken at central wl to reduce comp. burden
            elif name == "legendre_moments":
                f = scipy.interpolate.interp1d(
                    self.config._wavelengths_atmosphere,
                    arr_flat,
                    axis=1,
                )
                arr_flat_interp = f(self.config._wavelengths)

                numerator_integral = arr_flat_interp
                denominator_integral_local = np.ones_like(numerator_integral)

                # numerator = (
                #     self._spectral_response_function_sw_total
                #     * column.tau.reshape(-1, arr.shape[-1])[None, :, None, :]
                #     * column.ss_alb.reshape(-1, arr.shape[-1])[None, :, None, :]
                #     * arr[:, :, None, :]
                #     )
                # numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                # denominator_integral_local = np.trapezoid(
                #     self._spectral_response_function_sw_total
                #     * column.tau.reshape(-1, arr.shape[-1])[:, None, :]
                #     * column.ss_alb.reshape(-1, arr.shape[-1])[:, None, :],
                #     x=wavelengths,
                #     axis=-1,
                # )

            elif name in ["tau", "ref_idx_re", "ref_idx_im", "sfc", "asm_prm"]:
                # weigh all variables with srf * total flux
                numerator = (
                    self._spectral_response_function_sw_total * arr_flat[:, None, :]
                )
                numerator_integral = np.trapezoid(numerator, x=wavelengths, axis=-1)
                denominator = self._spectral_response_function_sw_total
                denominator_integral_local = denominator_integral_total[None, :]

            # Integrate along wavelength axis
            averaged_rows = numerator_integral / denominator_integral_local

            # Reshape back
            if arr.ndim > 1:
                averaged_rows = averaged_rows.reshape(*original_shape, n_bands)
            else:
                averaged_rows = averaged_rows.flatten()

            band_means[name] = averaged_rows

        return band_means

    def compute_band_average(self, components=None) -> None:
        """
        Compute band averages for a given component, depending on the user-defined
        band mode.

        Parameters
        ----------
        components : str or list
            Component(s) to update.
        """

        # if None, default to all possible components
        if components is None:
            components = ["land", "atmosphere", "solar"]
        elif isinstance(components, str):
            components = [components]

        if self.config.SPECTRAL.MODE == "band-snicar-default":
            # average solar variables (no band for land and atmosphere
            # in this mode)
            if "solar" in components:
                solar_flat_means = self.compute_flat_band_average(
                    self.solar,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["total_irradiance", "direct_beam", "diffuse"],
                )
                self.solar.total_irradiance = solar_flat_means["total_irradiance"]
                self.solar.direct_beam = solar_flat_means["direct_beam"]
                self.solar.diffuse = solar_flat_means["diffuse"]

        elif self.config.SPECTRAL.MODE in ["band-srf-weighted-mean"]:
            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING and "atmosphere" in components:
                atmosphere_weighted_means = self.compute_srf_weighted_average(
                    self.atmosphere,
                    self.config._wavelengths_atmosphere,
                    self._band_ranges,
                    var_names=["tau", "ss_alb", "legendre_moments"],
                )
                self.atmosphere.tau = atmosphere_weighted_means["tau"]
                self.atmosphere.ss_alb = atmosphere_weighted_means["ss_alb"]
                self.atmosphere.legendre_moments = atmosphere_weighted_means[
                    "legendre_moments"
                ]

            if "land" in components:
                if self.config.SOLVER.TYPE == "two-stream-ad":
                    land_weighted_means = self.compute_srf_weighted_average(
                        self.land,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=[
                            "ref_idx_re",
                            "ref_idx_im",
                            "sfc",
                            "tau",
                            "ss_alb",
                            "asm_prm",
                        ],
                    )
                    self.land.ref_idx_re = land_weighted_means["ref_idx_re"]
                    self.land.ref_idx_im = land_weighted_means["ref_idx_im"]
                    self.land.tau = land_weighted_means["tau"]
                    self.land.ss_alb = land_weighted_means["ss_alb"]
                    self.land.asm_prm = land_weighted_means["asm_prm"]
                    self.land.sfc = land_weighted_means["sfc"].flatten()

                else:
                    # average land variables
                    land_weighted_means = self.compute_srf_weighted_average(
                        self.land,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=[
                            "tau",
                            "ss_alb",
                            "legendre_moments",
                        ],
                    )
                    self.land.tau = land_weighted_means["tau"]
                    self.land.ss_alb = land_weighted_means["ss_alb"]
                    self.land.legendre_moments = land_weighted_means["legendre_moments"]

            if "solar" in components:
                solar_weighted_means = self.compute_srf_weighted_average(
                    self.solar,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["direct_beam", "diffuse", "total_irradiance"],
                )
                self.solar.direct_beam = solar_weighted_means["direct_beam"].flatten()
                self.solar.diffuse = solar_weighted_means["diffuse"].flatten()
                self.solar.total_irradiance = solar_weighted_means[
                    "total_irradiance"
                ].flatten()

        elif self.config.SPECTRAL.MODE == "band-srf-solar-weighted-mean":
            # only compute if it hasn't been yet
            if not hasattr(self, "_spectral_response_function_sw_total"):
                # cache solar weighted SRF
                self._spectral_response_function_sw_total = (
                    self.solar.total_irradiance[None, :]
                    * self._spectral_response_function
                )
                self._spectral_response_function_sw_diff = (
                    self.solar.diffuse[None, :] * self._spectral_response_function
                )
                self._spectral_response_function_sw_dir = (
                    self.solar.direct_beam[None, :] * self._spectral_response_function
                )

            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING and "atmosphere" in components:
                atmosphere_weighted_means = self.compute_srf_solar_weighted_average(
                    self.atmosphere,
                    self.config._wavelengths_atmosphere,
                    self._band_ranges,
                    var_names=["tau", "ss_alb", "legendre_moments"],
                )
                self.atmosphere.tau = atmosphere_weighted_means["tau"]
                self.atmosphere.ss_alb = atmosphere_weighted_means["ss_alb"]
                self.atmosphere.legendre_moments = atmosphere_weighted_means[
                    "legendre_moments"
                ]

            if "land" in components:
                if self.config.SOLVER.TYPE == "two-stream-ad":
                    land_weighted_means = self.compute_srf_solar_weighted_average(
                        self.land,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=[
                            "ref_idx_re",
                            "ref_idx_im",
                            "sfc",
                            "tau",
                            "ss_alb",
                            "asm_prm",
                        ],
                    )
                    self.land.ref_idx_re = land_weighted_means["ref_idx_re"]
                    self.land.ref_idx_im = land_weighted_means["ref_idx_im"]
                    self.land.tau = land_weighted_means["tau"]
                    self.land.ss_alb = land_weighted_means["ss_alb"]
                    self.land.asm_prm = land_weighted_means["asm_prm"]
                    self.land.sfc = land_weighted_means["sfc"].flatten()

                else:
                    # average land variables
                    land_weighted_means = self.compute_srf_solar_weighted_average(
                        self.land,
                        self.config._wavelengths_land,
                        self._band_ranges,
                        var_names=[
                            "tau",
                            "ss_alb",
                            "legendre_moments",
                        ],
                    )
                    self.land.tau = land_weighted_means["tau"]
                    self.land.ss_alb = land_weighted_means["ss_alb"]
                    self.land.legendre_moments = land_weighted_means["legendre_moments"]

            if "solar" in components:
                solar_weighted_means = self.compute_srf_solar_weighted_average(
                    self.solar,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["direct_beam", "diffuse", "total_irradiance"],
                )
                self.solar.direct_beam = solar_weighted_means["direct_beam"].flatten()
                self.solar.diffuse = solar_weighted_means["diffuse"].flatten()
                self.solar.total_irradiance = solar_weighted_means[
                    "total_irradiance"
                ].flatten()
