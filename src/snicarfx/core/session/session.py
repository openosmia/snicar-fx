"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib
import sys
from datetime import datetime
from importlib.metadata import version

import numpy as np
import xarray as xr

from ..components.atmosphere import AtmosphereColumn
from ..components.land import LandColumn
from ..components.solar import SolarIrradiance
from ..solvers.multi_stream_solver import solve_multi_stream_rt
from ..solvers.two_stream_solver import solve_two_stream_rt
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
            current_laps_files = [p.FILE for p 
                            in self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.values()]
            if (any(key not in current_laps 
                   for key in kwargs['LIGHT_ABSORBING_PARTICLES'].keys())
                or
                any(p['FILE'] not in current_laps_files 
                       for p in kwargs['LIGHT_ABSORBING_PARTICLES'].values()) 
                ):
                raise ValueError(
                    "LAP number and files cannot be changed at runtime. "
                    "Please modify them in the input YAML file. "
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

    def update_solver(self, update_dic, validate=True):
        """
        Update allowed solver fields from user-defined dictionary.
        """
        
        # all fields allowed except legendre moments as this would require to
        # recalculate all optical properties at the moment, esp for the 
        # atmosphere 
        
        allowed_fields = {
            'TYPE',
            'ATMOSPHERE_COUPLING',
            'OUTPUT_LEVELS', 
            'N_STREAMS', 
            'N_FOURIER_MODES', 
            'RELATIVE_AZIMUTH'
            }
        
        updates = self._prepare_updates(update_dic, allowed_fields)

        # store applied updates
        self._latest_updates["SOLVER"].update(updates)

        # validate by creating a new instance of Solver
        if validate:
            self.config.SOLVER.__class__(**updates)
        
        # # update solver parameters only if updates not empty
        if updates: 
            
            if 'TYPE' in updates:
                self.config.SOLVER.TYPE = updates['TYPE']
            if 'ATMOSPHERE_COUPLING' in updates: 
                self.config.SOLVER.ATMOSPHERE_COUPLING = updates['ATMOSPHERE_COUPLING']
                self.atmosphere_column.use_atmosphere = False
            if 'OUTPUT_LEVELS' in updates: 
                self.config.SOLVER.OUTPUT_LEVELS = updates['OUTPUT_LEVELS']
            if 'N_STREAMS' in updates: 
                self.config.SOLVER.N_STREAMS = updates['N_STREAMS']
            if 'N_FOURIER_MODES' in updates: 
                self.config.SOLVER.N_FOURIER_MODES = updates['N_FOURIER_MODES']
            if 'RELATIVE_AZIMUTH' in updates: 
                self.config.SOLVER.RELATIVE_AZIMUTH = updates['RELATIVE_AZIMUTH']
            

    def update_solar(self, update_dic, validate=True):
        """
        Update allowed solar fields from user-defined dictionary.
        """

        # only SZA in solar
        allowed_fields = {"SZA"}
        updates = self._prepare_updates(update_dic, allowed_fields)

        # store applied updates
        self._latest_updates["SOLAR"].update(updates)

        # validate by creating a new instance of Solver
        if validate:
            self.config.SOLAR.__class__(**updates)

        # update SZA and recompute irradiance only if updates not empty
        if updates:
            self.solar_irradiance.sza = updates["SZA"]
            if (self.config.SOLVER.TYPE == 'two-stream' 
                or not self.config.SOLVER.ATMOSPHERE_COUPLING): 
                # surface irradiance needs to be re-calculated if SZA updated 
                self.solar_irradiance.load_surface_irradiance()
                self.solar_irradiance.set_surface_irradiance(self.config)
                
            if "band-" in self.config.SPECTRAL.MODE:
                self.compute_band_average()

    def update_atmosphere(self, update_dic, validate=True):
        """
        Update allowed atmospheric fields from user-defined dictionary.
        """

        # for now we do not change sky conditions, atmospheric profile type 
        # & aerosol properties
        
        allowed_fields = {"INTEGRATED_AOD_550", 
                          "INTEGRATED_GAS_CONCENTRATIONS",
                          }
        updates = self._prepare_updates(update_dic, allowed_fields)

        # store applied updates
        self._latest_updates["ATMOSPHERE"].update(updates)

        # validate a copy of the config if requested 
        if validate:
            self.config.ATMOSPHERE.__class__(**updates)

        # update only if not empty
        if updates:

            if "INTEGRATED_AOD_550" in updates:
                # load aerosols properties if not in session already
                if not self.atmosphere_column.AOD:
                    self.atmosphere_column.set_aerosol_properties()
                # update AOD (final OP calculations after gas update)
                self.atmosphere_column.AOD = updates["INTEGRATED_AOD_550"]
                self.atmosphere_column.scale_tau_aerosols()
                
            # if any gas to update, re-compute gas optical thickness
            if "INTEGRATED_GAS_CONCENTRATIONS" in updates:
                self.atmosphere_column.integrated_gas_concentrations = updates["INTEGRATED_GAS_CONCENTRATIONS"]
                self.atmosphere_column.scale_atmospheric_profile()
                self.atmosphere_column.compute_gas_optical_thickness()
                
            # if aerosols, re-compute aerosol AND atmosphere optics 
            if self.atmosphere_column.AOD: 
                self.atmosphere_column.set_atmospheric_properties_with_aerosols()
            # if no aerosols, re-compute atmosphere optics w/out aerosols
            else: 
                self.atmosphere_column.set_atmospheric_properties_without_aerosols()
                
            if "band-" in self.config.SPECTRAL.MODE:
                self.compute_band_average()

    def update_land(self, update_dic, validate=True):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {
            "LAYER_TYPE",
            "GRAIN_SHAPE",
            "RF_TYPE",
            "LWC",
            "THICKNESS",
            "SPECIFIC_SURFACE_AREA",
            "DENSITY",
            
            "LIGHT_ABSORBING_PARTICLES"
        }
       
        updates = self._prepare_updates(update_dic, allowed_fields)
        
        # store applied updates
        self._latest_updates["LAND"].update(updates)

        # validate a copy of the config if requested 
        if validate:
            self.config.LAND.__class__(**updates)

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
                self.land_column.set_refractive_index_and_diffuse_fresnel_coeffs()
            
            # all allowed parameters require to re-calculate clean column ops
            self.land_column.set_column_ops_without_laps()
            
            # if there are particles we need to update the properties even if 
            # we do not change the particle concentrations as the clean snow/ice
            # has changed
            if self.config.LAND.LIGHT_ABSORBING_PARTICLES.root.keys():
                self.land_column.update_column_ops_with_laps()
            
            # finally update legendre moments
            self.land_column.set_legendre_moments()
            
            if "band-" in self.config.SPECTRAL.MODE:
                self.compute_band_average()
            

    def run(self, to_xarray=True):
        """
        Run the radiative transfer solver and return outputs
        """

        if self.config.SOLVER.TYPE == "two-stream":
            self.outputs = solve_two_stream_rt(self.land_column, self.solar_irradiance)
            # return outputs as a metadata-rich xarray dataset
            if to_xarray:
                self.outputs = self.format_twostream_results_to_xarray()

        elif self.config.SOLVER.TYPE == "multi-stream":

            self.outputs = solve_multi_stream_rt(
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
                "wavelength": (
                    self._band_ranges[:, -1]
                    if "band-" in self.config.SPECTRAL.MODE
                    else self.config._wavelengths_land
                ),
                "layer": np.arange(len(self.outputs.absorbed_flux_fraction_per_layer)),
            },
        )

        # add attributes
        ds.attrs.update(attrs)

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
            "creation_date": datetime.utcnow().isoformat(),
            "session_state": self._write_current_state(),
        }

        # store albedo, reflectance and radiance variables along with
        # their dimensions
        albedo_variables = {}
        directional_variables = {}
        for var_name, data in self.outputs.items():
            if "albedo_" in var_name:
                albedo_variables[var_name] = ("wavelength", data)

            elif "directional_" in var_name:
                if self.config.SOLVER.N_FOURIER_MODES == 1:
                    directional_variables[var_name] = (
                        ("viewing_angle", "wavelength"),
                        data,
                    )
                else:
                    directional_variables[var_name] = (
                        ("viewing_angle", "wavelength", "azimuth_angle"),
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
                    "viewing_angle": self.outputs["viewing_angle"],
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
                    "viewing_angle": self.outputs["viewing_angle"],
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
        ds["viewing_angle"].attrs.update(
            {
                "description": "Result viewing angle. Defined from 0 to 90 degrees, as 90-180 degrees is covered by the opposite azimuth angle.",
                "units": "degrees",
            }
        )

        if self.config.SOLVER.N_FOURIER_MODES > 1:
            ds["azimuth_angle"].attrs.update(
                {
                    "description": "Azimuth angle relative to the prescribed solar azimuth angle.",
                    "units": "degrees",
                }
            )

        return ds

    def apply_spectral_response_function(self) -> None:

        for key, var in self.outputs.items():

            if any(tag in key for tag in ["albedo_", "directional_"]):
                self.outputs[key] = (
                    np.nansum(
                        self.outputs[key][:, None, :]
                        * self._spectral_response_function[None, :, :],
                        axis=-1,
                    )
                    / np.nansum(self._spectral_response_function, axis=1)[None, :]
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

                    # trapezoidal integration along last axis (wavelength)
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

        for name in var_names:

            arr = getattr(column, name)

            n_bands = band_ranges.shape[0]
            averaged_rows = np.empty(
                np.append(arr.shape[:-1] if arr.ndim > 1 else 1, n_bands)
            )

            for b, (lam_min, lam_max, _) in enumerate(band_ranges):

                if name in ["flx_slr", "fs", "fd"]:
                    numerator = (
                        self.solar_irradiance.flx_slr
                        * self._spectral_response_function[b, :]
                    )
                    denominator = self._spectral_response_function[b, :]

                else:
                    numerator = (
                        self.solar_irradiance.flx_slr
                        * self._spectral_response_function[b, :]
                        * arr
                    )

                    denominator = (
                        self.solar_irradiance.flx_slr
                        * self._spectral_response_function[b, :]
                    )

                # trapezoidal integration along last axis (wavelength)
                numerator_integral = np.trapezoid(numerator, wavelengths, axis=-1)
                denominator_integral = np.trapezoid(denominator, wavelengths, axis=-1)

                averaged_rows[..., b] = numerator_integral / denominator_integral

            band_means[name] = averaged_rows

        return band_means

    def compute_band_average(self) -> None:

        if self.config.SPECTRAL.MODE == "band-snicar-default":

            # average solar variables
            solar_flat_means = self.compute_flat_band_average(
                self.solar_irradiance,
                self.config._wavelengths_solar,
                self._band_ranges,
                var_names=["flx_slr"],
            )
            self.solar_irradiance.flx_slr = solar_flat_means["flx_slr"]

            if self.config.SOLVER.TYPE == "two-stream":
                solar_flat_means = self.compute_flat_band_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["fs", "fd"],
                )
                self.solar_irradiance.fs = solar_flat_means["fs"]
                self.solar_irradiance.fd = solar_flat_means["fd"]

            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING == True:
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

            # average atmosphere variables
            if self.config.SOLVER.ATMOSPHERE_COUPLING == True:
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

            # average land variables
            land_weighted_means = self.compute_solar_weighted_average(
                self.land_column,
                self.config._wavelengths_land,
                self._band_ranges,
                var_names=["tau", "ss_alb", "legendre_moments", "asm_prm"],
            )
            self.land_column.tau = land_weighted_means["tau"]
            self.land_column.ss_alb = land_weighted_means["ss_alb"]
            self.land_column.legendre_moments = land_weighted_means["legendre_moments"]
            self.land_column.asm_prm = land_weighted_means["asm_prm"]

            if self.config.SOLVER.TYPE == "two-stream":
                land_weighted_means = self.compute_solar_weighted_average(
                    self.land_column,
                    self.config._wavelengths_land,
                    self._band_ranges,
                    var_names=["ref_idx_re", "ref_idx_im", "sfc"],
                )
                self.land_column.ref_idx_re = land_weighted_means["ref_idx_re"]
                self.land_column.ref_idx_im = land_weighted_means["ref_idx_im"]
                self.land_column.sfc = land_weighted_means["sfc"].flatten()

                solar_weighted_means = self.compute_solar_weighted_average(
                    self.solar_irradiance,
                    self.config._wavelengths_solar,
                    self._band_ranges,
                    var_names=["fs", "fd"],
                )
                self.solar_irradiance.fs = solar_weighted_means["fs"].flatten()
                self.solar_irradiance.fd = solar_weighted_means["fd"].flatten()

            # average solar variables
            solar_weighted_means = self.compute_solar_weighted_average(
                self.solar_irradiance,
                self.config._wavelengths_solar,
                self._band_ranges,
                var_names=["flx_slr"],
            )
            self.solar_irradiance.flx_slr = solar_weighted_means["flx_slr"].flatten()
