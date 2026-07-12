"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import copy
import os
import tempfile

import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import BaseModel, ValidationError

from snicarfx import Session
from snicarfx.core import AtmosphereColumn, LandColumn, SolarIrradiance


def test_session_attributes(session_multistream_coupled):
    """
    Verify that the required attributes of Session exist and have the correct
    types.

    Parameters
    ----------
    session_multistream_coupled : Session
        Instance of the Session class for multi-stream coupled configuration
    """

    session = session_multistream_coupled

    # check core components exist
    assert hasattr(session, "land")
    assert hasattr(session, "solar")
    assert hasattr(session, "atmosphere")
    assert hasattr(session, "config")

    # check spectral arrays are set (critical path in _set_spectral_array)
    assert hasattr(session.config, "_wavelengths")
    assert hasattr(session.config, "_wavelengths_solar")
    assert hasattr(session.config, "_wavelengths_land")

    assert hasattr(session_multistream_coupled, "config")
    assert isinstance(session_multistream_coupled.config, BaseModel)

    # if band mode, check band ranges
    if "band-" in session.config.SPECTRAL.MODE:
        assert hasattr(session, "_band_ranges")
        assert session._band_ranges.shape[1] == 3  # min, max, center


def test_session_initialization_twostream(test_input_file_twostream):
    """
    Verify Session initializes correctly for two-stream configuration.
    """
    session = Session(test_input_file_twostream)

    assert session.config.SOLVER.TYPE == "two-stream-ad"
    assert hasattr(session, "land")
    # two-stream might not have atmosphere if coupling is off, check config
    if session.config.SOLVER.ATMOSPHERE_COUPLING:
        assert hasattr(session, "atmosphere")


def test_run_solver_routing_two_stream(test_input_file_twostream):
    """
    Verify that run() executes the two-stream solver path correctly.
    """
    session = Session(test_input_file_twostream)

    # run and check outputs exist
    results = session.run(to_xarray=False)

    assert isinstance(results, dict)
    assert "albedo_boa" in results
    assert "bba_boa" in results


def test_run_solver_routing_multistream_ada(test_input_file_multistream_uncoupled):
    """
    Verify that run() executes the multi-stream ADA solver path.
    """
    session = Session(test_input_file_multistream_uncoupled)
    # ensure config is set to ADA
    session.config.SOLVER.TYPE = "multi-stream-ada"

    results = session.run(to_xarray=False)

    assert isinstance(results, dict)
    assert "albedo_boa" in results

def test_run_output_formats(session_multistream_coupled):
    """
    Verify that run() returns correct types for to_xarray=True vs False.
    """
    # test Dictionary output
    results_dict = session_multistream_coupled.run(to_xarray=False)
    assert isinstance(results_dict, dict)

    # test xarray output
    results_xr = session_multistream_coupled.run(to_xarray=True)
    assert isinstance(results_xr, xr.Dataset)

    # check xarray metadata attributes
    assert "model_name" in results_xr.attrs
    assert results_xr.attrs["model_name"] == "snicar-fx"
    assert "session_state" in results_xr.attrs
    assert "creation_date" in results_xr.attrs


def test_write_current_state(session_multistream_coupled):
    """
    Verify _write_current_state returns a dictionary dump of the config.
    """
    state = session_multistream_coupled._write_current_state()

    assert isinstance(state, dict)
    assert "SOLVER" in state
    assert "LAND" in state
    assert "SOLAR" in state


def test_format_twostream_results_to_xarray(test_input_file_twostream):
    """
    Verify formatting of two-stream results includes correct coordinates and attrs.
    """
    session = Session(test_input_file_twostream)

    # mock 2 wavelengths
    n_mock_wl = 2
    session.config._wavelengths_land = np.linspace(400, 700, n_mock_wl)
    session._band_ranges = np.column_stack(([350, 650], [450, 750], [400, 700]))

    session.outputs = {
        "albedo_boa": np.array([0.5, 0.6]),
        "bba_boa": 0.55,
        "absorbed_flux_fraction": np.array([0.1, 0.2]),
        "absorbed_flux_fraction_bottom": np.array([0.05, 0.05]),
    }

    ds = session.format_twostream_results_to_xarray()

    assert isinstance(ds, xr.Dataset)
    assert "wavelength" in ds.coords
    assert "layer" in ds.coords
    assert "albedo_boa" in ds.data_vars
    assert ds.attrs["model_name"] == "snicar-fx"

    # assert our mock worked
    assert ds.sizes["wavelength"] == 2
    assert ds.sizes["layer"] == 2


def test_format_multistream_results_to_xarray_mocked(
    test_input_file_multistream_coupled,
):
    """
    Verify formatting of multistream results handles N_FOURIER_MODES correctly.
    """

    session = Session(test_input_file_multistream_coupled)

    n_wl = (
        len(session.config._wavelengths)
        if "band-" not in session.config.SPECTRAL.MODE
        else session._band_ranges.shape[0]
    )

    if "band-" in session.config.SPECTRAL.MODE:
        n_wl = session._band_ranges.shape[0]
    else:
        n_wl = len(session.config._wavelengths_land)

    # mock outputs for N_FOURIER_MODES == 1
    session.config.SOLVER.N_FOURIER_MODES = 1
    session.outputs = {
        "albedo_toa": np.ones(n_wl) * 0.5,
        "bba_toa": 0.5,
        "directional_reflectance_boa_m0": np.ones((10, n_wl)) * 0.1,
        "polar_angle": np.linspace(0, 90, 10),
    }

    ds = session.format_multistream_results_to_xarray()
    assert isinstance(ds, xr.Dataset)

    session.config.SOLVER.N_FOURIER_MODES = 3
    session.outputs["azimuth_angle"] = np.linspace(0, 180, 5)
    session.outputs["directional_reflectance_boa"] = np.ones((10, n_wl, 5)) * 0.1

    ds_multi = session.format_multistream_results_to_xarray()
    assert "azimuth_angle" in ds_multi.coords


def test_compute_flat_band_average(session_multistream_coupled):
    """
    Verify compute_flat_band_average correctly integrates over wavelengths.
    """
    session = session_multistream_coupled

    # create a mock component with known data
    class MockComponent:
        def __init__(self):
            self.tau = np.ones((2, 10)) * 0.5
            self.ss_alb = np.ones((2, 10)) * 0.9

    mock_comp = MockComponent()
    wavelengths = np.linspace(400, 700, 10)
    band_ranges = np.array([[400, 500, 450], [500, 700, 600]])

    result = session.compute_flat_band_average(
        mock_comp, wavelengths, band_ranges, var_names=["tau", "ss_alb"]
    )

    assert "tau" in result
    assert "ss_alb" in result
    # result shape should be (layers, n_bands)
    assert result["tau"].shape == (2, 2)
    # values should be close to input since input was constant
    assert np.allclose(result["tau"], 0.5, rtol=1e-5)


def test_apply_spectral_response_function(session_multistream_coupled):
    """
    Verify that apply_spectral_response_function correctly
    integrates spectral outputs over the satellite spectral response
    function.

    Checks that outputs are reduced from wavelength resolution to band
    resolution and that values change during the integration process.
    """

    session = session_multistream_coupled

    n_wl = len(session.config._wavelengths_solar)
    n_bands = session._band_ranges.shape[0]

    # averaging a gradient will produce different values than the original points.
    session.outputs = {
        "albedo_toa": np.linspace(0, 1, n_wl),
        "directional_reflectance_boa_m0": np.tile(np.linspace(0, 1, n_wl), (10, 1)),
    }

    original_albedo = session.outputs["albedo_toa"].copy()

    session.apply_spectral_response_function()

    assert session.outputs["albedo_toa"].shape == (n_bands,)
    assert session.outputs["directional_reflectance_boa_m0"].shape[1] == n_bands

    assert not np.array_equal(session.outputs["albedo_toa"], original_albedo[:n_bands])


def test_get_package_root(session_multistream_coupled):
    package_root = session_multistream_coupled.get_package_root()

    assert "snicar-fx" in package_root.parts


def test_format_multistream_results_to_xarray(session_multistream_coupled):
    """
    Test that xarray outputs match expected shapes based on user
    inputs.

    Parameters
    ----------
    session_multistream_coupled : Session
        Instance of the Session class for multi-stream coupled configuration
    """
    results = session_multistream_coupled.run(to_xarray=True)

    n_bands = session_multistream_coupled._band_ranges[:, -1].shape[0]
    n_azimuth_angles = len(
        np.arange(*session_multistream_coupled.config.SOLVER.AZIMUTH_ANGLES)
    )
    n_polar_angles = len(
        np.arange(*session_multistream_coupled.config.SOLVER.POLAR_ANGLES)
    )

    expected_1d_shape = (n_bands,)
    expected_2d_shape = (n_polar_angles, n_bands)
    expected_3d_shape = (n_polar_angles, n_bands, n_azimuth_angles)

    assert results["albedo_toa"].shape == expected_1d_shape
    assert results["albedo_boa"].shape == expected_1d_shape

    assert results["directional_radiance_toa"].shape == expected_3d_shape
    assert results["directional_reflectance_boa"].shape == expected_3d_shape

    assert results["directional_reflectance_boa_m0"].shape == expected_2d_shape
    assert results["directional_reflectance_toa_m0"].shape == expected_2d_shape


def test_update_api(
    test_input_file_multistream_coupled,
    update_api_params,
    absolute_tolerance_update_api,
):
    """
    Test the update API by comparing results obtained by modifying
    session wit the update API vs. manually modifying the input file and
    re-initializing a new Session instance.

    Parameters
    ----------
    test_input_file_multistream_coupled : str
        File name of the input file used to re-initialize Session.
    update_api_params : dict
        Component, field and value of the parameter to update.
    absolute_tolerance_update_api : float
        Tolerance on error between updated and re-initialized values.
    """

    component = update_api_params["component"]
    field = update_api_params["field"]
    value = update_api_params["value"]

    # use fresh session to modify the field with the update API
    session_uapi = Session(test_input_file_multistream_coupled)
    updates = {field: value}

    if component == "SOLAR":
        session_uapi.update_solar(updates)
    elif component == "ATMOSPHERE":
        session_uapi.update_atmosphere(updates)
    elif component == "LAND":
        session_uapi.update_land(updates)
    elif component == "SOLVER":
        session_uapi.update_solver(updates)

    results_uapi = session_uapi.run(to_xarray=False)

    # manually modify the input file and create a new session (not
    # recommended in snicarfx but required here to test the update
    # API, which is the recommanded way)
    with open(test_input_file_multistream_coupled) as f:
        config_dict = yaml.safe_load(f)

    # apply the same update as with the upate API, but manually
    if field == "INTEGRATED_GAS_CONCENTRATIONS":
        for gas, conc in updates["INTEGRATED_GAS_CONCENTRATIONS"].items():
            config_dict[component][field][gas] = conc
    else:
        config_dict[component][field] = value

    # write to temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
        yaml.dump(config_dict, tmp)
        tmp_path = tmp.name

    # instanciate and run with the new, updated input file
    session_dup = Session(tmp_path)
    results_dup = session_dup.run(to_xarray=False)

    # clean up temp file
    os.unlink(tmp_path)

    assert (
        np.isclose(
            results_dup["directional_reflectance_toa"],
            results_uapi["directional_reflectance_toa"],
            atol=absolute_tolerance_update_api,
            rtol=0.0,
        )
    ).all()
    assert (
        np.isclose(
            results_dup["albedo_toa"],
            results_uapi["albedo_toa"],
            atol=absolute_tolerance_update_api,
            rtol=0.0,
        )
    ).all()

    # in addition, test a specific case often used in batch processing
    # where light absorbing particle file names are not being passed
    # again
    if field == "LIGHT_ABSORBING_PARTICLES":
        value_without_file = {
            k: {subk: subv for subk, subv in v.items() if subk != "FILE"}
            for k, v in value.items()
        }
        updates = {field: value_without_file}
        session_uapi = Session(test_input_file_multistream_coupled)
        session_uapi.update_land(updates, validate=False)
        results_uapi = session_uapi.run(to_xarray=True)

        assert (
            np.isclose(
                results_dup["directional_reflectance_toa"],
                results_uapi["directional_reflectance_toa"],
                atol=absolute_tolerance_update_api,
                rtol=0.0,
            )
        ).all()
        assert (
            np.isclose(
                results_dup["albedo_toa"],
                results_uapi["albedo_toa"],
                atol=absolute_tolerance_update_api,
                rtol=0.0,
            )
        ).all()


def test_update_api_sequential_scaling(
    test_input_file_multistream_coupled,
    update_api_scaling_params,
    absolute_tolerance_update_api,
):
    """
    Test the update API by comparing results obtained by modifying
    session wit the update API vs. manually modifying the input file
    re-initializing a new Session.

    This time, by keeping the same session through a sequence of
    atmosphere updates to make sure updates are independent of each
    other (i.e. that (1) zero-scalings do not prevent future scalings
    and (2) a loss of precision is not propagated).

    Parameters
    ----------
    test_input_file_multistream_coupled : str
        File name of the input file used to re-initialize Session.
    update_api_params : dict
        Component, field and value of the parameter to update.
    absolute_tolerance_update_api : float
        Tolerance on error between updated and re-initialized values.

    """

    field = update_api_scaling_params["field"]
    sequence = update_api_scaling_params["sequence"]

    # use fresh session to modify the field with the update API, and
    # use it throughout the sequence of updates
    session_uapi = Session(test_input_file_multistream_coupled)

    # loop over the sequence containing different updates to scale,
    # including zeros followed by non-zero updates
    for values in sequence:
        updates = {field: values}
        session_uapi.update_atmosphere(updates)
        results_uapi = session_uapi.run(to_xarray=False)

        # manually modify the input file and create a new session (not
        # recommended in snicarfx but required here to test the update
        # API, which is the recommanded way)
        with open(test_input_file_multistream_coupled) as f:
            config_dict = yaml.safe_load(f)

        # apply the same update as with the upate API
        config_dict["ATMOSPHERE"][field] = values

        # write to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
            yaml.dump(config_dict, tmp)
            tmp_path = tmp.name

        # instanciate (at every update, as opposed to the reused API
        # session) and run with the new, updated input file
        session_dup = Session(tmp_path)
        results_dup = session_dup.run(to_xarray=False)

        # clean up temp file
        os.unlink(tmp_path)

        assert (
            np.isclose(
                results_dup["directional_reflectance_toa"],
                results_uapi["directional_reflectance_toa"],
                atol=absolute_tolerance_update_api,
                rtol=0.0,
            )
        ).all()
        assert (
            np.isclose(
                results_dup["albedo_toa"],
                results_uapi["albedo_toa"],
                atol=absolute_tolerance_update_api,
                rtol=0.0,
            )
        ).all()


def test_update_api_invalid_fields(
    session_multistream_coupled, invalid_update_api_field
):
    """
    Verify that update_solar, update_atmosphere, and update_land
    reject non-existent fields.
    """
    method_name, invalid_field = invalid_update_api_field

    # Dynamically get the method (e.g., session.update_solar)
    update_method = getattr(session_multistream_coupled, method_name)

    # assert that the field is rejected
    with pytest.raises(ValueError):
        update_method({invalid_field: 42})


def test_update_api_invalid_lengths(
    session_multistream_coupled, invalid_update_api_length
):
    """
    Verify that update methods reject values with incorrect list lengths
    (mismatching the number of layers defined in the config).
    """
    method_name, field, invalid_value = invalid_update_api_length

    update_method = getattr(session_multistream_coupled, method_name)
    updates = {field: invalid_value}

    # assert that the value is rejected
    with pytest.raises(ValidationError):
        update_method(updates)


def test_update_api_unphysical_values(
    session_multistream_coupled, unphysical_update_api_value
):
    """
    Verify that update methods reject invalid enums, out-of-range numbers,
    and malformed nested structures.
    """
    method_name, field, invalid_value = unphysical_update_api_value

    update_method = getattr(session_multistream_coupled, method_name)
    updates = {field: invalid_value}

    # assert that the unphysical value is rejected
    with pytest.raises(ValidationError):
        update_method(updates)


def test_update_api_immutable_laps(
    session_multistream_coupled, invalid_update_api_lap_structure
):
    """
    Verify that LIGHT_ABSORBING_PARTICLES cannot have new types added
    or files changed, even if the data structure is valid.
    """
    method_name, updates = invalid_update_api_lap_structure
    update_method = getattr(session_multistream_coupled, method_name)

    # assert that the lap update is rejected
    with pytest.raises(ValueError):
        update_method(updates)

def test_compute_band_average(
    session_multistream_coupled
):
    """
    Verify that the different spectral modes run.
    """

    session = copy.deepcopy(session_multistream_coupled)
    
    session.config.SPECTRAL.MODE = (
        "band-srf-solar-weighted-mean"
        )
    session.land = LandColumn(session.config)
    session.solar = SolarIrradiance(session.config)
    session.atmosphere = AtmosphereColumn(session.config)
    session.compute_band_average()

    assert np.all(np.isfinite(session.solar.total_irradiance))
    assert np.all(np.isfinite(session.land.tau))
    assert np.all(np.isfinite(session.atmosphere.tau))

