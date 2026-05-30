"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import os
import tempfile

import numpy as np
import yaml
from pydantic import BaseModel

from snicarfx import Session


def test_session_attributes(session_multistream_coupled):
    """
    Verify that the required attributes of Session exist and have the correct
    types.

    Parameters
    ----------
    session_multistream_coupled : Session
        Instance of the Session class for multi-stream coupled configuration
    """

    assert hasattr(session_multistream_coupled, "config")
    assert isinstance(session_multistream_coupled.config, BaseModel)


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

    expected_1D_shape = (n_bands,)
    expected_2D_shape = (n_polar_angles, n_bands)
    expected_3D_shape = (n_polar_angles, n_bands, n_azimuth_angles)

    assert results["albedo_toa"].shape == expected_1D_shape
    assert results["albedo_boa"].shape == expected_1D_shape

    assert results["directional_radiance_toa"].shape == expected_3D_shape
    assert results["directional_reflectance_boa"].shape == expected_3D_shape

    assert results["directional_reflectance_boa_m0"].shape == expected_2D_shape
    assert results["directional_reflectance_toa_m0"].shape == expected_2D_shape


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

    # apply the same update as with the upate API
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
