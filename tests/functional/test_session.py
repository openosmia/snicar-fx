"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pytest
import yaml
from snicarfx.core.session.session import Session
import tempfile
import os
import numpy as np
from pydantic import BaseModel


def test_session_attributes(session):
    """
    Verify that the required attributes of Session exist and have the correct
    types.

    Parameters
    ----------
    session : Session
        Instance of the Session class
    """

    assert hasattr(session, "config")
    assert isinstance(session.config, BaseModel)


def test_get_package_root(session):

    package_root = session.get_package_root()

    assert "snicar-fx" in package_root.parts


# def test_format_multistream_results_to_xarray(session2):

#     results = session2.run(to_xarray=True)

#     assert results["albedo_toa"].shape == session2._band_ranges[:, -1].shape

#     assert results["directional_radiance_toa"].shape == (
#         len(np.arange(*session2.config.SOLVER.AZIMUTH_ANGLES)),
#         len(np.arange(*session2.n_angles)),
#         session2._band_ranges[:, -1].shape[0],
#     )


def test_update_api(
    test_input_file2,
    update_api_params,
    absolute_tolerance_update_api_gs,
    absolute_tolerance_update_api,
):
    """
    Test the update API by comparing results obtained by modifying
    session wit the update API vs. manually modifying the input file
    re-initializing a new Session.

    """

    component = update_api_params["component"]
    field = update_api_params["field"]
    value = update_api_params["value"]

    # use fresh session to modify the field with the update API
    session_uapi = Session(test_input_file2)
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
    with open(test_input_file2, "r") as f:
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

    # INTEGRATED_GAS_CONCENTRATIONS uses property scaling across
    # several orders of magnitude so that the match can never be
    # perfect, use a tight tolerance
    if field == "INTEGRATED_GAS_CONCENTRATIONS":
        tolerance = absolute_tolerance_update_api_gs
    else:
        tolerance = absolute_tolerance_update_api

    assert (
        np.isclose(
            results_dup["directional_reflectance_toa"],
            results_uapi["directional_reflectance_toa"],
            atol=tolerance,
            rtol=0.0,
        )
    ).all()
    assert (
        np.isclose(
            results_dup["albedo_toa"],
            results_uapi["albedo_toa"],
            atol=tolerance,
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
        session_uapi = Session(test_input_file2)
        session_uapi.update_land(updates, validate=False)
        results_uapi = session_uapi.run(to_xarray=True)

        assert (
            np.isclose(
                results_dup["directional_reflectance_toa"],
                results_uapi["directional_reflectance_toa"],
                atol=tolerance,
                rtol=0.0,
            )
        ).all()
        assert (
            np.isclose(
                results_dup["albedo_toa"],
                results_uapi["albedo_toa"],
                atol=tolerance,
                rtol=0.0,
            )
        ).all()
