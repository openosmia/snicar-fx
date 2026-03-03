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


@pytest.mark.parametrize(
    "component, field, value",
    [
        # SOLAR
        (
            "SOLAR",
            "SZA",
            42,
        ),
        (
            "SOLAR",
            "SAA",
            180,
        ),
        # SOLVER
        ("SOLVER", "OUTPUT_LEVELS", "TOA"),
        ("SOLVER", "N_FOURIER_MODES", 3),
        (
            "SOLVER",
            "AZIMUTH_ANGLES",
            (10, 170, 5),
        ),
        (
            "SOLVER",
            "POLAR_ANGLES",
            (10, 170, 5),
        ),
        # ATMOSPHERE
        (
            "ATMOSPHERE",
            "INTEGRATED_AOD_550",
            0.42,
        ),
        (
            "ATMOSPHERE",
            "INTEGRATED_GAS_CONCENTRATIONS",
            {"H2O": 15, "NO2": 1e-02, "O3": 0.01},
        ),
        # LAND
        ("LAND", "LAYER_TYPE", (0, 0, 0)),
        ("LAND", "GRAIN_SHAPE", (0, 0, 0)),
        ("LAND", "RF_TYPE", "Pic16"),
        ("LAND", "LWC", (0.01, 0.01, 0.01)),
        (
            "LAND",
            "THICKNESS",
            (0.07, 0.04, 0.1),
        ),
        (
            "LAND",
            "SPECIFIC_SURFACE_AREA",
            (1, 2, 3),
        ),
        (
            "LAND",
            "DENSITY",
            (600, 700, 800),
        ),
        (
            "LAND",
            "LIGHT_ABSORBING_PARTICLES",
            {
                "BC1": {"FILE": "bc_ChCB_rn40_dns1270.nc", "CONC": (2, 20, 200)},
                "BC2": {"FILE": "bc_ChCB_rn40_dns1270.nc", "CONC": (3, 30, 300)},
            },
        ),
    ],
)
def test_update_api(
    test_input_file2, component, field, value, absolute_tolerance_update_api
):
    """
    Test the update API by comparing results obtained by modifying
    session wit the update API vs. manually modifying the input file
    re-initializing a new Session.

    """

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

    results_uapi = session_uapi.run(to_xarray=True)

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
    results_dup = session_dup.run(to_xarray=True)

    # clean up temp file
    os.unlink(tmp_path)

    # INTEGRATED_GAS_CONCENTRATIONS uses property scaling across
    # several orders of magnitude so that the match can never be
    # perfect, use a tight tolerance.
    if field == "INTEGRATED_GAS_CONCENTRATIONS":
        assert (
            np.isclose(
                results_dup["directional_reflectance_toa"],
                results_uapi["directional_reflectance_toa"],
                atol=absolute_tolerance_update_api,
            )
        ).all()
        assert (
            np.isclose(
                results_dup["albedo_toa"],
                results_uapi["albedo_toa"],
                atol=absolute_tolerance_update_api,
            )
        ).all()
    else:
        assert (
            results_dup["directional_reflectance_toa"]
            == results_uapi["directional_reflectance_toa"]
        ).all()
        assert (results_dup["albedo_toa"] == results_uapi["albedo_toa"]).all()

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
            results_dup["directional_reflectance_toa"]
            == results_uapi["directional_reflectance_toa"]
        ).all()
        assert (results_dup["albedo_toa"] == results_uapi["albedo_toa"]).all()
