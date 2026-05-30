"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import numpy as np
import xarray as xr
import copy

from snicarfx.core.solvers.two_stream_solver_ad import solve_two_stream_rt_ad
from tests.solvers.utils import use_data_snicaradv4


def test_twostreams_outputs(
    session_twostream,
    params_twostream,
    benchmark_snicaradv4_spectral_data,
    benchmark_snicaradv4_bba_data,
    benchmark_snicaradv4_absorbed_flux_data,
    absolute_tolerance_benchmark,
):
    """
    Assert that snicar-fx reproduces the spectral albedo, BBA and absorbed
    solar fluxes modelled by SNICAR-ADv4 within a tolerance of 1e-5 when the
    exact same model configuration is used. Model configuration is defined via
    the input test file, the `params` parameter as well as the match_matlab_config
    function.


    Parameters
    ----------
    session_twostream : Session
        Instance of Session class from snicar-fx.
    params_twostream : array
        Sets of parameters used as input for the model.
    land_column : LandColumn
        Instance of the LandColumn class
    benchmark_snicaradv4_spectral_data : array
        Spectral albedo data generated with SNICAR-ADv4 for the parameter grid
        `params_twostream`.
    benchmark_snicaradv4_bba_data : array
        Broadband albedo data generated with SNICAR-ADv4 for the parameter grid
        `params_twostream`.
    benchmark_snicaradv4_absorbed_flux_data : array
        Absorbed solar flux data generated with SNICAR-ADv4 for the parameter grid
        `params_twostream`.
    absolute_tolerance_benchmark: float
        Tolerance value for the error.

    """

    layer_type, density, radius, sza, bc, thickness_profile, direct = params_twostream

    land_column = copy.deepcopy(session_twostream.land_column)
    irradiance = copy.deepcopy(session_twostream.solar_irradiance)

    # Setup inputs
    land_column = session_twostream.land_column
    irradiance = session_twostream.solar_irradiance

    # # calculate irradiance
    irradiance.sza = sza

    if direct == 1:
        irradiance.sky_conditions = "clear"
    elif direct == 0:
        irradiance.sky_conditions = "cloudy"

    # match irradiance type, fnl coeffs and ref idx from Matlab config
    land_column, irradiance = use_data_snicaradv4(land_column, irradiance)

    # calculate column ssa, g, mac
    land_column.thickness_profile = thickness_profile
    land_column.layer_type = [layer_type] * len(land_column.thickness_profile)
    land_column.density = [density] * len(land_column.thickness_profile)
    land_column.layer_mass = [
        land_column.density[i] * land_column.thickness_profile[i]
        for i in range(len(land_column.thickness_profile))
    ]

    snow_idx = np.where(np.array(land_column.layer_type) == 0)[0]
    ice_idx = np.where(np.array(land_column.layer_type) != 0)[0]

    for i in snow_idx:
        file_ssps = str(
            "./tests/test_data/ice_spherical_grains_BH83/"
            + f"ice_{land_column.rf_type}/ice_{land_column.rf_type}_"
            + "{}.nc".format(str(radius).rjust(4, "0"))
        )

        with xr.open_dataset(file_ssps) as ssps:
            land_column.ext_cff[i, :] = ssps["ext_cff_mss"].values
            land_column.ss_alb[i, :] = ssps["ss_alb"].values
            land_column.asm_prm[i, :] = ssps["asm_prm"].values
            land_column.tau[i, :] = (
                land_column.layer_mass[i] * land_column.ext_cff[i, :]
            )

    for i in ice_idx:
        file_ssps = str(
            "./tests/test_data/bubbly_ice_files_BH83/"
            + "bbl_{}.nc".format(str(radius).rjust(4, "0"))
        )
        with xr.open_dataset(file_ssps) as ssps:
            sca_cff_vlm_air_bbl = ssps["sca_cff_vlm"].values
            vlm_frac_air = 1 - land_column.density[i] / 917
            scattering_cff = sca_cff_vlm_air_bbl * vlm_frac_air / land_column.density[i]
            abs_cff = (
                (4 * np.pi * land_column.ref_idx_im) / (land_column._wavelengths) / 917
            )
            land_column.ss_alb[i, :] = scattering_cff / (scattering_cff + abs_cff)
            land_column.asm_prm[i, :] = ssps["asm_prm"].values
            land_column.ext_cff[i, :] = scattering_cff + abs_cff
            land_column.tau[i, :] = land_column.layer_mass[i] * (
                scattering_cff + abs_cff
            )

    land_column.lap_concentrations[:, 0] = bc * 1e-9

    land_column.update_column_ops_with_laps()

    # solve RTE
    outputs = solve_two_stream_rt_ad(land_column, irradiance)

    # spectral albedo only until 2705nm for now, as the asymmetry parameter is
    # clipped to 0.99 in SNICAR-ADv4 but not in snicar-fx, producing larger
    # discrepancies than the tolerance of 1e-5.

    # fetch index of thickness profile
    thickness_profile_idx = np.where(
        np.all(benchmark_snicaradv4_bba_data.dz_layers == thickness_profile, axis=1)
    )[0][0]

    assert np.allclose(
        outputs["albedo_boa"][:250],
        benchmark_snicaradv4_spectral_data.sel(
            layer_type=layer_type + 1,
            density=density,
            reff=radius,
            sza=sza,
            bc=bc,
            direct=direct,
            thickness_profiles=thickness_profile_idx,
        )["albedo"].values[:250],
        atol=absolute_tolerance_benchmark,
        rtol=0.0,
    )

    assert np.allclose(
        outputs["bba_boa"],
        benchmark_snicaradv4_bba_data.sel(
            layer_type=layer_type + 1,
            density=density,
            reff=radius,
            sza=sza,
            bc=bc,
            direct=direct,
            thickness_profiles=thickness_profile_idx,
        )["BBA"].values,
        atol=absolute_tolerance_benchmark,
        rtol=0.0,
    )

    assert np.allclose(
        np.nansum(outputs["absorbed_flux_fraction"]),
        benchmark_snicaradv4_absorbed_flux_data.sel(
            layer_type=layer_type + 1,
            density=density,
            reff=radius,
            sza=sza,
            bc=bc,
            direct=direct,
            thickness_profiles=thickness_profile_idx,
        )["flux_absorbed"].values,
        atol=absolute_tolerance_benchmark,
        rtol=0.0,
    )
