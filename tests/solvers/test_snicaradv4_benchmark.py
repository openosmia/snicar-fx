import numpy as np
import pytest
import xarray as xr

from snicarfx.core import (
    ColumnProperties,
    ModelInputs,
    SolarIrradiance,
    solve_adding_doubling,
)
from tests.conftest import twostream_parameter_grid
from tests.solvers.utils import match_matlab_config


@pytest.mark.parametrize("idx, params", enumerate(twostream_parameter_grid()))
def test_twostreams_outputs(
    idx,
    params,
    column,
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
    idx : array
        Indices of parameter sets.
    params : array
        Sets of parameters used as input for the model.
    column : ColumnProperties
        Instance of the ColumnProperties class
    benchmark_snicaradv4_spectral_data : array
        Spectral albedo data generated with SNICAR-ADv4 for the parameter grid
        `params`.
    benchmark_snicaradv4_bba_data : array
        Broadband albedo data generated with SNICAR-ADv4 for the parameter grid
        `params`.
    benchmark_snicaradv4_absorbed_flux_data : array
        Absorbed solar flux data generated with SNICAR-ADv4 for the parameter grid
        `params`.
    absolute_tolerance_benchmark: float
        Tolerance value for the error.
    
    """

    layer_type, density, radius, sza, bc, thickness_profile, direct = params

    # Setup inputs
    model_inputs = ModelInputs("./tests/inputs_tests.yaml")
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs)

    column = match_matlab_config(column)

    # calculate irradiance
    irradiance.direct = direct
    irradiance.sza = sza
    irradiance.set_irradiance()

    # calculate column ssa, g, mac
    column.thickness_profile = thickness_profile
    column.layer_type = [layer_type] * len(column.thickness_profile)
    column.density = [density] * len(column.thickness_profile)
    column.layer_mass = [
        column.density[i] * column.thickness_profile[i]
        for i in range(len(column.thickness_profile))
    ]

    snow_idx = np.where(np.array(column.layer_type) == 0)[0]
    ice_idx = np.where(np.array(column.layer_type) != 0)[0]

    for i in snow_idx:
        file_ssps = str(
            "./tests/test_data/ice_spherical_grains_BH83/"
            + f"ice_{column.rf_type}/ice_{column.rf_type}_"
            + "{}.nc".format(str(radius).rjust(4, "0"))
        )

        with xr.open_dataset(file_ssps) as ssps:
            column.ss_alb[i, :] = ssps["ss_alb"].values
            column.ext_cff[i, :] = ssps["ext_cff_mss"].values
            column.asm_prm[i, :] = ssps["asm_prm"].values
            column.tau[i, :] = column.layer_mass[i] * column.ext_cff[i, :]

    for i in ice_idx:
        file_ssps = str(
            "./tests/test_data/bubbly_ice_files_BH83/"
            + "bbl_{}.nc".format(str(radius).rjust(4, "0"))
        )
        with xr.open_dataset(file_ssps) as ssps:
            sca_cff_vlm_air_bbl = ssps["sca_cff_vlm"].values
            vlm_frac_air = 1 - column.density[i] / 917
            scattering_cff = sca_cff_vlm_air_bbl * vlm_frac_air / column.density[i]
            abs_cff = (4 * np.pi * column.ref_idx_im) / (column.wavelengths) / 917
            column.ext_cff[i, :] = scattering_cff + abs_cff
            column.ss_alb[i, :] = scattering_cff / column.ext_cff[i, :]
            column.asm_prm[i, :] = ssps["asm_prm"].values
            column.tau[i, :] = column.layer_mass[i] * column.ext_cff[i, :]

    column.lap_concentrations[:, 0] = bc * 1e-9

    column.update_column_ops_with_laps()

    # solve RTE
    outputs = solve_adding_doubling(column, irradiance)

    # spectral albedo only until 2705nm for now, as the asymmetry parameter is 
    # clipped to 0.99 in SNICAR-ADv4 but not in snicar-fx, producing larger
    # discrepancies than the tolerance of 1e-5.
    
    assert np.allclose(
        outputs.albedo[:250],
        benchmark_snicaradv4_spectral_data[idx][:250],
        atol=absolute_tolerance_benchmark,
    )
    # BBA
    assert np.allclose(
        outputs.BBA,
        benchmark_snicaradv4_bba_data[idx],
        atol=absolute_tolerance_benchmark,
    )
    # Absorbed flux
    assert np.allclose(
        outputs.abs_slr_tot,
        benchmark_snicaradv4_absorbed_flux_data[idx],
        atol=absolute_tolerance_benchmark,
    )
