"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

This file creates the Fortan ADA solver data necessary to test it
against its Python implementation.

"""

import subprocess

import numpy as np
import xarray as xr

import snicarfx.core.advanced_doubling_adding_solver_vectorized as adv_solver_mod
from snicarfx.core import ColumnProperties, ModelInputs, SolarIrradiance

# %% instanciate snicar-fx classes

model_inputs = ModelInputs("./inputs_tests.yaml")
column = ColumnProperties(model_inputs)
irradiance = SolarIrradiance(model_inputs)

# %% declare ranges of parameter to use in runs of the ADA solver

# single scattering albedo values
w_list = np.arange(0.2, 0.7, 0.1)

# g values
g_list = [0.32, 0.86]

# fixed number of angles
n_angles = 8

# optical depth values
t_od_list = np.arange(5, 220, 20)

# discrete wavelength indices to use
wavelength_index_list = [10, 40, 60, 80]

# calculate Legendre polynomials required by the Fortran solver
nodes, weights = np.polynomial.legendre.leggauss(n_angles * 2)
cos_angle = nodes[n_angles:]
cos_weight = weights[n_angles:]

# allocate results of the Fortran solver
f90_results = np.zeros(
    (len(w_list), len(t_od_list), len(g_list), len(wavelength_index_list))
)

# %% run the Fortran solver

for wvl_enumarator, wavelength_index in enumerate(wavelength_index_list):

    # save g in files first because too big to pass at run time
    for g in g_list:

        column.asm_prm[:, wavelength_index] = g
        column.update_column_ops_with_laps()

        tst = adv_solver_mod.solve_advanced_adding_doubling(column, irradiance)

        solver = adv_solver_mod._AdvancedDoublingAddingSolver(column, irradiance)
        ff_wvl = solver.ff[:, :, :, wavelength_index]
        bb_wvl = solver.bb[:, :, :, wavelength_index]

        np.savetxt(
            f"./test_data/ff_g{g}.csv",
            ff_wvl[:, :, 0],
            delimiter=",",
        )

        np.savetxt(
            f"./test_data/bb_g{g}.csv",
            bb_wvl[:, :, 0],
            delimiter=",",
        )

    # re-initialize snicar-fx classes
    column = ColumnProperties(model_inputs)
    irradiance = SolarIrradiance(model_inputs)

    # extract values required by the Fortran solver
    cos_sun = irradiance.cos_sza
    solar_irradiance = solver.solar_irradiance[wavelength_index]

    # pass other simpler arguments now, at run time
    for t_od_enumerator, t_od in enumerate(t_od_list):
        for w_enumerator, w in enumerate(w_list):
            for g_enumerator, g in enumerate(g_list):

                # update python variables
                column.ss_alb[:, wavelength_index] = w
                column.tau[:, wavelength_index] = t_od
                column.asm_prm[:, wavelength_index] = g
                column.update_column_ops_with_laps()

                # initiate the python ADA solver to extract delta
                # scaled variables and feed them to the fortran ADA
                # solver, which doesn't have delta scaling
                solver = adv_solver_mod._AdvancedDoublingAddingSolver(
                    column, irradiance
                )

                # run fortran version by passing variables to the executable
                subprocess.run(
                    # ./run_ADA w T_OD g COS_SUN Solar_irradiance
                    (
                        f"./run_ADA {solver.w[0, wavelength_index]} "
                        f"{solver.t_od[0, wavelength_index]} {g} {cos_sun} "
                        f"{solar_irradiance} "
                    ),
                    shell=True,
                    executable="/bin/bash",
                )

                # read outputs from the fortran ADA solver
                s_level_rad_up_f90_k0 = np.loadtxt(
                    "../test_data/s_Level_Rad_UP_temp.csv"
                )

                # compute albedo of the fortran ADA solver to compare
                # with the python version in tests
                albedo_f90 = (
                    2
                    * np.pi
                    * np.sum(s_level_rad_up_f90_k0 * cos_angle * cos_weight)
                    / (solar_irradiance * cos_sun)
                )

                # store albedo results
                f90_results[
                    w_enumerator, t_od_enumerator, g_enumerator, wvl_enumarator
                ] = albedo_f90


# %% save results

data_xr = xr.Dataset(
    data_vars={"albedo": (["w", "t_od", "g", "wavelength_index"], f90_results)},
    coords={
        "w": w_list,
        "t_od": t_od_list,
        "g": g_list,
        "wavelength_index": wavelength_index_list,
    },
    attrs={
        "description": (
            "Albedo produced with the Fortran version of the "
            "multistream solver. All dimension values are before "
            "delta scaling."
        )
    },
)

data_xr.to_netcdf(path="../test_data/benchmark_ADA_spectral_albedo.nc")
