"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

This file creates the Fortan ADA solver data necessary to test it
against its Python implementation.

"""

import subprocess

import numpy as np
import xarray as xr

from snicarfx.core import Session
from snicarfx.core.solvers.multi_stream_solver import (
    solve_multi_stream_rt,
    _MultiStreamSolver,
)


# %% instanciate snicar-fx simulation

simulation = Session("./inputs_tests.yaml")

# %% declare ranges of parameter to use in runs of the ADA solver

# number of angles
n_streams = simulation.config.SOLVER.N_STREAMS

# single scattering albedos
w_list = np.arange(0.2, 0.7, 0.1)

# asymmetry parameters
g_list = [0.32, 0.82]

# optical depths
t_od_list = np.arange(5, 220, 20)

# discrete wavelength indices
wavelength_index_list = [10, 40, 60, 80]

# calculate Legendre polynomials required by the Fortran solver
# nodes, weights = np.polynomial.legendre.leggauss(n_streams * 2)
# cos_angle = nodes[n_streams:]
# cos_weight = weights[n_streams:]

# allocate results of the Fortran solver
f90_results = np.zeros(
    (len(w_list), len(t_od_list), len(g_list), len(wavelength_index_list))
)

# %% run the Fortran solver

for wvl_enumarator, wavelength_index in enumerate(wavelength_index_list):

    # save g in files first because too big to pass at run time
    for g in g_list:

        simulation.land_column.asm_prm[:, :] = g
        simulation.land_column.legendre_moments = (
            simulation.land_column.asm_prm[None, :, :]
            ** np.arange(simulation.land_column.n_expansion)[:, None, None]
        )

        solver = _MultiStreamSolver(
            simulation.land_column,
            simulation.atmosphere_column,
            simulation.solar_irradiance,
            simulation.config.SOLVER.OUTPUT_LEVELS,
            n_streams,
        )

        ff_wvl = solver.ff[:, :, :, wavelength_index]
        bb_wvl = solver.bb[:, :, :, wavelength_index]

        np.savetxt(
            f"./test_data/test_data/ff_g{g}.csv",
            ff_wvl[:, :, 0],
            delimiter=",",
        )

        np.savetxt(
            f"./test_data/test_data/bb_g{g}.csv",
            bb_wvl[:, :, 0],
            delimiter=",",
        )

    # re-initialize snicar-fx session
    simulation = Session("./inputs_tests.yaml")

    # extract values required by the Fortran solver
    cos_sun = solver.cos_sun
    solar_irradiance = solver.solar_irradiance[wavelength_index]

    # pass other simpler arguments now, at run time
    for t_od_enumerator, t_od in enumerate(t_od_list):
        for w_enumerator, w in enumerate(w_list):
            for g_enumerator, g in enumerate(g_list):

                # update python variables
                simulation.land_column.ss_alb[:, :] = w
                simulation.land_column.tau[:, :] = t_od
                simulation.land_column.asm_prm[:, :] = g
                simulation.land_column.legendre_moments = (
                    simulation.land_column.asm_prm[None, :, :]
                    ** np.arange(simulation.land_column.n_expansion)[:, None, None]
                )

                # initiate the python ADA solver to extract delta
                # scaled variables and feed them to the fortran ADA
                # solver, which doesn't have delta scaling
                solver = _MultiStreamSolver(
                    simulation.land_column,
                    simulation.atmosphere_column,
                    simulation.solar_irradiance,
                    simulation.config.SOLVER.OUTPUT_LEVELS,
                    n_streams,
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
                    "./test_data/test_data/s_Level_Rad_UP_temp.csv"
                )

                # compute albedo of the fortran ADA solver to compare
                # with the python version in tests
                albedo_f90 = (
                    2
                    * np.pi
                    * np.sum(
                        s_level_rad_up_f90_k0 * solver.cos_angle * solver.cos_weight
                    )
                    / (solar_irradiance * cos_sun)
                )

                # store albedo results
                f90_results[
                    w_enumerator, t_od_enumerator, g_enumerator, wvl_enumarator
                ] = albedo_f90

                # uncomment to check py/f90 match directly here
                # py_results = solve_multi_stream_rt(
                #     simulation.land_column,
                #     simulation.atmosphere_column,
                #     simulation.solar_irradiance,
                #     simulation.config.SOLVER.OUTPUT_LEVELS,
                #     n_streams,
                # )
                # print(np.abs(py_results["albedo_boa"][wavelength_index] - albedo_f90))

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

data_xr.to_netcdf(path=f"./test_data/benchmark_ADA_spectral_albedo.nc")
