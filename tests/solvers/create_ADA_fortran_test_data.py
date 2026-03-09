"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

This file creates the Fortan ADA solver data necessary to test it
against its Python implementation.

"""

import shutil
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr

from snicarfx import Session
from snicarfx.core.solvers.multi_stream_solver import (
    _MultiStreamSolver,
)

# %% compile f90 code

# prepare for compilation
txt_file = Path("./fortran_CRTM_ADA_solver.f90.txt")
f90_file = txt_file.with_suffix("")
txt_file.rename(f90_file)

subprocess.run(
    "gfortran ./fortran_CRTM_ADA_solver.f90 -o run_ADA",
    shell=True,
    executable="/bin/bash",
)

# %% instanciate snicar-fx simulation

simulation = Session("../inputs_tests.yaml")

# %% declare ranges of parameter to use in runs of the ADA solver

# number of angles
# n_streams = simulation.config.SOLVER.N_STREAMS

# number of Fourier modes
# n_fouriers = simulation.config.SOLVER.N_FOURIER_MODES

# single scattering albedos
w_list = np.arange(0.2, 0.7, 0.1)

# asymmetry parameters
g_list = [0.32, 0.82]

# optical depths
t_od_list = np.arange(5, 220, 20)

# discrete wavelength indices
wavelength_index_list = [10, 40, 60, 80]

# allocate results of the Fortran solver
f90_results = np.zeros(
    (len(w_list), len(t_od_list), len(g_list), len(wavelength_index_list))
)

# %% run the Fortran solver

for wvl_enumarator, wavelength_index in enumerate(wavelength_index_list):

    # save g and cos values in files first because too bulky to pass
    # at run time
    for g in g_list:

        simulation.land_column.asm_prm[:, :] = g
        simulation.land_column.legendre_moments = (
            simulation.land_column.asm_prm[None, :, :]
            ** np.arange(simulation.land_column.n_expansion + 2)[:, None, None]
        )

        solver = _MultiStreamSolver(
            simulation.land_column,
            simulation.atmosphere_column,
            simulation.solar_irradiance,
            simulation.config.SOLVER,
        )
        solver.reset_state(m=simulation.config.SOLVER.N_FOURIER_MODES - 1)
        solver.set_phase_matrices()

        ff_wvl = solver.ff[:, :, :, wavelength_index]
        bb_wvl = solver.bb[:, :, :, wavelength_index]

        # save cos weight and cos angle as computed by the python
        # version
        if wvl_enumarator == 0:
            np.savetxt(
                "./cos_weight.csv",
                solver.cos_weight,
                delimiter=",",
            )
            np.savetxt(
                "./cos_angle.csv",
                solver.cos_angle,
                delimiter=",",
            )

        # save ff and bb to be read in f90 code
        np.savetxt(
            f"./ff_g{g}.csv",
            ff_wvl[:, :, 0],
            delimiter=",",
        )

        np.savetxt(
            f"./bb_g{g}.csv",
            bb_wvl[:, :, 0],
            delimiter=",",
        )

    # re-initialize snicar-fx session
    simulation = Session("../inputs_tests.yaml")

    # extract values required by the Fortran solver
    cos_sun = solver.cos_sun
    solar_irradiance = solver.solar_irradiance[wavelength_index]

    # pass other simpler arguments now, at run time
    for t_od_enumerator, t_od in enumerate(t_od_list):
        for w_enumerator, w in enumerate(w_list):
            for g_enumerator, g in enumerate(g_list):

                # re-initialize snicar-fx session
                simulation = Session("../inputs_tests.yaml")

                # update python variables
                simulation.land_column.ss_alb[:, :] = w
                simulation.land_column.tau[:, :] = t_od
                simulation.land_column.asm_prm[:, :] = g
                simulation.land_column.legendre_moments = (
                    simulation.land_column.asm_prm[None, :, :]
                    ** np.arange(simulation.land_column.n_expansion + 2)[:, None, None]
                )

                # initiate the python ADA solver to extract delta
                # scaled variables and feed them to the fortran ADA
                # solver, which doesn't have delta scaling
                solver = _MultiStreamSolver(
                    simulation.land_column,
                    simulation.atmosphere_column,
                    simulation.solar_irradiance,
                    simulation.config.SOLVER,
                )
                solver.reset_state(m=simulation.config.SOLVER.N_FOURIER_MODES - 1)
                solver.set_phase_matrices()

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
                s_level_rad_up_f90_k0 = np.loadtxt("./s_Level_Rad_UP_temp.csv")

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

                simulation.land_column.ss_alb[:, :] = w
                simulation.land_column.tau[:, :] = t_od
                simulation.land_column.asm_prm[:, :] = g
                simulation.land_column.legendre_moments = (
                    simulation.land_column.asm_prm[None, :, :]
                    ** np.arange(simulation.land_column.n_expansion + 2)[:, None, None]
                )

                # uncomment to check py/f90 match directly here
                # py_results = solve_multi_stream_rt(
                #     simulation.land_column,
                #     simulation.atmosphere_column,
                #     simulation.solar_irradiance,
                #     simulation.config.SOLVER,
                # )
                # print(np.abs(py_results["albedo_boa"][wavelength_index] - albedo_f90))
                # reflectance_f90 = (np.pi * s_level_rad_up_f90_k0) / (
                #     solar_irradiance * cos_sun
                # )
                # print(
                #     w,
                #     t_od,
                #     g,
                #     np.abs(
                #         py_results["directional_reflectance_boa"][-1, wavelength_index]
                #         - reflectance_f90[-1]
                #     ),
                # )

# %% save results

data_xr = xr.Dataset(
    data_vars={
        "albedo": (
            ["w", "t_od", "g", "wavelength_index"],
            f90_results,
            {
                "long_name": "Spectral albedo",
                "description": "Surface albedo computed with the Fortran ADA multistream solver",
                "units": "dimensionless",
            },
        )
    },
    coords={
        "w": (
            "w",
            w_list,
            {"long_name": "Single scattering albedo", "units": "dimensionless"},
        ),
        "t_od": (
            "t_od",
            t_od_list,
            {"long_name": "Optical depth", "units": "dimensionless"},
        ),
        "g": (
            "g",
            g_list,
            {"long_name": "Asymmetry parameter", "units": "dimensionless"},
        ),
        "wavelength_index": (
            "wavelength_index",
            wavelength_index_list,
            {"long_name": "Discrete wavelength index", "units": "index"},
        ),
    },
    attrs={
        "title": "Benchmark spectral albedo from Fortran ADA solver",
        "summary": (
            "This NetCDF file contains spectral albedo results computed with "
            "the Fortran version of the ADA radiative transfer solver. "
            "The results are intended for validation against the Python implementation within snicar-fx."
        ),
        "creation_date": datetime.utcnow().isoformat(),
        "model_name": "snicar-fx",
        "model_version": version("snicarfx"),
        "model_url": "https://github.com/openosmia/snicar-fx",
    },
)

data_xr.to_netcdf(path="../test_data/benchmark_ADA_spectral_albedo.nc")

# %% remove temporary files


files_to_remove = [
    "./ff_g*.csv",
    "./bb_g*.csv",
    "./cos_*.csv",
    "./*.mod",
    "./run_ADA",
    "s_*.csv",
]

for pattern in files_to_remove:
    for file_path in Path(".").glob(pattern):
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)

# %% store f90 file back in txt

f90_file.rename(txt_file)
