#!/usr/bin/python
"""Runs benchmarking and fuzzing tests on BioSNICAR.

To run configure these tests, update the values in conftest.py
Then navigate to the tests folder and run

`pytest .`

The tests will automatically run - green dots indicate tests
passing successfully. A plot of N random spectra pairs will
be saved to the /tests folder.

The fuzzer exists to run snocar with a wide range of input variables
to check that no combinations break the code. It is quite memory intensive
to fuzz over a very large parameter space. 10^3 runs is ok on a decent
spec laptop. I have divided the fuzzer into two separate functions. One
has coverage for "config" variables that set up the radiative transfer
e.g. direct vs diffuse, approximation type, etc. The other is more for
conditions of the ice column, e.g. density, effective radius, LAPs.

To toggle the fuzzer on/off change the value of "fuzz" in conftest.py

"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from snicarfx.rt_solvers.adding_doubling_solver import adding_doubling_solver
from snicarfx.classes import ColumnProperties, SolarIrradiance, ModelConfig


def test_AD_solver(new_benchmark_ad, input_file):
    """Tests AD solver against SNICAR_ADv4 benchmark.

    This func generates a new file - py_benchmark_data.csv - that contains
    spectral and broadband albedo simulated by BioSNICAR for a range of input
    configurations. The same set of simulations was also run using a previously
    published version of the SNICAR code written in Matlab by Chloe Whicker at
    University of Michigan and run on the UMich server. This function
    only creates the equivalent dataset using BioSNICAR, it doesn't compare the two.

    Equivalence between the Python and Matlab model configuration is controlled by
    a call to match_matlab_config(). This function can be toggled off by setting
    new_benchmark_ad to False in conftest.py.

    Args:
        new_benchmark_ad: Boolean toggling this function on/off

    Returns:
        None but saves py_benchmark_data.csv to ./tests/test_data/

    """
    if new_benchmark_ad:

        model_config = ModelConfig("./tests/inputs_tests.yaml")
        column = ColumnProperties(model_config)
        irradiance = SolarIrradiance(model_config)

        column = match_matlab_config(column)

        lyrList = [0, 1]
        densList = [400, 500, 600, 700, 800]
        reffList = [200, 400, 600, 800, 1000]
        zenList = [30, 40, 50, 60]
        bcList = [500, 1000, 2000]
        dzList = [
            [0.02, 0.04, 0.06, 0.08, 0.1],
            [0.04, 0.06, 0.08, 0.10, 0.15],
            [0.05, 0.10, 0.15, 0.2, 0.5],
            [0.15, 0.2, 0.25, 0.3, 0.5],
            [0.5, 0.5, 0.5, 1, 10],
        ]

        ncols = (
            len(lyrList)
            * len(densList)
            * len(reffList)
            * len(zenList)
            * len(bcList)
            * len(dzList)
        )

        assert ncols == 3000

        specOut = np.zeros(shape=(ncols, 481))
        counter = 0
        for layer_type in lyrList:
            for density in densList:
                for reff in reffList:
                    for zen in zenList:
                        for bc in bcList:
                            for dz in dzList:
                                # calculate irradiance
                                irradiance.solzen = zen
                                irradiance.calculate_irradiance()

                                # calculate column ssa, g, mac
                                column.thickness = dz
                                column.layer_type = [layer_type] * len(column.thickness)
                                column.rho = [density] * len(column.thickness)

                                snow_idx = np.where(np.array(column.layer_type) == 0)[0]
                                ice_idx = np.where(np.array(column.layer_type) != 0)[0]

                                for i in snow_idx:
                                    file_ssps = str(
                                        "./tests/test_data/ice_spherical_grains_BH83/"
                                        + f"ice_{column.rf_type}/ice_{column.rf_type}_"
                                        + "{}.nc".format(str(reff).rjust(4, "0"))
                                    )

                                    with xr.open_dataset(file_ssps) as ssps:
                                        column.ss_alb[i, :] = ssps["ss_alb"].values
                                        column.ext_cff[i, :] = ssps[
                                            "ext_cff_mss"
                                        ].values
                                        column.asm_prm[i, :] = ssps["asm_prm"].values
                                        column.tau[i, :] = (
                                            column.layer_mass[i] * column.ext_cff[i, :]
                                        )

                                for i in ice_idx:
                                    file_ssps = str(
                                        "./tests/test_data/bubbly_ice_files_BH83/"
                                        + "bbl_{}.nc".format(str(reff).rjust(4, "0"))
                                    )
                                    with xr.open_dataset(file_ssps) as ssps:
                                        column.asm_prm[i, :] = ssps["asm_prm"].values
                                        sca_cff_vlm_air_bbl = ssps["sca_cff_vlm"].values
                                        vlm_frac_air = 1 - column.density[i] / 917
                                        scattering_cff = (
                                            sca_cff_vlm_air_bbl
                                            * vlm_frac_air
                                            / column.density[i]
                                        )
                                        abs_cff = (
                                            (4 * np.pi * column.ref_idx_im)
                                            / (column.wavelengths)
                                            / 917
                                        )
                                        column.ext_cff[i, :] = scattering_cff + abs_cff
                                        column.ss_alb[i, :] = (
                                            scattering_cff / column.ext_cff[i, :]
                                        )
                                        column.tau[i, :] = (
                                            column.layer_mass[i] * column.ext_cff[i, :]
                                        )

                                column.lap_concentrations[:, 0] = [
                                    bc,
                                    bc,
                                    bc,
                                    bc,
                                    bc,
                                ]

                                column.add_laps_to_column_ops()

                                # solve RTE
                                outputs = adding_doubling_solver(column, irradiance)

                                specOut[counter, 0:480] = outputs.albedo
                                specOut[counter, 480] = outputs.BBA
                                counter += 1

        np.savetxt("./tests/test_data/py_benchmark_data.csv", specOut, delimiter=",")

    else:
        pass

    return


def test_AD_solver_clean(new_benchmark_ad_clean, input_file):
    """Tests Toon solver against SNICAR_ADv4 benchmark for impurity-free ice.

    This func generates a new file - py_benchmark_data_clean.csv - that contains
    spectral and broadband albedo simulated by BioSNICAR for a range of input
    configurations. The same set of simulations was also run using a previously
    published version of the SNICAR code written in Matlab by Chloe Whicker at
    University of Michigan and run on the UMich server. This function
    only creates the equivalent dataset using BioSNICAR, it doesn't compare the two.
    The difference between this function and test_v4 is that no impurities are included
    in the model configuration.

    Equivalence between the Python and Matlab model configuration is controlled by
    a call to match_matlab_config(). This function can be toggled off by setting
    new_benchmark_clean to False in conftest.py.

    Args:
        new_benchmark_clean: Boolean toggling this function on/off

    Returns:
        None but saves py_benchmark_data_clean.csv to ./tests/test_data/

    """

    if new_benchmark_ad_clean:

        model_config = ModelConfig("./tests/inputs_tests.yaml")
        column = ColumnProperties(model_config)
        irradiance = SolarIrradiance(model_config)

        column = match_matlab_config(column)

        print(
            "generating benchmark data using params equivalent to snicarv4 (AD solver)"
        )

        lyrList = [0, 1]
        densList = [400, 500, 600, 700, 800]
        reffList = [200, 400, 600, 800, 1000]
        zenList = [30, 40, 50, 60]
        bcList = [0]
        dzList = [
            [0.02, 0.04, 0.06, 0.08, 0.1],
            [0.04, 0.06, 0.08, 0.10, 0.15],
            [0.05, 0.10, 0.15, 0.2, 0.5],
            [0.15, 0.2, 0.25, 0.3, 0.5],
            [0.5, 0.5, 0.5, 1, 10],
        ]

        ncols = (
            len(lyrList)
            * len(densList)
            * len(reffList)
            * len(zenList)
            * len(bcList)
            * len(dzList)
        )

        specOut = np.zeros(shape=(ncols, 481))
        counter = 0
        for layer_type in lyrList:
            for density in densList:
                for reff in reffList:
                    for zen in zenList:
                        for bc in bcList:
                            for dz in dzList:

                                # calculate irradiance
                                irradiance.solzen = zen
                                irradiance.calculate_irradiance()

                                # calculate column ssa, g, mac
                                column.thickness = dz
                                column.layer_type = [layer_type] * len(column.thickness)
                                column.rho = [density] * len(column.thickness)

                                snow_idx = np.where(np.array(column.layer_type) == 0)[0]
                                ice_idx = np.where(np.array(column.layer_type) != 0)[0]

                                for i in snow_idx:
                                    file_ssps = str(
                                        "./tests/test_data/ice_spherical_grains_BH83/"
                                        + f"ice_{column.rf_type}/ice_{column.rf_type}_"
                                        + "{}.nc".format(str(reff).rjust(4, "0"))
                                    )

                                    with xr.open_dataset(file_ssps) as ssps:
                                        column.ss_alb[i, :] = ssps["ss_alb"].values
                                        column.ext_cff[i, :] = ssps[
                                            "ext_cff_mss"
                                        ].values
                                        column.asm_prm[i, :] = ssps["asm_prm"].values
                                        column.tau[i, :] = (
                                            column.layer_mass[i] * column.ext_cff[i, :]
                                        )

                                for i in ice_idx:
                                    file_ssps = str(
                                        "./tests/test_data/bubbly_ice_files_BH83/"
                                        + "bbl_{}.nc".format(str(reff).rjust(4, "0"))
                                    )
                                    with xr.open_dataset(file_ssps) as ssps:
                                        column.asm_prm[i, :] = ssps["asm_prm"].values
                                        sca_cff_vlm_air_bbl = ssps["sca_cff_vlm"].values
                                        vlm_frac_air = 1 - column.density[i] / 917
                                        scattering_cff = (
                                            sca_cff_vlm_air_bbl
                                            * vlm_frac_air
                                            / column.density[i]
                                        )
                                        abs_cff = (
                                            (4 * np.pi * column.ref_idx_im)
                                            / (column.wavelengths)
                                            / 917
                                        )
                                        column.ext_cff[i, :] = scattering_cff + abs_cff
                                        column.ss_alb[i, :] = (
                                            scattering_cff / column.ext_cff[i, :]
                                        )
                                        column.tau[i, :] = (
                                            column.layer_mass[i] * column.ext_cff[i, :]
                                        )

                                column.lap_concentrations[:, 0] = [
                                    bc,
                                    bc,
                                    bc,
                                    bc,
                                    bc,
                                ]

                                column.add_laps_to_column_ops()

                                # solve RTE
                                outputs = adding_doubling_solver(column, irradiance)

                                specOut[counter, 0:480] = outputs.albedo
                                specOut[counter, 480] = outputs.BBA
                                counter += 1

        np.savetxt(
            "./tests/test_data/py_benchmark_data_clean.csv", specOut, delimiter=","
        )

    else:
        pass

    return


def test_realistic_bba_ad(get_matlab_data, get_python_data):
    """Tests that BBA values are never >1 or <0.

    Simple test that ensures broadband albedo predicted by model never
    goes outside valid range of 0-1.

    Args:
        get_matlab_data: matlab-snicar-generated csv file of spectral and broadband albedo
        get_python_data: BioSNICAR generated csv file of spectral and broadband albedo

    Returns:
        None

    Raises:
        tests fail if the two datasets differ in length
        tests fail if any BBA values in matlab data are <0 or >1
        tests fail if any BBA values in python data are <0 or >1
    """
    # are the values predicted by the model always physical (i.e. between 0-1)
    # do the files have the right shape and size?

    mat = get_matlab_data
    py = get_python_data

    bb_py = py.loc[:, 481]
    bb_mat = mat.loc[:, 481]

    assert len(bb_py) == len(bb_mat)
    assert bb_py[bb_py > 1].count() == 0 and bb_py[bb_py < 0].count() == 0
    assert bb_mat[bb_mat > 1].count() == 0 and bb_mat[bb_mat < 0].count() == 0


def test_compare_pyBBA_to_matBBA(get_matlab_data, get_python_data, set_tolerance):
    """Tests that BBA values match between BioSNICAR data and the benchmark.

    Element-wise comparison between BBAs in equivalent positions in the Matlab benchmark
    dataset and the newly generated BioSNICAR dataset for the AD solver.

    Args:
        get_matlab_data: matlab-snicar-generated csv file of spectral and broadband albedo
        get_python_data: BioSNICAR generated csv file of spectral and broadband albedo
        set_tolerance: threshold error for BBAs to be considered equal

    Returns:
        None

    Raises:
        tests fail if the difference between any pair of BBA values exceeds set_tolerance

    """

    mat = get_matlab_data
    py = get_python_data
    tol = set_tolerance
    bb_py = py.loc[:, 481]
    bb_mat = mat.loc[:, 481]
    error = np.array(abs(bb_mat - bb_py))
    print(np.max(error))
    assert len(error[error > tol]) == 0


def test_compare_pyBBA_to_matBBA_clean(
    get_matlab_data_clean, get_python_data_clean, set_tolerance
):
    """Tests that BBA values match between BioSNICAR data and the benchmark for clean ice.

    Element-wise comparison between BBAs in equivalent positions in the Matlab benchmark
    dataset and the newly generated BioSNICAR dataset for the AD solver with no impurities.

    Args:
        get_matlab_data: matlab-snicar-generated csv file of spectral and broadband albedo
        get_python_data: BioSNICAR generated csv file of spectral and broadband albedo
        set_tolerance: threshold error for BBAs to be considered equal

    Returns:
        None

    Raises:
        tests fail if the difference between any pair of BBA values exceeds set_tolerance

    """

    mat = get_matlab_data_clean
    py = get_python_data_clean
    tol = set_tolerance
    bb_py = py.loc[:, 481]
    bb_mat = mat.loc[:, 481]
    error = np.array(abs(bb_mat - bb_py))
    assert len(error[error > tol]) == 0


def test_compare_pyspec_to_matspec_ad(get_matlab_data, get_python_data, set_tolerance):
    """Tests that spectral albedo values match between BioSNICAR data and the AD benchmark.

    Element-wise comparison between spectral albedo in equivalent positions in the Matlab benchmark
    dataset and the newly generated BioSNICAR dataset for the AD solver. Albedo is compared
    wavelength by wavelength for each column in the datasets.

    Args:
        get_matlab_data: matlab-snicar-generated csv file of spectral and broadband albedo
        get_python_data: BioSNICAR generated csv file of spectral and broadband albedo
        set_tolerance: threshold error for BBAs to be considered equal

    Returns:
        None

    Raises:
        tests fail if the difference between any pair of albedo values exceeds set_tolerance

    """

    mat = get_matlab_data
    py = get_python_data
    tol = set_tolerance
    bb_spec = py.loc[:, :480]
    bb_spec = mat.loc[:, :480]
    error = np.array(abs(bb_spec - bb_spec))
    assert len(error[error > tol]) == 0


def test_compare_pyspec_to_matspec_clean(
    get_matlab_data_clean, get_python_data_clean, set_tolerance
):
    """Tests that spectral albedo values match between BioSNICAR data and the Toon benchmark.

    Element-wise comparison between spectral albedo in equivalent positions in the Matlab benchmark
    dataset and the newly generated BioSNICAR dataset for the AD solver with no impurities.
    Albedo is compared wavelength by wavelength for each column in the datasets.

    Args:
        get_matlab_data_clean: matlab-snicar-generated csv file of spectral and broadband albedo
        get_python_data_clean: BioSNICAR generated csv file of spectral and broadband albedo
        set_tolerance: threshold error for BBAs to be considered equal

    Returns:
        None

    Raises:
        tests fail if the difference between any pair of albedo values exceeds set_tolerance

    """

    mat = get_matlab_data_clean
    py = get_python_data_clean
    tol = set_tolerance
    bb_spec = py.loc[:, :480]
    bb_spec = mat.loc[:, :480]
    error = np.array(abs(bb_spec - bb_spec))
    assert len(error[error > tol]) == 0


def test_plot_random_spectra_pairs(get_matlab_data, get_python_data, get_n_spectra):
    """Plots random selection of N spectra pairs and saves to /tests/test_data.

    Args:
        get_matlab_data: matlab-generated csv file of spectral and broadband albedo
        get_python_data: python-generated csv file of spectral and broadband albedo
        get_n_spectra: number of pairs of spectral albedo to plot

    Returns:
        None but saves py_mat_comparison.png to /tests/test_data/
    """

    mat = get_matlab_data
    py = get_python_data
    n_spec = get_n_spectra
    idxs = random.sample(range(0, py.shape[0]), n_spec)

    wavelength = np.arange(200, 5000, 10)
    py_spectra = py.iloc[0:-1, idxs]
    mat_spectra = mat.iloc[0:-1, idxs]

    plt.plot(wavelength, py_spectra)
    plt.plot(wavelength, mat_spectra, linestyle=None, marker="x")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Albedo")
    plt.title("solid lines = Python\ncrosses = Matlab")

    plt.savefig("./tests/test_data/py_mat_comparison.png")


def match_matlab_config(column):
    """Ensures model config is equal to the Matlab version used to generate benchmark data.

    This function resets values in instances of Ice, Illumination and ModelConfig to ensure
    equivalence between BioSNICAR and the Matlab code used to generate the benchmark data.
    Also ensures all vars have correct length, and re-executes the class functions in Ice and
    Illumination that update refractive indices and at-surface irradiance.

    Args:
        column: instance of Ice class
        irradiance: instance of SolarIrradiance class
        model_config: instance of ModelConfig class

    Returns:



    """

    column.ref_idx_im = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).im_Pic16.values
    column.ref_idx_re = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).re_Pic16.values
    column.fl_r_dif_b = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fb_ice_Pic16.values
    column.fl_r_dif_a = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fa_ice_Pic16.values

    return column


@pytest.mark.parametrize("dir", [0, 1])
@pytest.mark.parametrize("aprx", [1, 2, 3])
@pytest.mark.parametrize("inc", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("ref", ["Wrn84", "Wrn08", "Pic16"])
def test_config_fuzzer(dir, aprx, inc, ref, fuzz, input_file):
    """Checks model runs correctly with range of input value combinations.

    Fuzzer checks that model functions correctly across range of configurations.
    This fuzzer specifically checks rtm config and illumination parameters. The
    range of values is set in the parameterize decorators on this function and can
    be adjusted to test for specific failures or to increase test coverage. The
    defaults are designed to balance coverage with execution time. This func can be
    toggled off by setting fuzz to fase in conftest.py

    Args:
        dir: Boolean toggling between direct and diffuse irradiance
        aprx: choice of two-stream approximation
        inc: choice of spectral distribution of incoming irradiance
        ref: chocie of refractive indices (Warren 1984, Warren 2008, Picard 2016)
        fuzz: boolean toggling this fuxxing func on/off

    Returns:
        None

    Raises:
        tests fail if snicar throws an exception with a particular configuration
    """

    if fuzz:
        model_config = ModelConfig("./tests/inputs_tests.yaml")
        column = ColumnProperties(model_config)
        irradiance = SolarIrradiance(model_config)

        column = match_matlab_config(column)

        irradiance.direct = dir
        column.rf_type = ref
        irradiance.incoming = inc
        irradiance.calculate_irradiance()

        column.calculate_column_ops_clean()

        column.add_laps_to_column_ops()

        outputs_ad = adding_doubling_solver(column, irradiance)

    else:
        pass

    return


@pytest.mark.parametrize("ssa", [5, 10, 20])
@pytest.mark.parametrize("rho", [400, 600, 800])
@pytest.mark.parametrize("zen", [50, 60, 70])
@pytest.mark.parametrize("dust", [0, 50000])
@pytest.mark.parametrize("soot", [0, 50000])
def test_var_fuzzer(ssa, rho, zen, dust, soot, fuzz, input_file):
    """Checks model runs correctly with range of input value combinations.

    Fuzzer checks that model functions correctly across range of configurations.
    This fuzzer spoecifically checks input parameters including ice physical properties.
    The range of values is set in the parameterize decorators on this function and can
    be adjusted to test for specific failures or to increase test coverage. The
    defaults are designed to balance coverage with execution time. This func can be
    toggled off by setting fuzz to fase in conftest.py

    Args:
        ssa: specific surface area
        rho: density of ice layer (kg/m3)
        zen: zenith angle of direct beam (degrees from vertical)
        dust: concentration of mineral dust in each layer of the model (ppb)
        algae: concentration of glacir algae in each layer of the model (ppb)

    Returns:
        None

    Raises:
        tests fail if snicar throws an exception with a particular configuration
    """

    if fuzz:
        model_config = ModelConfig("./tests/inputs_tests.yaml")
        column = ColumnProperties(model_config)
        irradiance = SolarIrradiance(model_config)

        column = match_matlab_config(column)

        irradiance.solzen = zen
        irradiance.calculate_irradiance()

        column.ssa = [ssa] * len(column.thickness)
        column.density = [rho] * len(column.thickness)
        column.calculate_column_ops_clean()

        # add impurities to the ice column

        # to change:
        column.lap_concentrations = np.vstack(
            [[np.ones(column.nbr_lyr) * soot], [np.ones(column.nbr_lyr) * dust]]
        ).T

        lap1 = xr.open_dataset(
            "./data/optical_properties/light_absorbing_particles/mie_sot_ChC90_dns_1317.nc"
        )
        lap2 = xr.open_dataset(
            "./data/optical_properties/light_absorbing_particles/dust_balkanski_central_size1.nc"
        )

        column.lap_ext_cff = np.vstack(
            [[lap1["ext_cff_mss"].values], [lap2["ext_cff_mss"].values]]
        )

        column.lap_ss_alb = np.vstack(
            [[lap1["ss_alb"].values], [lap2["ss_alb"].values]]
        )

        column.lap_asm_prm = np.vstack(
            [[lap1["asm_prm"].values], [lap2["asm_prm"].values]]
        )

        column.add_laps_to_column_ops()

        outputs_ad = adding_doubling_solver(column, irradiance)

    else:
        pass

    return


if __name__ == "__main__":
    pass
