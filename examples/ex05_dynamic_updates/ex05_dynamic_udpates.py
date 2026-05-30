"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import matplotlib.pyplot as plt

from snicarfx import Session

# initialize the simulation
simulation = Session("../ex02_multistream_surface/inputs_ex02.yaml")

# just a random set of decreasing/increasing SSA/BC concentration
ssa_first_layer = [2, 1, 0.5]
bc_concentration_first_layer = [1, 50, 100]

plt.figure()

for ssa, bc_concentration in zip(
    ssa_first_layer, bc_concentration_first_layer, strict=True
):
    # update specific surface area and black carbon concentration of
    # the first layer, without validating changes
    simulation.update_land(
        {
            "SPECIFIC_SURFACE_AREA": [ssa, 1.0, 1.0],
            "LIGHT_ABSORBING_PARTICLES": {
                "BC": {"CONC": [bc_concentration, 0.0, 0.0]},
            },
        },
        validate=False,
    )

    # run the simulation
    results = simulation.run(to_xarray=True)

    # plot Bottom of Atmosphere (BOA) albedo for each update
    plt.plot(
        results["wavelength"],
        results["albedo_boa"],
        label=f"SSA1: {ssa}"
        + r" m$^2$kg$^{-1}$,"
        + f" BC: {bc_concentration} "
        + r"kgkg$_{ice}$$^{-1}$",
    )


plt.xlabel("Wavelength")
plt.ylabel("Albedo")
plt.legend()
