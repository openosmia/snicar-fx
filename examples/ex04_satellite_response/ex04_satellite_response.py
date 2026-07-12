"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import matplotlib.pyplot as plt

from snicarfx import Session

# initialize the simulation
simulation = Session("./inputs_ex04.yaml")

# run the simulation
results = simulation.run(to_xarray=True)

# plot Bottom of Atmosphere (BOA) and Top of Atmosphere (TOA) albedo
plt.figure()
plt.plot(
    results["wavelength"],
    results["albedo_boa"],
    label="BOA albedo",
    marker=".",
)
plt.plot(
    results["wavelength"],
    results["albedo_toa"],
    label="TOA albedo",
    marker=".",
)
plt.xlabel("Wavelength")
plt.ylabel("Albedo")
plt.legend()
