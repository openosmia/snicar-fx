"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import matplotlib.pyplot as plt

from snicarfx import Session

# initialize the simulation
simulation = Session("./inputs_ex02.yaml")

# run the simulation
results = simulation.run(to_xarray=True)

# plot Bottom of Atmosphere (BOA) albedo, i.e. surface albedo
plt.figure()
plt.plot(results["wavelength"], results["albedo_boa"])
plt.xlabel("Wavelength")
plt.ylabel("Albedo")
