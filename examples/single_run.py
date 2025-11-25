#!/usr/bin/env python3
"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import matplotlib.pyplot as plt

from snicarfx.core import Session

simulation = Session("./src/snicarfx/inputs.yaml")
results = simulation.run()

plt.figure(figsize=(6, 4))
plt.plot(results.wavelengths, results.albedo)
plt.xlabel("Wavelengths (meters)")
plt.ylabel("Albedo")
plt.ylim(0, 1)
plt.grid()
