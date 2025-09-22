#!/usr/bin/env python3
"""
This file is part of the snicar-fx software package. 

https://github.com/openosmia/snicar-fx 


Author(s)
---------
snicar-fx development team

"""

import matplotlib.pyplot as plt

from snicarfx import snicarfx_wrapper

input_file = './src/snicarfx/inputs.yaml'

outputs = snicarfx_wrapper.run_two_stream(input_file)

plt.figure(figsize=(6, 4))
plt.plot(outputs.wavelengths, outputs.albedo)
plt.xlabel('Wavelengths (meters)')
plt.ylabel('Albedo')
plt.ylim(0,1)
plt.grid()