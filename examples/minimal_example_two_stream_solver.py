#!/usr/bin/env python3

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