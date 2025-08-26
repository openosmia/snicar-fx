import matplotlib.pyplot as plt
from snicarfx.core import (
    ColumnProperties,
    SolarIrradiance,
    ModelInputs
)
from snicarfx.core.adding_doubling_solver import solve_adding_doubling

model_config = ModelInputs('./src/snicarfx/inputs.yaml')
column = ColumnProperties(model_config)
irradiance = SolarIrradiance(model_config) 

outputs_snicar = solve_adding_doubling(
      column, irradiance
)

plt.figure(figsize=(6, 4))
plt.plot(column.wavelengths, outputs_snicar.albedo)
plt.xlabel('Wavelengths (meters)')
plt.ylabel('Albedo')
plt.ylim(0,1)
plt.grid()