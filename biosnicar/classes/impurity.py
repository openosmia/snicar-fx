import os
import numpy as np
import xarray as xr
import biosnicar
import yaml

class Impurity:
    """Light absorbing impurity.

    Instances of Impurity are one discrete type of light absorbing impurity with
    a distinct set of optical properties.

    Attributes:
        name: name of impurity
        cfactor: concentration factor used to convert field measurements to model config (default=1)
        unit: the unit the concentration should be represented in (0 = ppb, 1 = cells/mL)
        conc: concentration of the impurity in each layer (in units of self.unit)
        file: name of netCDF file containing optical properties and size distribution
        impurity_properties: instance of opened file self.file
        mac: mass absorption coefficient (m2/kg or m2/cell)
        ssa: single scattering albedo
        g: asymmetry parameter

    """

    def __init__(self, file, coated, unit, name, conc, file_tag=None):

        self.name = name
        self.unit = unit
        self.conc = conc
        self.file = file
        self.coated = coated
        self.path = (
            str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
            + f'/data/OP_data/{file_tag}band/lap/'
            )
        
    def get_impurity_properties(self):
        self.impurity_properties = xr.open_dataset(self.path + self.file)
        
        if self.coated:
            mac_stub = "ext_cff_mss_ncl"
        elif (self.name == "ga") or (self.name == "sa"):
            mac_stub = "ext_xsc"
        else:
            mac_stub = "ext_cff_mss"

        self.mac = self.impurity_properties[mac_stub].values
        self.ssa = self.impurity_properties["ss_alb"].values
        self.g = self.impurity_properties["asm_prm"].values

        